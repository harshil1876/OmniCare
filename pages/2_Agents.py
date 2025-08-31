import os
import re
import json
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain.schema import OutputParserException

# LLMs
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Vector store (Chroma) + Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- Setup ----------
st.set_page_config(page_title="OmniCore Hybrid Agents", page_icon="🤖", layout="wide")
load_dotenv()

DATA_PATH = "data/sales_data.csv"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AUDIT_LOG_FILE = "audit_logs.json"

# ---------- Utilities ----------

def safe_extract_output(res) -> str:
    """
    Robustly extract agent output regardless of backend return schema.
    Handles dicts with 'output' or 'text', lists, or raw strings.
    """
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        if "output" in res and isinstance(res["output"], str):
            return res["output"]
        if "text" in res and isinstance(res["text"], str):
            return res["text"]
        # langchain agents often put final text under 'output' or 'output_text'
        if "output_text" in res and isinstance(res["output_text"], str):
            return res["output_text"]
        # Handle langchain agent response structure
        if "result" in res and isinstance(res["result"], str):
            return res["result"]
        # Fallback to stringify dict
        return json.dumps(res, ensure_ascii=False)
    # Fallback to str
    return str(res)

def stringify_exception(e: Exception, limit: int = 140) -> str:
    msg = f"{type(e).__name__}: {str(e)}"
    return (msg[:limit] + "...") if len(msg) > limit else msg

def standardize_agent_response(raw_output: str, query: str, execution_path: str) -> str:
    """Ensure all agent responses follow consistent format"""
    if not raw_output or len(raw_output.strip()) < 5:
        return raw_output
    
    # Skip if already well-formatted
    if "**" in raw_output or "|" in raw_output:
        return raw_output
    
    # Check for raw number output
    lines = raw_output.strip().split('\n')
    try:
        if len(lines) == 1:
            float(raw_output.strip())
            return f"""**Analysis Result:**

The answer to "{query}" is: **{raw_output.strip()}**

**Context:**
- Method: {execution_path.title()} processing
- Result type: Single aggregate value"""
    except ValueError:
        pass
    
    # Enhance minimal responses
    if (len(lines) <= 3 and 
        not any(word in raw_output.lower() for word in ['analysis', 'result', 'interpretation'])):
        return f"""**Analysis Result:**

{raw_output}

**Summary:** Analysis completed successfully via {execution_path.title()}."""
    
    return raw_output

def enhance_query_for_consistency(query: str, complexity: str) -> str:
    """Add formatting requirements to queries"""
    format_req = """

Please format your response professionally:
- Explain what you're calculating
- Present results clearly (tables for groups, context for single values)
- Add brief interpretation"""
    
    return f"{query}{format_req}"

# ---------- Audit & Tracking ----------
class QueryTracker:
    def __init__(self):
        if 'query_stats' not in st.session_state:
            st.session_state.query_stats = {
                'total_queries': 0,
                'ollama_calls': 0,
                'gemini_calls': 0,
                'fallback_hits': 0,
                'pandas_agent': 0,
                'audit_logs': []
            }
    
    def log_query(self, query: str, complexity: str, execution_path: str, success: bool, response_time: float):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'complexity': complexity,
            'execution_path': execution_path,
            'success': success,
            'response_time_ms': response_time * 1000,
        }
        
        st.session_state.query_stats['total_queries'] += 1
        st.session_state.query_stats['audit_logs'].append(log_entry)
        
        # Update path counters
        if execution_path == 'ollama':
            st.session_state.query_stats['ollama_calls'] += 1
        elif execution_path == 'gemini':
            st.session_state.query_stats['gemini_calls'] += 1
        elif execution_path == 'fallback':
            st.session_state.query_stats['fallback_hits'] += 1
        elif execution_path == 'pandas-agent':
            st.session_state.query_stats['pandas_agent'] += 1
            
        # Persist to file (optional, best-effort)
        try:
            with open(AUDIT_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception:
            pass  # Silent fail for audit logging
    
    def get_stats(self) -> Dict:
        return st.session_state.query_stats

# Global tracker
tracker = QueryTracker()

# ---------- Schema Detection & Management ----------
class SchemaManager:
    def __init__(self):
        if 'detected_schema' not in st.session_state:
            st.session_state.detected_schema = None
    
    def _clean_llm_json(self, text: str) -> str:
        """Clean LLM output to get valid JSON"""
        # Find JSON-like content between curly braces
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return ""
        
        json_str = json_match.group()
        
        # Fix common JSON formatting issues:
        # 1. Fix unquoted property names
        json_str = re.sub(r'(\s*)([\w_]+)(\s*):([^"])', r'\1"\2"\3:\4', json_str)
        
        # 2. Fix single quotes to double quotes (handle both keys and values)
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r':"\1"', json_str)
        
        # 3. Remove trailing commas in arrays/objects
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 4. Ensure string values are properly quoted
        json_str = re.sub(r':\s*([^"{}\[\],\s][^,}\]]*)', r':"\1"', json_str)
        
        return json_str

    def detect_schema_with_llm(self, df: pd.DataFrame, ollama_llm, gemini_llm) -> Dict:
        """Use LLM to intelligently detect and categorize DataFrame schema"""
        
        # Prefer Gemini for schema reasoning, fallback to Ollama, else rule-based
        llm = gemini_llm if gemini_llm else ollama_llm
        if not llm:
            return self._fallback_schema_detection(df)
        
        # Create schema detection prompt with stronger formatting guidance
        sample_data = df.head(3).to_string(index=False, max_cols=10)
        dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        schema_prompt = f"""Return ONLY a JSON object (no other text) that classifies this DataFrame's schema.
Use ONLY double quotes for all property names and string values.

DataFrame Info:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Column Types: {dtypes_info}

Sample Data:
{sample_data}

Return this exact structure with real column names from the data:
{{
    "numeric_columns": ["list of numeric columns"],
    "categorical_columns": ["list of text/categorical columns"],
    "date_columns": ["list of date columns"],
    "id_columns": ["list of ID columns"],
    "business_metrics": ["list of KPI columns like sales, revenue"],
    "groupby_candidates": ["list of grouping columns"],
    "primary_business_entities": ["main entities"],
    "schema_insights": "brief dataset description"
}}"""

        try:
            # Get LLM response
            if gemini_llm:
                response = gemini_llm.invoke(schema_prompt)
            else:
                response = ollama_llm.invoke(schema_prompt)
            
            # Extract and clean the response
            response_text = safe_extract_output(response)
            cleaned_json = self._clean_llm_json(response_text)
            
            if cleaned_json:
                try:
                    schema = json.loads(cleaned_json)
                    
                    # Validate schema structure
                    required_keys = ["numeric_columns", "categorical_columns", "date_columns", 
                                  "business_metrics", "groupby_candidates", "schema_insights"]
                    
                    if all(key in schema for key in required_keys):
                        st.session_state.detected_schema = schema
                        return schema
                    else:
                        st.info("Schema missing required fields → Using fallback")
                except json.JSONDecodeError:
                    st.info("Failed to parse schema JSON → Using fallback")
            
        except Exception as e:
            st.info(f"LLM schema detection failed: {stringify_exception(e)} → Using fallback")
        
        # Fallback to rule-based detection
        return self._fallback_schema_detection(df)
    
    def _fallback_schema_detection(self, df: pd.DataFrame) -> Dict:
        """Rule-based schema detection as fallback"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        
        # Detect date columns
        date_cols = []
        for col in df.columns:
            lc = col.lower()
            if 'date' in lc or 'time' in lc or 'month' in lc or 'year' in lc:
                try:
                    pd.to_datetime(df[col].head(10), errors='raise')
                    date_cols.append(col)
                except Exception:
                    pass
        
        # Business heuristics
        business_keywords = ['sales', 'sale', 'revenue', 'profit', 'cost', 'price', 'amount', 'value', 'qty', 'quantity', 'margin']
        business_metrics = [col for col in numeric_cols if any(kw in col.lower() for kw in business_keywords)]
        
        groupby_candidates = [col for col in df.columns
                              if (col in categorical_cols or col in date_cols) and df[col].nunique(dropna=True) < max(50, len(df) * 0.5)]
        
        id_keywords = ['id', 'key', 'code', 'number', 'uuid']
        id_cols = [col for col in df.columns if any(kw in col.lower() for kw in id_keywords)]
        
        schema = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols,
            "id_columns": id_cols,
            "business_metrics": business_metrics or numeric_cols[:3],
            "groupby_candidates": groupby_candidates[:10],
            "primary_business_entities": groupby_candidates[:3],  # Top 3
            "schema_insights": f"Dataset with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns"
        }
        
        st.session_state.detected_schema = schema
        return schema
    
    def get_schema(self) -> Optional[Dict]:
        return st.session_state.detected_schema

schema_manager = SchemaManager()

# ---------- Cached resources ----------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    except Exception as e:
        st.warning(f"Embeddings init failed: {stringify_exception(e)}")
        return None

@st.cache_resource(show_spinner=False)
def get_vector_store():
    if not os.path.exists(CHROMA_DIR):
        return None
    try:
        embs = get_embeddings()
        if not embs:
            return None
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embs)
    except Exception as e:
        st.warning(f"Chroma init failed: {stringify_exception(e)}")
        return None

@st.cache_resource(show_spinner=False)
def initialize_llms():
    """Initialize both Ollama (local) and Gemini (cloud) once."""
    ollama_llm = None
    gemini_llm = None

    # Ollama
    try:
        ollama_llm = OllamaLLM(
            model="llama3.1",
            temperature=0,
            num_predict=500,  # Increased from 200 for fuller responses
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096,  # Increased from 1024 for more context
            top_k=20,
        )
        # Test connection
        test_response = ollama_llm.invoke("Hello, respond with 'Ollama working'")
        if "working" not in str(test_response).lower():
            st.warning("⚠️ Ollama test response unexpected - may have connection issues")
    except Exception as e:
        st.warning(f"⚠️ Ollama initialization failed: {stringify_exception(e)}")

    # Gemini
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",  # Updated to available model
                google_api_key=api_key,
                temperature=0.1,
                convert_system_message_to_human=True,
            )
            # Test connection
            test_response = gemini_llm.invoke("Hello, respond with 'Gemini working'")
            if "working" not in str(test_response).lower():
                st.warning("⚠️ Gemini test response unexpected - may have connection issues")
        else:
            st.warning("⚠️ GEMINI_API_KEY missing in .env")
    except Exception as e:
        st.warning(f"⚠️ Gemini initialization failed: {stringify_exception(e)}")

    return ollama_llm, gemini_llm

@st.cache_resource(show_spinner=False)
def load_sales_data(path: str):
    """Load CSV to DataFrame with minimal sanity checks."""
    if not os.path.exists(path):
        return None, f"Missing data file at {path}"
    try:
        df = pd.read_csv(path)
        # Optional: ensure datetime if any column named like 'date'
        for col in df.columns:
            if "date" in col.lower():
                with pd.option_context("mode.chained_assignment", None):
                    try:
                        df[col] = pd.to_datetime(df[col], errors="ignore")
                    except Exception:
                        pass
        return df, None
    except Exception as e:
        return None, f"Failed to load data: {stringify_exception(e)}"

# ---------- Agent builders ----------

def create_ollama_agent(df: pd.DataFrame, ollama_llm: OllamaLLM):
    if not ollama_llm:
        return None
    try:
        return create_pandas_dataframe_agent(
            llm=ollama_llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=30,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            return_intermediate_steps=False,
            # Add consistent system message
            agent_kwargs={
                "system_message": "You are a data analyst. Always provide well-formatted, explained responses. Don't return raw numbers without context."
            }
        )
    except Exception as e:
        st.warning(f"Ollama agent creation failed: {stringify_exception(e)}")
        return None

def create_gemini_agent(df: pd.DataFrame, gemini_llm: ChatGoogleGenerativeAI):
    if not gemini_llm:
        return None
    try:
        return create_pandas_dataframe_agent(
            llm=gemini_llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=30,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            return_intermediate_steps=False,
            # Add consistent system message
            agent_kwargs={
                "system_message": "You are a senior data analyst. Always provide professional, well-formatted responses with context and interpretation. Present grouped data in tables."
            }
        )
    except Exception as e:
        st.warning(f"Gemini agent creation failed: {stringify_exception(e)}")
        return None

# Enhanced output validation and post-processing
def enhance_agent_output(raw_output: str, query: str, df: pd.DataFrame) -> str:
    """Post-process agent output to ensure consistent formatting"""
    if not raw_output or len(raw_output.strip()) < 10:
        return raw_output
    
    # Check if output is just a raw number
    try:
        # If the output is just a number, enhance it
        float(raw_output.strip())
        # It's a raw number - add context
        return f"""**Query Result:**

The answer to "{query}" is: **{raw_output.strip()}**

**Context:**
- Dataset: {df.shape[0]} rows × {df.shape[1]} columns
- Calculation: Direct numerical result from pandas operation
- Result type: Single aggregate value"""
        
    except ValueError:
        # Not a raw number, check if it needs table formatting
        lines = raw_output.strip().split('\n')
        
        # If output contains multiple data rows but no proper formatting
        if len(lines) > 3 and not any('|' in line or '**' in line for line in lines):
            # Try to detect if this is grouped data that needs table formatting
            if any(word in query.lower() for word in ['by', 'each', 'per', 'group']):
                return f"""**Analysis Results:**

{raw_output}

**Interpretation:**
This shows the breakdown of your requested analysis across different categories/groups in the dataset."""
    
    return raw_output

# ---------- Query Classification System ----------

def classify_query_complexity(query: str, schema: Dict) -> str:
    """Simplified classification for Simple and Complex only"""
    q = query.lower()
    
    # Check for grouping operations (default to SIMPLE unless complex conditions met)
    grouping_indicators = ['by', 'per', 'for each', 'grouped', 'group', 'across']
    if any(indicator in q for indicator in grouping_indicators):
        # If grouping with multiple dimensions or time analysis, treat as COMPLEX
        time_terms = ['trend', 'over time', 'growth', 'change', 'month', 'year', 'daily', 'weekly']
        analysis_terms = ['analyze', 'analyse', 'analysis', 'insight', 'pattern', 'correlation']
        
        if any(term in q for term in time_terms + analysis_terms):
            return "COMPLEX"
        return "SIMPLE"
    
    # Always route to COMPLEX for advanced analysis
    complex_patterns = [
        'analyze', 'analyse', 'analysis', 'insight', 'pattern', 'trend',
        'correlation', 'relationship', 'predict', 'forecast', 'estimate',
        'describe', 'elaborate', 'explain', 'detail', 'compare', 'contrast',
        'versus', 'vs', 'impact of', 'effect of', 'chart', 'plot', 'graph',
        'visual', 'pie', 'bar', 'histogram', 'scatter'
    ]
    
    if any(pattern in q for pattern in complex_patterns):
        return "COMPLEX"
    
    # Check for time-based analysis
    if any(term in q for term in ['trend', 'over time', 'growth', 'change', 'month', 'year']):
        return "COMPLEX"
    
    # Check for multiple metrics mentioned
    if schema.get('business_metrics'):
        metric_mentions = sum(1 for metric in schema['business_metrics'] 
                            if metric.lower() in q)
        if metric_mentions > 1:
            return "COMPLEX"
    
    # Default to SIMPLE for most queries
    return "SIMPLE"

def preprocess_query_for_llm(query: str, df: pd.DataFrame, llm_type: str, schema: Dict) -> str:
    """Enhanced preprocessing with schema context and consistent formatting"""
    q = query.lower()
    
    # Check if this is a chart query
    chart_keywords = ['chart', 'plot', 'graph', 'visuali', 'pie', 'bar', 'histogram', 'scatter', 'line chart']
    is_chart_query = any(keyword in q for keyword in chart_keywords)
    
    # Consistent formatting instructions for both LLM types
    format_instructions = """
RESPONSE FORMATTING RULES:
- Always provide context and explanation with your numerical results
- For single values: explain what the number represents
- For grouped/aggregated data: present in clear table format with headers
- Include brief interpretation of the results
- Use proper formatting (tables, bullet points) for readability
"""
    
    if llm_type == "OLLAMA":
        schema_context = f"""
Schema Info:
- Business Metrics: {schema.get('business_metrics', [])}
- Group-by Columns: {schema.get('groupby_candidates', [])}
- Date Columns: {schema.get('date_columns', [])}
"""
        return f"""You are a helpful data analyst. Answer the user's question about the dataset using python/pandas.

{format_instructions}

{schema_context}
Question: {query}

DataFrame info: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {list(df.columns)}

IMPORTANT: Always explain your results and provide context. Don't just return raw numbers.
Provide a clear, well-formatted answer with interpretation."""

    elif llm_type == "GEMINI":
        head_sample = df.head(2).to_string(index=False, max_cols=8)
        schema_insights = schema.get('schema_insights', 'Business dataset')
        
        # Add chart-specific instructions for Gemini
        chart_instructions = ""
        if is_chart_query:
            chart_instructions = """
CHART CREATION INSTRUCTIONS:
- You are working in Streamlit environment
- NEVER use plt.show() - it will not work
- Instead use st.pyplot(plt.gcf()) to display charts
- Always import streamlit as st
- Use plt.figure(figsize=(width, height)) for proper sizing
- Add proper titles, labels, and formatting
- Call plt.close() after st.pyplot() to prevent memory issues

Example pattern:
```python
import matplotlib.pyplot as plt
import streamlit as st
plt.figure(figsize=(10, 6))
# your plotting code here
plt.title('Chart Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
st.pyplot(plt.gcf())
plt.close()
```
"""
        
        return f"""You are a senior data analyst. Provide comprehensive analysis for this question.

{format_instructions}
{chart_instructions}

Dataset Context: {schema_insights}
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Key Business Metrics: {schema.get('business_metrics', [])}
- Grouping Dimensions: {schema.get('groupby_candidates', [])}
- Date Columns: {schema.get('date_columns', [])}

Sample Data:
{head_sample}

Question: {query}

Please provide:
1. Direct answer with specific numbers/insights
2. Business context and interpretation  
3. Any actionable recommendations (if applicable)
4. Proper formatting (tables for grouped data, explanations for single values)

IMPORTANT: Always format your responses professionally with context and explanation.
Use the python_repl_ast tool for complex pandas operations."""
    return query

# ---------- Pandas helpers ----------

def run_pandas_agent(query: str, df: pd.DataFrame, gemini_llm=None, ollama_llm=None) -> Optional[str]:
    """
    Execute query using a Pandas agent with enhanced prompting for consistency.
    """
    agent = None
    agent_type = None
    
    try:
        if gemini_llm:
            agent = create_gemini_agent(df, gemini_llm)
            agent_type = "Gemini"
        if not agent and ollama_llm:
            agent = create_ollama_agent(df, ollama_llm)
            agent_type = "Ollama"
        if not agent:
            st.warning("No LLM available for Pandas agent.")
            return None
        
        st.info(f"Using {agent_type} Pandas Agent...")
        
        # Enhanced query with formatting instructions
        enhanced_query = f"""
{query}

RESPONSE FORMAT REQUIREMENTS:
- Always provide context and explanation with numerical results
- Format grouped data in clear table format
- Include brief interpretation of results
- Don't return raw numbers without explanation
- Use professional business analysis tone
"""
        
        # Execute with timeout protection
        with st.spinner(f"{agent_type} processing..."):
            res = agent.invoke({"input": enhanced_query})
            output = safe_extract_output(res)
            
            # Clean response if it came from Gemini
            if agent_type == "Gemini":
                output = clean_gemini_response(output)
            
            # Apply output enhancement
            output = enhance_agent_output(output, query, df)
            
            # Validate output quality
            if output and len(output.strip()) > 10:
                return output
            else:
                st.warning(f"{agent_type} agent returned minimal output: {output}")
                return None
                
    except Exception as e:
        st.warning(f"Pandas agent ({agent_type}) failed: {stringify_exception(e)}")
        return None

# ---------- Chart Validation Helper ----------

def validate_chart_display() -> bool:
    """Validate that chart was successfully created and displayed"""
    try:
        import matplotlib.pyplot as plt
        # Check if there's an active figure
        if plt.gcf().get_axes():
            return True
        return False
    except Exception:
        return False

def safe_chart_display() -> str:
    """Safely display chart with validation"""
    try:
        import matplotlib.pyplot as plt
        if validate_chart_display():
            st.pyplot(plt.gcf())
            plt.close()
            return "Chart displayed successfully."
        else:
            plt.close()  # Clean up even if no chart
            return "No chart was generated."
    except Exception as e:
        try:
            plt.close()  # Always try to clean up
        except:
            pass
        return f"Chart display error: {stringify_exception(e)}"

# ---------- Knowledge Base search ----------

def kb_search(query: str) -> str:
    vs = get_vector_store()
    if not vs:
        return "📝 No knowledge base available."
    try:
        docs = vs.similarity_search(query, k=3)
        if not docs:
            return "❓ No relevant information found in knowledge base."
        items = []
        for i, d in enumerate(docs, 1):
            content = d.page_content[:1200]
            items.append(f"**{i}.** {content}")
        return "📖 **Knowledge Base Results:**\n\n" + "\n\n".join(items)
    except Exception as e:
        return f"❌ Knowledge base search failed: {stringify_exception(e)}"

# ---------- Enhanced fallback analysis ----------

def enhanced_fallback_analysis(df: pd.DataFrame, query: str, schema: Dict) -> str:
    """Enhanced fallback using schema information"""
    sample = df.head(2).to_string(index=False, max_cols=8)
    business_metrics = schema.get('business_metrics', [])
    groupby_cols = schema.get('groupby_candidates', [])
    
    return f"""🤔 **Query:** "{query}"

**💡 What I can analyze:**

**🔄 Simple (Ollama + Fallbacks):**
- Group by analysis: {', '.join(groupby_cols[:3]) if groupby_cols else 'categories'}
- Business metrics: {', '.join(business_metrics[:3]) if business_metrics else 'calculations'}
- Rankings and comparisons

**💎 Complex (Gemini + Advanced):**
- Trend analysis and forecasting
- Multi-dimensional correlations
- Business insights and recommendations
- Scenario modeling

**📊 Your Dataset Context:**
- **Shape:** {df.shape[0]} rows × {df.shape[1]} columns  
- **Key Metrics:** {business_metrics[:4] if business_metrics else 'Not detected'}
- **Dimensions:** {groupby_cols[:4] if groupby_cols else 'Not detected'}
- **Schema Insight:** {schema.get('schema_insights', 'Business dataset')}

**Sample:**
{sample}
"""

# ---------- Execution strategies with fallback ----------

def execute_simple_strategy(query: str, df: pd.DataFrame, schema: Dict, ollama_llm, gemini_llm) -> Tuple[str, str]:
    """Simple queries with consistent output formatting"""
    start_time = time.time()
    
    # 1. Try Ollama first with enhanced preprocessing
    if ollama_llm:
        try:
            agent = create_ollama_agent(df, ollama_llm)
            if agent:
                enhanced_prompt = preprocess_query_for_llm(query, df, "OLLAMA", schema)
                try:
                    res = agent.invoke({"input": enhanced_prompt})
                    output = safe_extract_output(res)
                    
                    # Enhanced output processing
                    output = enhance_agent_output(output, query, df)
                    
                    if (output.strip() and 
                        len(output) > 30 and 
                        not any(err in output.lower() for err in ["error", "failed", "cannot"])):
                        
                        execution_time = time.time() - start_time
                        tracker.log_query(query, "SIMPLE", "ollama", True, execution_time)
                        return f"**Ollama Analysis:**\n\n{output}", "ollama"
                except (OutputParserException, ValueError) as e:
                    st.info(f"Ollama parsing failed: {stringify_exception(e)}")
            else:
                st.info("Ollama agent creation failed → Trying Pandas Agent...")
        except Exception as e:
            st.info(f"Ollama setup failed: {stringify_exception(e)} → Trying Pandas Agent...")

    # 2. Try Pandas Agent with enhanced preprocessing
    try:
        enhanced_query = f"""
Please analyze: {query}

Instructions:
- Provide clear explanation with your numerical results
- Format grouped data as tables
- Include brief interpretation
- Don't return raw numbers without context
"""
        pandas_res = run_pandas_agent(enhanced_query, df, gemini_llm=gemini_llm, ollama_llm=ollama_llm)
        if pandas_res and len(pandas_res.strip()) > 20:
            pandas_res = clean_gemini_response(pandas_res)
            pandas_res = enhance_agent_output(pandas_res, query, df)
            
            execution_time = time.time() - start_time
            tracker.log_query(query, "SIMPLE", "pandas-agent", True, execution_time)
            return f"**Pandas Agent Analysis:**\n\n{pandas_res}", "pandas-agent"
    except Exception:
        st.info("Pandas Agent failed → Using enhanced fallback...")

    # 3. Final fallback
    execution_time = time.time() - start_time
    tracker.log_query(query, "SIMPLE", "fallback", False, execution_time)
    return enhanced_fallback_analysis(df, query, schema), "fallback"

def execute_complex_strategy(query: str, df: pd.DataFrame, schema: Dict, gemini_llm) -> Tuple[str, str]:
    """Complex queries: Gemini → Pandas Agent (fallback) → Enhanced fallback - WITH CHART VALIDATION"""
    start_time = time.time()
    
    # Check if this is a chart query for additional validation
    q = query.lower()
    chart_keywords = ['chart', 'plot', 'graph', 'visuali', 'pie', 'bar', 'histogram', 'scatter', 'line chart']
    is_chart_query = any(keyword in q for keyword in chart_keywords)
    
    # 1. Try Gemini for advanced reasoning
    if gemini_llm:
        try:
            prompt = preprocess_query_for_llm(query, df, "GEMINI", schema)
            agent = create_gemini_agent(df, gemini_llm)
            if agent:
                with st.spinner("Gemini analyzing..."):
                    res = agent.invoke({"input": prompt})
                output = safe_extract_output(res)
                
                # Clean the Gemini response
                output = clean_gemini_response(output)
                
                # For chart queries, add validation message
                if is_chart_query:
                    chart_status = safe_chart_display()
                    if "successfully" in chart_status:
                        output += f"\n\n---\n*{chart_status}*"
                    else:
                        output += f"\n\n---\n*⚠️ {chart_status}*"
                
                # Better validation for Gemini responses
                if (output.strip() and 
                    len(output) > 30 and
                    not output.lower().startswith("i cannot") and
                    not output.lower().startswith("i'm sorry")):
                    
                    execution_time = time.time() - start_time
                    tracker.log_query(query, "COMPLEX", "gemini", True, execution_time)
                    return f"Gemini Advanced Analysis:\n\n{output}", "gemini"
                else:
                    st.info("Gemini returned minimal response → Trying Pandas Agent…")
        except Exception as e:
            st.warning(f"Gemini API failed: {stringify_exception(e)}")

    # 2. Fallback → Pandas Agent (Gemini preference, then Ollama)
    try:
        pandas_res = run_pandas_agent(query, df, gemini_llm=gemini_llm, ollama_llm=None)
        if pandas_res and len(pandas_res.strip()) > 20:
            # Clean pandas agent response too (in case it's also from Gemini)
            pandas_res = clean_gemini_response(pandas_res)
            
            # Chart validation for pandas agent too
            if is_chart_query:
                chart_status = safe_chart_display()
                pandas_res += f"\n\n---\n*{chart_status}*"
            
            execution_time = time.time() - start_time
            tracker.log_query(query, "COMPLEX", "pandas-agent", True, execution_time)
            return f"Pandas Agent (Complex Fallback):\n\n{pandas_res}", "pandas-agent"
    except Exception:
        st.info("Pandas Agent also failed. Using enhanced fallback…")
    
    # 3. Final fallback → Enhanced analysis
    execution_time = time.time() - start_time
    tracker.log_query(query, "COMPLEX", "fallback", False, execution_time)
    return enhanced_fallback_analysis(df, query, schema), "fallback"

def execute_hybrid_strategy(query: str, df: pd.DataFrame, schema: Dict, ollama_llm, gemini_llm) -> str:
    """Main router with schema-aware execution - Simple and Complex only"""
    complexity = classify_query_complexity(query, schema)
    st.info(f"🎯 Query classified as **{complexity}** | Schema-aware routing")
    
    if complexity == "SIMPLE":
        st.info("🔄 Route: Ollama → Pandas Agent → Enhanced fallback")
        result, path = execute_simple_strategy(query, df, schema, ollama_llm, gemini_llm)  # Using simple strategy
    else:  # COMPLEX
        st.info("💎 Route: Gemini → Pandas Agent → Enhanced fallbacks")
        result, path = execute_complex_strategy(query, df, schema, gemini_llm)
    
    # Add execution path info to result
    path_emoji = {
        "ollama": "🦙", "gemini": "💎", "fallback": "🔧", "pandas-agent": "🐼"
    }
    
    return f"{result}\n\n---\n*Executed via: {path_emoji.get(path, '❓')} {path.upper()}*"

# ---------- Gemini Response Cleaning ----------
def clean_gemini_response(response_text: str) -> str:
    """Clean up Gemini response by removing excessive markdown formatting with error handling"""
    if not response_text or not isinstance(response_text, str):
        return str(response_text) if response_text else ""
    
    try:
        # Remove double asterisks (bold markdown)
        cleaned = response_text.replace('**', '')
        
        # Remove single asterisks at word boundaries (italic markdown) with error handling
        import re
        try:
            cleaned = re.sub(r'\*(\w[^*]*\w)\*', r'\1', cleaned)
        except re.error:
            # Fallback: simple replacement if regex fails
            cleaned = cleaned.replace('*', '')
        
        # Remove triple backticks and language specifiers with error handling
        try:
            cleaned = re.sub(r'```\w*\n', '', cleaned)
            cleaned = cleaned.replace('```', '')
        except re.error:
            # Fallback: simple replacement
            cleaned = cleaned.replace('```', '')
        
        # Clean up excessive newlines with error handling
        try:
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        except re.error:
            # Fallback: leave as is if regex fails
            pass
        
        return cleaned.strip()
        
    except Exception as e:
        # If all cleaning fails, return original text
        st.warning(f"Response cleaning failed: {stringify_exception(e)}")
        return response_text

# ---------- UI ----------

st.title("🤖 OmniCore — Schema-Aware Hybrid AI Agents")
st.caption("Dynamic schema detection • Smart routing • Multi-layer fallbacks • Audit tracking")

# Initialize LLMs once
ollama_llm, gemini_llm = initialize_llms()

# Load and analyze data schema
df, data_err = load_sales_data(DATA_PATH)
schema: Dict = {}

if not data_err and df is not None:
    # Detect schema using LLM
    with st.spinner("🔍 Detecting dataset schema..."):
        schema = schema_manager.detect_schema_with_llm(df, ollama_llm, gemini_llm)

# Sidebar with enhanced status and analytics
with st.sidebar:
    st.subheader("🔧 System Status")

    # Data & Schema status
    if data_err:
        st.error(f"❌ {data_err}")
    else:
        st.success("✅ sales_data.csv ready")
        if df is not None:
            st.write(f"📊 {df.shape[0]} rows × {df.shape[1]} cols")
            
            # Schema insights
            if schema:
                st.success("🧠 Schema detected via LLM")
                with st.expander("📋 Schema Details"):
                    st.write(f"**Business Metrics:** {', '.join(schema.get('business_metrics', [])[:5]) or 'N/A'}")
                    st.write(f"**Grouping Columns:** {', '.join(schema.get('groupby_candidates', [])[:8]) or 'N/A'}")
                    st.write(f"**Date Columns:** {', '.join(schema.get('date_columns', []) or []) or 'N/A'}")
                    st.write(f"**Insights:** {schema.get('schema_insights', 'N/A')}")
            else:
                st.warning("⚠️ Schema detection failed (using rule-based defaults)")

    # LLMs status
    st.markdown("### 🤖 AI Models")
    if ollama_llm:
        st.success("🦙 Ollama Ready")
        st.caption("Local • Fast • Private")
    else:
        st.error("❌ Ollama Unavailable")
        st.caption("Check if Ollama service is running")
    
    if gemini_llm:
        st.success("💎 Gemini Ready")
        st.caption("Cloud • Advanced • Powerful")
    else:
        st.warning("⚠️ Gemini not configured")
        st.caption("Add GEMINI_API_KEY to .env")

    # Knowledge Base
    st.markdown("### 📚 Knowledge Base")
    if get_vector_store():
        st.success("✅ Chroma Ready")
    else:
        st.info("ℹ️ No KB found")

    # Query Analytics
    st.markdown("### 📈 Query Analytics")
    stats = tracker.get_stats()
    if stats['total_queries'] > 0:
        st.metric("Total Queries", stats['total_queries'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🦙 Ollama", stats['ollama_calls'])
            st.metric("💎 Gemini", stats['gemini_calls'])
        with col2:
            st.metric("🔧 Fallbacks", stats['fallback_hits'])
            st.metric("🐼 Pandas Agent", stats['pandas_agent'])
        
        # Success rate
        successful_queries = sum([
            stats['ollama_calls'], stats['gemini_calls'], stats['pandas_agent']
        ])
        success_rate = (successful_queries / stats['total_queries']) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%", help="% queries with meaningful responses")
    else:
        st.info("No queries yet")
    
    # Performance tips
    st.markdown("---")
    st.markdown("### ⚡ Optimization Tips")
    st.write("- Schema auto-detected for smart routing")
    st.write("- Simple: Ollama → Pandas Agent fallback")
    st.write("- Complex: Gemini → Enhanced fallbacks")

# Main interface
if data_err:
    st.error(f"❌ {data_err}")
    st.info("""
    **Quick Fix:**
    1. Create a `data/` folder in your project
    2. Place your CSV file as `sales_data.csv`
    3. Or update the `DATA_PATH` variable in the code
    """)
else:
    # Data preview with schema
    with st.expander("📊 Dataset Overview", expanded=False):
        if df is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📋 Schema Analysis")
                if schema:
                    st.write("**🎯 Business Focus:**")
                    st.write(schema.get('schema_insights', 'Standard dataset'))
                    
                    if schema.get('business_metrics'):
                        st.write("**💰 Key Metrics:**")
                        for metric in schema['business_metrics'][:5]:
                            st.write(f"• {metric}")
                    
                    if schema.get('groupby_candidates'):
                        st.write("**📊 Dimensions:**")
                        for dim in schema['groupby_candidates'][:8]:
                            st.write(f"• {dim}")
                else:
                    st.warning("Schema detection failed")
            
            with col2:
                st.subheader("📄 Sample Data")
                st.dataframe(df.head(8), use_container_width=True)

    # Enhanced routing explanation
    with st.expander("🎯 Smart Routing System"):
        st.markdown("""
**🧠 Schema-Aware Classification:**
- **SIMPLE** → 🦙 Ollama (local) → 🐼 Pandas Agent → Enhanced fallback  
- **COMPLEX** → 💎 Gemini (cloud) → 🐼 Pandas Agent → Enhanced fallback

**🔄 Multi-Layer Fallbacks:**
1. Local LLM processing (Ollama) for simple complexity  
2. Cloud LLM reasoning (Gemini) for complex analysis  
3. Pandas agents as secondary option  
4. Enhanced rule-based analysis as final fallback

**📊 Schema-Driven Optimization:**
- Auto-detects business metrics, dimensions, date columns  
- Routes queries to optimal execution layer  
- Tracks performance for continuous improvement
        """)

    # Dynamic examples based on detected schema
    with st.expander("💡 Schema-Optimized Query Examples"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔄 SIMPLE (Ollama+)**") 
            examples = []
            if schema.get('business_metrics') and schema.get('groupby_candidates'):
                metric = schema['business_metrics'][0]
                dimension = schema['groupby_candidates'][0]
                examples = [
                    f"Average {metric} by {dimension}",
                    f"Top 10 {dimension} by {metric}",
                    f"{metric} analysis by category"
                ]
            examples.extend(["Filter data by condition", "Monthly trends"])
            for ex in examples[:6]:
                st.write(f"• {ex}")
        
        with col2:
            st.markdown("**💎 COMPLEX (Gemini)**")
            examples = ["Analyze trends and patterns", "Correlations and insights"]
            if schema.get('primary_business_entities') and len(schema['primary_business_entities']) > 1:
                entities = schema['primary_business_entities'][:2]
                examples.extend([f"Compare {entities[0]} vs {entities[1]}", f"Predict {entities[0]} performance"])
            examples.extend(["Business recommendations", "What-if scenarios", "Create visualizations"])
            for ex in examples[:6]:
                st.write(f"• {ex}")

# Query input with enhanced help
query = st.text_input(
    "🗣️ Ask your question:",
    placeholder="e.g., 'Analyze sales trends by region' or 'Show top 5 products'",
    help="System will auto-detect complexity and route to optimal execution layer"
)

# Mode selection
mode = st.radio(
    "Choose source:",
    options=["Sales Data", "Knowledge Base"],
    horizontal=True,
    index=0,
    help="Select data source for query analysis"
)

# Query execution
if query:
    st.markdown("---")
    with st.spinner("🧠 Processing with schema-aware routing..."):
        try:
            if mode == "Knowledge Base":
                response = kb_search(query)
            else:  # Sales Data
                if df is not None and schema:
                    response = execute_hybrid_strategy(query, df, schema, ollama_llm, gemini_llm)
                else:
                    response = "❌ Sales data not available. Please check if the data file is loaded correctly."

            # Display results
            st.subheader("📝 Analysis Results")
            if isinstance(response, str) and len(response) > 1500:
                st.text_area("Detailed Response:", response, height=500, help="Long response - scroll to see all content")
            else:
                st.markdown(response)

        except Exception as e:
            st.error(f"❌ Error during processing: {stringify_exception(e)}")
            tracker.log_query(query, "ERROR", "error", False, 0)

# Query history and audit (optional)
if st.sidebar.button("📊 Show Query Audit"):
    with st.expander("📈 Query Performance Audit", expanded=True):
        stats = tracker.get_stats()
        if stats['audit_logs']:
            # Recent queries
            st.subheader("🕐 Recent Queries")
            recent_logs = stats['audit_logs'][-10:]  # Last 10 queries
            for log in reversed(recent_logs):
                success_icon = "✅" if log['success'] else "❌"
                path_icon = {
                    "ollama": "🦙", "gemini": "💎",
                    "fallback": "🔧", "pandas-agent": "🐼"
                }.get(log['execution_path'], "❓")
                
                st.write(f"{success_icon} {path_icon} **{log['complexity']}** | {log['response_time_ms']:.0f}ms")
                st.caption(f"Query: {log['query'][:160]}{'...' if len(log['query'])>160 else ''}")
                st.caption(f"Path: {log['execution_path']} | Time: {log['timestamp']}")
                st.divider()
        else:
            st.info("No queries logged yet")

# Footer with system info
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🤖 **OmniCore v2.0**")
    st.caption("Schema-aware hybrid routing")

with col2:
    if 'df' in locals() and df is not None:
        st.caption(f"📊 **Dataset:** {df.shape[0]}×{df.shape[1]}")
    st.caption("🧠 Dynamic LLM selection")

with col3:
    stats = tracker.get_stats()
    st.caption(f"📈 **Queries:** {stats['total_queries']}")
    if stats['total_queries'] > 0:
        successful_queries = sum([stats['ollama_calls'], stats['gemini_calls'], stats['pandas_agent']])
        success_rate = (successful_queries / stats['total_queries']) * 100
        st.caption(f"✅ **Success:** {success_rate:.1f}%")