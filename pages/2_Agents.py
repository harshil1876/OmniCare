import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.schema import OutputParserException
from dotenv import load_dotenv
import pandas as pd
import os
import re

# Load environment variables from .env
load_dotenv()

# Initialize both LLMs
@st.cache_resource
def initialize_llms():
    """Initialize both Ollama and Gemini LLMs with error handling"""
    ollama_llm = None
    gemini_llm = None
    
    try:
        # Lightweight Ollama for basic tasks
        ollama_llm = OllamaLLM(
            model="llama2:latest",
            temperature=0,
            num_predict=200,  # Short responses for basic queries
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=1024,  # Smaller context for efficiency
            top_k=20,
        )
        st.success("🦙 Ollama LLM initialized")
    except Exception as e:
        st.warning(f"⚠️ Ollama initialization failed: {str(e)}")
    
    try:
        # ✅ Load Gemini API key from .env
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if gemini_api_key:
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=gemini_api_key,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            st.success("💎 Gemini LLM initialized")
        else:
            st.warning("⚠️ Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
            
    except Exception as e:
        st.warning(f"⚠️ Gemini initialization failed: {str(e)}")
    
    return ollama_llm, gemini_llm

def classify_query_complexity(query):
    """Classify query as BASIC, MODERATE, or COMPLEX"""
    query_lower = query.lower()
    
    # BASIC queries - use Ollama (fast, local)
    basic_indicators = [
        "head", "first", "show", "display", "columns", "shape", "size", 
        "describe", "info", "summary", "list", "what are", "how many"
    ]
    
    # COMPLEX queries - use Gemini (advanced reasoning)
    complex_indicators = [
        "analyze", "insights", "trends", "patterns", "correlations", 
        "predictions", "recommendations", "optimize", "compare", "contrast",
        "what if", "scenario", "forecast", "model", "relationship",
        "anomal", "outlier", "cluster", "segment", "classify"
    ]
    
    # MODERATE queries - try Ollama first, fallback to Gemini
    moderate_indicators = [
        "total", "sum", "average", "mean", "max", "min", "count",
        "group by", "filter", "where", "sort", "rank", "top",
        "calculate", "compute", "find", "search", "get"
    ]
    
    # Time-based or conditional complexity
    time_complexity = any(word in query_lower for word in ["on", "between", "during", "when", "before", "after"])
    conditional_complexity = any(word in query_lower for word in ["if", "where", "condition", "criteria"])
    multiple_operations = query_lower.count("and") > 1 or query_lower.count("or") > 1
    
    # Classification logic
    if any(indicator in query_lower for indicator in complex_indicators):
        return "COMPLEX"
    elif any(indicator in query_lower for indicator in basic_indicators):
        return "BASIC"
    elif (time_complexity and conditional_complexity) or multiple_operations:
        return "COMPLEX"
    elif any(indicator in query_lower for indicator in moderate_indicators):
        return "MODERATE"
    else:
        # Default classification based on query length and complexity
        if len(query.split()) > 10 or len(re.findall(r'[0-9]{4}', query)) > 1:
            return "COMPLEX"
        else:
            return "BASIC"

def create_ollama_agent(df):
    """Create lightweight Ollama agent for basic queries"""
    try:
        ollama_llm, _ = initialize_llms()
        if not ollama_llm:
            return None
            
        agent = create_pandas_dataframe_agent(
            llm=ollama_llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
            max_iterations=100,  # Limited iterations for speed
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        return agent
    except Exception as e:
        st.warning(f"Ollama agent creation failed: {str(e)}")
        return None

def create_gemini_agent(df):
    """Create powerful Gemini agent for complex queries"""
    try:
        _, gemini_llm = initialize_llms()
        if not gemini_llm:
            return None
            
        agent = create_pandas_dataframe_agent(
            llm=gemini_llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=100,  # More iterations for complex analysis
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        return agent
    except Exception as e:
        st.warning(f"Gemini agent creation failed: {str(e)}")
        return None

def preprocess_query_for_llm(query, df, llm_type):
    """Preprocess query based on LLM type"""
    if llm_type == "OLLAMA":
        # Simple, direct instructions for Ollama
        return f"""Answer this question about the dataframe: {query}

Dataframe info:
- Columns: {list(df.columns)}
- Shape: {df.shape}

Use python_repl_ast tool to execute pandas code."""
    
    elif llm_type == "GEMINI":
        # Detailed context and instructions for Gemini
        return f"""You are a data analysis expert. Analyze this dataframe and answer the question comprehensively.

Dataset Information:
- Columns: {list(df.columns)}
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Data types: {dict(df.dtypes)}

Sample data:
{df.head(2).to_string()}

Question: {query}

Please provide:
1. Direct answer to the question
2. Key insights from the analysis
3. Any notable patterns or recommendations

Use the python_repl_ast tool to execute pandas operations."""
    
    return query

def execute_hybrid_agent_strategy(query, df):
    """Execute hybrid strategy based on query complexity"""
    
    # Classify query complexity
    complexity = classify_query_complexity(query)
    st.info(f"🎯 Query classified as: **{complexity}**")
    
    if complexity == "BASIC":
        return execute_basic_strategy(query, df)
    elif complexity == "MODERATE": 
        return execute_moderate_strategy(query, df)
    else:  # COMPLEX
        return execute_complex_strategy(query, df)

def execute_basic_strategy(query, df):
    """Handle basic queries with Ollama or direct analysis"""
    st.info("⚡ Using fast local analysis for basic query...")
    
    # Try direct analysis first for very basic queries
    direct_result = try_direct_analysis(query, df)
    if direct_result:
        return f"🦙 **Ollama Local Analysis:**\n\n{direct_result}"
    
    # Try Ollama agent
    ollama_agent = create_ollama_agent(df)
    if ollama_agent:
        try:
            processed_query = preprocess_query_for_llm(query, df, "OLLAMA")
            result = ollama_agent.invoke(processed_query)
            
            output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
            if output and len(output.strip()) > 10:
                return f"🦙 **Ollama Analysis:**\n\n{output}"
        except Exception:
            st.warning("⚠️ Ollama couldn’t handle this query, switching to fallback...")
    
    # Fallback to enhanced direct analysis
    return enhanced_fallback_analysis(df, query)


def execute_moderate_strategy(query, df):
    """Handle moderate queries - try Ollama first, then Gemini"""
    st.info("🔄 Using hybrid approach for moderate complexity query...")
    
    # Try Ollama first (faster)
    ollama_agent = create_ollama_agent(df)
    if ollama_agent:
        try:
            processed_query = preprocess_query_for_llm(query, df, "OLLAMA")
            result = ollama_agent.invoke(processed_query)
            
            output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
            if output and len(output.strip()) > 20 and "error" not in output.lower():
                return f"🦙 **Ollama Analysis:**\n\n{output}"
        except Exception:
            st.info("Switching directly to Gemini....")
    
    # Fallback to Gemini for better analysis
    return execute_complex_strategy(query, df)

def execute_complex_strategy(query, df):
    """Handle complex queries with Gemini"""
    st.info("💎 Using advanced Gemini analysis for complex query...")
    
    gemini_agent = create_gemini_agent(df)
    if gemini_agent:
        try:
            processed_query = preprocess_query_for_llm(query, df, "GEMINI")
            
            with st.spinner("🧠 Gemini is performing advanced analysis..."):
                result = gemini_agent.invoke(processed_query)
            
            output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
            if output and len(output.strip()) > 10:
                return f"💎 **Gemini Advanced Analysis:**\n\n{output}"
                
        except Exception as e:
            st.warning(f"Gemini analysis failed: {str(e)}")
    
    # Final fallback to enhanced analysis
    st.info("🔧 Using enhanced fallback analysis...")
    return enhanced_fallback_analysis(df, query)

def try_direct_analysis(query, df):
    """Quick direct analysis for very basic queries"""
    query_lower = query.lower()
    
    # Very basic operations that don't need LLM
    if "head" in query_lower or "first" in query_lower:
        n = extract_number_from_query(query) or 5
        return f"First {n} rows:\n\n{df.head(n).to_string(index=False)}"
    
    elif "shape" in query_lower or "size" in query_lower:
        return f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns"
    
    elif "columns" in query_lower:
        return f"Columns: {', '.join(df.columns)}\n\nColumn details:\n" + "\n".join([f"• {col} ({df[col].dtype})" for col in df.columns])
    
    elif "describe" in query_lower and len(query.split()) <= 3:
        return f"Dataset description:\n\n{df.describe().to_string()}"
    
    elif "unique" in query_lower:
        return f"Unique values per column:\n\n{df.nunique().to_string()}"
    
    elif "value counts" in query_lower:
        # Try to detect column name from query, fallback to first column
        detected_col = None
        for col in df.columns:
            if col.lower() in query_lower:
                detected_col = col
                break
        col = detected_col or df.columns[0]
        return f"Value counts for '{col}':\n\n{df[col].value_counts().to_string()}"
    
    return None

def enhanced_fallback_analysis(df, query):
    """Enhanced fallback analysis for when agents fail"""
    try:
        query_lower = query.lower()
        
        # Dynamic column detection
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        date_columns = detect_date_columns(df)
        
        # Handle different query types
        if any(viz_word in query_lower for viz_word in ["plot", "graph", "chart", "visualize"]):
            return handle_visualization_analysis(df, query, numeric_columns, text_columns)
        
        elif any(analysis_word in query_lower for analysis_word in ["analyze", "analysis", "insights", "trends"]):
            return handle_comprehensive_analysis(df, query, numeric_columns, text_columns, date_columns)
        
        elif any(date_word in query_lower for date_word in ["on", "date", "day"]) and date_columns:
            return handle_date_specific_analysis(df, query, query_lower, date_columns, numeric_columns)
        
        elif any(agg_word in query_lower for agg_word in ["total", "sum", "average", "mean"]) and numeric_columns:
            return handle_aggregation_analysis(df, query, query_lower, numeric_columns)
        
        else:
            return generate_helpful_response(df, query)
        
    except Exception as e:
        return f"Analysis error: {str(e)}\n\nDataset info: {df.shape[0]} rows, {df.shape[1]} columns"

def detect_date_columns(df):
    """Detect potential date columns"""
    date_columns = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col].head(10), errors='raise')
            date_columns.append(col)
        except:
            continue
    return date_columns

def handle_visualization_analysis(df, query, numeric_cols, text_cols):
    """Provide visualization insights"""
    return f"""📊 **Visualization Analysis for:** {query}

**Recommended Charts:**
{f"• Histogram: {numeric_cols[0]} distribution" if numeric_cols else ""}
{f"• Bar Chart: {text_cols[0]} categories" if text_cols else ""}
{f"• Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}" if len(numeric_cols) > 1 else ""}

**Data Summary:**
{df.describe().to_string() if numeric_cols else "No numeric data for statistics"}
"""

def handle_comprehensive_analysis(df, query, numeric_cols, text_cols, date_cols):
    """Provide comprehensive data analysis"""
    analysis = f"📈 **Comprehensive Analysis for:** {query}\n\n"
    
    # Statistical summary
    if numeric_cols:
        analysis += "**📊 Statistical Summary:**\n"
        for col in numeric_cols:
            stats = df[col].agg(['mean', 'std', 'min', 'max'])
            analysis += f"• {col}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Range=[{stats['min']}-{stats['max']}]\n"
    
    # Categorical analysis
    if text_cols:
        analysis += "\n**📋 Categorical Analysis:**\n"
        for col in text_cols:
            top_3 = df[col].value_counts().head(3)
            analysis += f"• {col}: {df[col].nunique()} unique values, Top 3: {dict(top_3)}\n"
    
    # Correlation analysis
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        analysis += f"\n**🔗 Correlations:**\n{corr_matrix.to_string()}\n"
    
    return analysis

def handle_date_specific_analysis(df, query, query_lower, date_cols, numeric_cols):
    """Handle date-specific queries"""
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', query)
    if not dates:
        return "Could not extract valid date. Use format like '2023-01-02'"
    
    date_col = date_cols[0]
    target_date = dates[0]
    
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col]).dt.strftime('%Y-%m-%d')
        filtered_df = df_copy[df_copy[date_col] == target_date]
        
        if filtered_df.empty:
            return f"No data found for {target_date}"
        
        result = f"📅 **Analysis for {target_date}:**\n\n"
        
        if "total" in query_lower and numeric_cols:
            for col in numeric_cols:
                total = filtered_df[col].sum()
                result += f"• Total {col}: {total:,}\n"
        
        result += f"\n**Records ({len(filtered_df)}):**\n{filtered_df.to_string(index=False)}"
        return result
        
    except Exception as e:
        return f"Date analysis error: {str(e)}"

def handle_aggregation_analysis(df, query, query_lower, numeric_cols):
    """Handle aggregation queries"""
    result = "📊 **Aggregation Analysis:**\n\n"
    
    for col in numeric_cols:
        if "total" in query_lower or "sum" in query_lower:
            result += f"• Total {col}: {df[col].sum():,}\n"
        if "average" in query_lower or "mean" in query_lower:
            result += f"• Average {col}: {df[col].mean():.2f}\n"
        if "max" in query_lower:
            result += f"• Maximum {col}: {df[col].max():,}\n"
        if "min" in query_lower:
            result += f"• Minimum {col}: {df[col].min():,}\n"
    
    return result

def generate_helpful_response(df, query):
    """Generate helpful response for unmatched queries"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    return f"""🤔 **Query:** "{query}"

**💡 What I can help with:**

**Basic Queries (⚡ Fast Local):**
- "Show first 5 rows" 
- "Describe the data"
- "What columns are available?"

**Moderate Queries (🔄 Hybrid):**
- "Total sales" / "Average sales"
- "Data for specific date"
- "Filter and summarize"

**Complex Queries (💎 Advanced AI):**
- "Analyze trends and patterns"
- "Provide insights and recommendations" 
- "Compare and correlate data"

**📊 Your Dataset:**
- {df.shape[0]} rows × {df.shape[1]} columns
- Numeric: {numeric_cols}
- Text: {text_cols}

**Sample:** 
{df.head(2).to_string(index=False)}
"""

def extract_number_from_query(query):
    """Extract number from query"""
    numbers = re.findall(r'\d+', query)
    return int(numbers[0]) if numbers else None

def run_agent(query):
    """Main function with hybrid LLM strategy"""
    
    # Check data file
    if not os.path.exists("data/sales_data.csv"):
        st.error("❌ sales_data.csv not found in /data folder!")
        return "Error: Data file not found."
            
    # Load data
    try:
        df = pd.read_csv("data/sales_data.csv")
        st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        with st.expander("📊 Data Preview"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Columns & Types:**")
                for col in df.columns:
                    st.write(f"• {col} ({df[col].dtype})")
            with col2:
                st.write("**Sample Data:**")
                st.dataframe(df.head())
                
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return "Error loading data."
        
    # Route query
    sales_keywords = ["sales", "data", "df", "total", "average", "analyze", "show", "product", "region"]
    is_sales_query = any(keyword in query.lower() for keyword in sales_keywords)
        
    if is_sales_query:
        st.info("🔍 Routing to hybrid LLM analysis...")
        return execute_hybrid_agent_strategy(query, df)
    else:
        # Knowledge base search
        st.info("📚 Searching knowledge base...")
        if os.path.exists("./chroma_db"):
            try:
                embeddings = HuggingFaceEmbeddings()
                vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
                docs = vector_store.similarity_search(query, k=3)
                if docs:
                    return "📖 **Knowledge Base Results:**\n\n" + "\n\n".join([f"**{i+1}.** {doc.page_content}" for i, doc in enumerate(docs)])
                else:
                    return "❓ No relevant information found in knowledge base."
            except Exception as e:
                return f"❌ Knowledge base error: {str(e)}"
        else:
            return "📝 No knowledge base available."

# Streamlit UI
st.set_page_config(page_title="Hybrid AI Agents", page_icon="🤖", layout="wide")

st.title("🤖 Hybrid AI Data Analysis Agents")
st.markdown("*Intelligent routing: Ollama for speed 🦙 • Gemini for complexity 💎*")

# Initialize LLMs and show status
ollama_llm, gemini_llm = initialize_llms()

# Display LLM status
col1, col2 = st.columns(2)
with col1:
    if ollama_llm:
        st.success("🦙 **Ollama Ready** - Fast Local Analysis")
    else:
        st.error("❌ **Ollama Unavailable**")
        
with col2:
    if gemini_llm:
        st.success("💎 **Gemini Ready** - Advanced AI Analysis") 
    else:
        st.warning("⚠️ **Gemini Unavailable** - Set API key")

# Strategy explanation
with st.expander("🎯 How Hybrid Strategy Works"):
    st.markdown("""
    **🚀 Query Classification:**
    - **BASIC** → 🦙 Ollama (Fast, Local): Simple data operations
    - **MODERATE** → 🔄 Hybrid: Try Ollama first, fallback to Gemini  
    - **COMPLEX** → 💎 Gemini (Advanced): Deep analysis, insights, trends
    
    **✅ Benefits:**
    - ⚡ Faster responses for simple queries
    - 🧠 Advanced AI for complex analysis  
    - 💰 Cost optimization (less API usage)
    - 🔄 Reliable fallbacks
    """)

# Example queries
with st.expander("💡 Example Queries by Complexity"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🦙 BASIC (Ollama)**
        - `Show first 5 rows`
        - `What columns exist?`
        - `Describe the data`
        - `Dataset shape`
        """)
    
    with col2:
        st.markdown("""
        **🔄 MODERATE (Hybrid)**
        - `Total sales`
        - `Average by region`
        - `Sales on 2023-01-02`
        - `Product with sales 100`
        """)
    
    with col3:
        st.markdown("""
        **💎 COMPLEX (Gemini)**
        - `Analyze sales trends`
        - `Find patterns and insights`
        - `Recommendations for optimization`
        - `Correlations and predictions`
        """)

# Main query input
query = st.text_input(
    "🗣️ **Ask your question:**",
    placeholder="e.g., 'Analyze sales trends' or 'Show first 5 rows'",
    help="System will automatically choose the best AI for your query"
)

# Process query
if query:
    with st.spinner("🧠 Processing with hybrid AI strategy..."):
        try:
            response = run_agent(query)
            
            st.markdown("---")
            st.markdown("### 📝 Analysis Results:")
            
            if len(response) > 1000:
                st.text_area("Detailed Response:", response, height=400)
            else:
                st.markdown(response)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Enhanced sidebar
with st.sidebar:
    st.markdown("## 🔧 System Status")
    
    # Data status
    if os.path.exists("data/sales_data.csv"):
        st.success("✅ **Data File Ready**")
        try:
            df = pd.read_csv("data/sales_data.csv")
            st.write(f"📊 **{df.shape[0]}** rows × **{df.shape[1]}** columns")
            st.caption(f"Columns: {', '.join(df.columns.tolist())}")
        except:
            st.warning("⚠️ Data file issues")
    else:
        st.error("❌ **sales_data.csv missing**")
    
    # LLM status
    st.markdown("### 🤖 AI Models")
    if ollama_llm:
        st.success("🦙 Ollama Active")
    else:
        st.error("❌ Ollama Offline")
        
    if gemini_llm:
        st.success("💎 Gemini Active")
    else:
        st.warning("⚠️ Gemini API Key Needed")
    
    # Knowledge base
    if os.path.exists("./chroma_db"):
        st.success("✅ Knowledge Base Ready")
    else:
        st.info("ℹ️ No Knowledge Base")
    
    st.markdown("---")
    st.markdown("### ⚡ Performance Tips")
    st.markdown("""
    - Use specific keywords for better routing
    - Basic queries are fastest with Ollama
    - Complex analysis uses advanced Gemini AI
    - System auto-selects optimal model
    """)