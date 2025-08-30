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

# ---------- Audit & Tracking ----------
class QueryTracker:
    def __init__(self):
        if 'query_stats' not in st.session_state:
            st.session_state.query_stats = {
                'total_queries': 0,
                'heuristic_hits': 0,
                'ollama_calls': 0,
                'gemini_calls': 0,
                'fallback_hits': 0,
                'pandas_direct': 0,
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
        if execution_path == 'heuristic':
            st.session_state.query_stats['heuristic_hits'] += 1
        elif execution_path == 'ollama':
            st.session_state.query_stats['ollama_calls'] += 1
        elif execution_path == 'gemini':
            st.session_state.query_stats['gemini_calls'] += 1
        elif execution_path == 'fallback':
            st.session_state.query_stats['fallback_hits'] += 1
        elif execution_path == 'pandas_direct':
            st.session_state.query_stats['pandas_direct'] += 1
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
    
    def detect_schema_with_llm(self, df: pd.DataFrame, ollama_llm, gemini_llm) -> Dict:
        """Use LLM to intelligently detect and categorize DataFrame schema"""
        
        # Prefer Gemini for schema reasoning, fallback to Ollama, else rule-based
        llm = gemini_llm if gemini_llm else ollama_llm
        if not llm:
            return self._fallback_schema_detection(df)
        
        # Create schema detection prompt
        sample_data = df.head(3).to_string(index=False, max_cols=10)
        dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        schema_prompt = f"""Analyze this DataFrame and provide a JSON schema classification:

DataFrame Info:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Column Types: {dtypes_info}

Sample Data:
{sample_data}

Please return ONLY a valid JSON with this structure:
{{
    "numeric_columns": ["list of numeric columns for calculations"],
    "categorical_columns": ["list of text/categorical columns"],
    "date_columns": ["list of date/datetime columns"],
    "id_columns": ["list of ID/identifier columns"],
    "business_metrics": ["list of business KPI columns like sales, revenue, etc"],
    "groupby_candidates": ["list of columns good for grouping/segmentation"],
    "primary_business_entities": ["main entities like product, customer, region"],
    "schema_insights": "Brief description of what this dataset represents"
}}"""
        try:
            if gemini_llm:
                response = gemini_llm.invoke(schema_prompt)
            else:
                response = ollama_llm.invoke(schema_prompt)
            
            response_text = safe_extract_output(response)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                schema = json.loads(json_match.group())
                st.session_state.detected_schema = schema
                return schema
        except Exception as e:
            st.info(f"LLM schema detection failed: {stringify_exception(e)} -> Using fallback")
        
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

# ---------- Dynamic Heuristics Engine ----------
class DynamicHeuristics:
    def __init__(self, schema: Dict):
        self.schema = schema or {}
        
    def try_heuristic_analysis(self, query: str, df: pd.DataFrame) -> Optional[str]:
        """Apply dynamic heuristics based on detected schema"""
        q = query.lower()
        
        # Basic info queries
        if any(word in q for word in ["head", "first", "show rows"]):
            n = self._extract_number(query) or 5
            n = max(1, min(50, n))
            return f"📊 **First {n} rows:**\n\n{df.head(n).to_string(index=False, max_cols=10)}"
        
        if "shape" in q or "size" in q:
            return f"📏 **Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns"
        
        if "columns" in q or "schema" in q:
            return self._format_schema_info(df)
        
        # Aggregate queries using schema
        if self._is_aggregate_query(q):
            agg = self._handle_aggregates(query, df, q)
            if agg:
                return agg
        
        # Date-based queries
        if self._is_date_query(q) and self.schema.get('date_columns'):
            date_ans = self._handle_date_queries(query, df, q)
            if date_ans:
                return date_ans
        
        # Top/Bottom queries
        if self._is_ranking_query(q):
            rank = self._handle_ranking(query, df, q)
            if rank:
                return rank
        
        # Group by queries
        if self._is_groupby_query(q):
            gb = self._handle_groupby(query, df, q)
            if gb:
                return gb
        
        return None
    
    def _extract_number(self, text: str) -> Optional[int]:
        nums = re.findall(r'\d+', text)
        return int(nums[0]) if nums else None
    
    def _is_aggregate_query(self, q: str) -> bool:
        agg_words = ["total", "sum", "average", "mean", "count", "max", "min"]
        return any(word in q for word in agg_words)
    
    def _is_date_query(self, q: str) -> bool:
        return any(word in q for word in ["on", "date", "day", "month", "year", "between"])
    
    def _is_ranking_query(self, q: str) -> bool:
        return any(word in q for word in ["top", "bottom", "best", "worst", "highest", "lowest"])
    
    def _is_groupby_query(self, q: str) -> bool:
        return any(word in q for word in [" by ", "group", " per ", " each "]) or "group by" in q
    
    def _format_schema_info(self, df: pd.DataFrame) -> str:
        schema = self.schema
        info = ["📋 **Dataset Schema:**", ""]
        
        if schema.get('business_metrics'):
            info.append(f"💰 **Business Metrics:** {', '.join(schema['business_metrics'])}")
        
        if schema.get('groupby_candidates'):
            info.append(f"📊 **Grouping Dimensions:** {', '.join(schema['groupby_candidates'])}")
        
        if schema.get('date_columns'):
            info.append(f"📅 **Date Columns:** {', '.join(schema['date_columns'])}")
        
        info.append(f"\n**All Columns ({len(df.columns)}):**")
        for col in df.columns:
            info.append(f"• {col} ({df[col].dtype})")
        
        return "\n".join(info)
    
    def _handle_aggregates(self, query: str, df: pd.DataFrame, q: str) -> Optional[str]:
        results = ["📊 **Aggregate Analysis:**", ""]
        metrics = self.schema.get('business_metrics', self.schema.get('numeric_columns', []))
        if not metrics:
            return None
        
        for col in metrics[:6]:  # Limit to 6 columns
            if col not in df.columns:
                continue
            try:
                if "total" in q or "sum" in q:
                    results.append(f"• Total {col}: {pd.to_numeric(df[col], errors='coerce').sum():,.2f}")
                if "average" in q or "mean" in q:
                    results.append(f"• Average {col}: {pd.to_numeric(df[col], errors='coerce').mean():,.2f}")
                if "max" in q:
                    results.append(f"• Maximum {col}: {pd.to_numeric(df[col], errors='coerce').max():,.2f}")
                if "min" in q:
                    results.append(f"• Minimum {col}: {pd.to_numeric(df[col], errors='coerce').min():,.2f}")
                if "count" in q:
                    results.append(f"• Count {col}: {df[col].count():,}")
            except Exception:
                continue
                
        return "\n".join(results) if len(results) > 2 else None
    
    def _handle_date_queries(self, query: str, df: pd.DataFrame, q: str) -> str:
        date_cols = self.schema.get('date_columns', [])
        if not date_cols:
            return None
            
        # Extract date from query (YYYY-MM-DD)
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', query)
        if not dates:
            return "Please specify date in YYYY-MM-DD format"
        
        target_date = dates[0]
        date_col = date_cols[0]
        
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            filtered = df_copy[df_copy[date_col].dt.strftime('%Y-%m-%d') == target_date]
            
            if filtered.empty:
                return f"No data found for {target_date}"
            
            results = [f"📅 **Analysis for {target_date} ({len(filtered)} records):**", ""]
            
            # Show key metrics if available
            metrics = self.schema.get('business_metrics', [])
            for metric in metrics[:3]:
                if metric in filtered.columns:
                    total = pd.to_numeric(filtered[metric], errors='coerce').sum()
                    results.append(f"• {metric}: {total:,.2f}")
            
            results.append(f"\n**Sample Records:**\n{filtered.head(5).to_string(index=False, max_cols=8)}")
            return "\n".join(results)
            
        except Exception as e:
            return f"Date analysis error: {stringify_exception(e)}"
    
    def _handle_ranking(self, query: str, df: pd.DataFrame, q: str) -> str:
        n = self._extract_number(query) or 5
        n = max(1, min(50, n))
        
        # Find the metric to rank by
        metrics = self.schema.get('business_metrics', self.schema.get('numeric_columns', []))
        groupby_cols = self.schema.get('groupby_candidates', [])
        
        if not metrics or not groupby_cols:
            return None
        
        try:
            metric_col = metrics[0]  # Use first business metric
            group_col = groupby_cols[0]  # Use first grouping column
            
            s = pd.to_numeric(df[metric_col], errors='coerce')
            tmp = df.assign(_metric=s)
            grouped = tmp.groupby(group_col)["_metric"].sum()
            
            if "top" in q or "best" in q or "highest" in q:
                ranked = grouped.nlargest(n)
                direction = "Top"
            else:  # bottom, worst, lowest
                ranked = grouped.nsmallest(n)
                direction = "Bottom"
            
            results = [f"🏆 **{direction} {n} {group_col} by {metric_col}:**", ""]
            for idx, (name, value) in enumerate(ranked.items(), 1):
                results.append(f"{idx}. {name}: {value:,.2f}")
            return "\n".join(results)
            
        except Exception as e:
            return f"Ranking analysis error: {stringify_exception(e)}"
    
    def _handle_groupby(self, query: str, df: pd.DataFrame, q: str) -> str:
        groupby_cols = self.schema.get('groupby_candidates', [])
        metrics = self.schema.get('business_metrics', self.schema.get('numeric_columns', []))
        
        if not groupby_cols or not metrics:
            return None
        
        try:
            # Find mentioned columns in query
            group_col = None
            metric_col = None
            
            for col in groupby_cols:
                if col.lower() in q:
                    group_col = col
                    break
            
            for col in metrics:
                if col.lower() in q:
                    metric_col = col
                    break
            
            # Use defaults if not found
            group_col = group_col or groupby_cols[0]
            metric_col = metric_col or metrics[0]
            
            # Determine aggregation type
            s = pd.to_numeric(df[metric_col], errors='coerce')
            tmp = df.assign(_metric=s)
            if "total" in q or "sum" in q:
                grouped = tmp.groupby(group_col)["_metric"].sum()
                label = "sum"
            elif "average" in q or "mean" in q:
                grouped = tmp.groupby(group_col)["_metric"].mean()
                label = "mean"
            elif "count" in q:
                grouped = df.groupby(group_col).size()
                label = "count"
            else:
                grouped = tmp.groupby(group_col)["_metric"].sum()
                label = "sum"
            
            results = [f"📈 **{metric_col} ({label}) by {group_col}:**", ""]
            head = grouped.sort_values(ascending=False).head(10)
            for name, value in head.items():
                if isinstance(value, (int, float)):
                    results.append(f"• {name}: {value:,.2f}")
                else:
                    results.append(f"• {name}: {value:,}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Group-by analysis error: {stringify_exception(e)}"

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
            verbose=True,  # Changed to True for debugging
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=100,  # Reduced from 100 to prevent long waits
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            return_intermediate_steps=False,
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
            verbose=True,  # Changed to True for debugging
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=100,  # Reduced from 100 to prevent long waits
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            return_intermediate_steps=False,
        )
    except Exception as e:
        st.warning(f"Gemini agent creation failed: {stringify_exception(e)}")
        return None

# ---------- Query routing with schema-aware classification ----------

def classify_query_complexity(query: str, schema: Dict) -> str:
    """Enhanced classification using schema information with improved pattern matching"""
    # Normalize query for better matching
    q = query.lower().strip()
    q = ' '.join(q.split())  # Normalize whitespace
    
    # Schema-aware indicators
    business_entities = schema.get('primary_business_entities', []) or []
    business_metrics = schema.get('business_metrics', []) or []
    
    # Prepare normalized business metrics for matching
    normalized_metrics = [metric.lower() for metric in business_metrics]
    
    # Simple pattern matchers
    def fuzzy_match(pattern: str, text: str) -> bool:
        """Simple fuzzy matching for common variations"""
        pattern = pattern.lower()
        text = text.lower()
        # Handle common typos and variations
        variations = [
            pattern,
            pattern.replace(' ', ''),  # Remove spaces
            ''.join(c for c in pattern if c.isalnum()),  # Alphanumeric only
        ]
        return any(var in text for var in variations)
    
    # Define query patterns with variations
    SIMPLE_PATTERNS = {
        'basic_stats': [
            r'total [\w\s]+ for',  # Specific column totals
            r'sum of [\w\s]+',
            r'show me [\w\s]+ total',
            r'what is the total',
            r'calculate total',
        ],
        'data_view': [
            r'show|display|list|get',
            r'what are|what is',
            r'give me',
        ],
        'basic_info': [
            r'shape|size|count|length',
            r'columns|schema|structure',
        ]
    }
    
    MEDIUM_PATTERNS = {
        'aggregations': [
            r'total by|sum by|average by',
            r'group|aggregate',
            r'breakdown of',
            r'distribution of',
        ],
        'filtering': [
            r'filter|where|which|when',
            r'find all|show all|get all',
            r'greater than|less than|between',
        ],
        'rankings': [
            r'top \d+|bottom \d+',
            r'highest|lowest',
            r'best|worst',
        ]
    }
    
    COMPLEX_PATTERNS = {
        'analysis': [
            r'analyze|analyse|analysis',
            r'insight|pattern|trend',
            r'correlation|relationship',
            r'predict|forecast|estimate',
            r'describe|elaborate|explain|detail',
        ],
        'visualization': [
            r'chart|plot|graph|visual',
            r'pie|bar|histogram|scatter',
        ],
        'advanced': [
            r'compare|contrast|versus|vs',
            r'impact of|effect of',
            r'segment|cluster|classify',
            r'month over month|year over year',
            r'over time|time series',
        ]
    }
    
    # Check for specific simple cases first (exact matches with business metrics)
    for metric in normalized_metrics:
        simple_metric_patterns = [
            f"total {metric}",
            f"sum of {metric}",
            f"show {metric}",
            f"{metric} total",
        ]
        if any(fuzzy_match(pattern, q) for pattern in simple_metric_patterns):
            return "SIMPLE"
    
    # Visualization queries are always complex
    for pattern in COMPLEX_PATTERNS['visualization']:
        if fuzzy_match(pattern, q):
            return "COMPLEX"
    
    # Check for multiple entity/metric mentions
    mentions_multiple_entities = sum(1 for entity in business_entities if entity.lower() in q) > 1
    mentions_multiple_metrics = sum(1 for metric in business_metrics if metric.lower() in q) > 1
    
    if mentions_multiple_entities or mentions_multiple_metrics:
        return "COMPLEX"
    
    # Check for time-based analysis
    time_patterns = [
        r'trend', r'over time', r'growth', r'change',
        r'month', r'year', r'daily', r'weekly'
    ]
    if any(fuzzy_match(pattern, q) for pattern in time_patterns):
        return "COMPLEX"
    
    # Check pattern categories
    for patterns in COMPLEX_PATTERNS.values():
        if any(fuzzy_match(pattern, q) for pattern in patterns):
            return "COMPLEX"
    
    for patterns in MEDIUM_PATTERNS.values():
        if any(fuzzy_match(pattern, q) for pattern in patterns):
            return "MEDIUM"
    
    for patterns in SIMPLE_PATTERNS.values():
        if any(fuzzy_match(pattern, q) for pattern in patterns):
            return "SIMPLE"
    
    # Length-based fallback
    words = q.split()
    if len(words) > 12 or len(q) > 100:
        return "COMPLEX"
    if len(words) > 6:
        return "MEDIUM"
    
    return "SIMPLE"  # Default to simple for short, unclassified queries

def preprocess_query_for_llm(query: str, df: pd.DataFrame, llm_type: str, schema: Dict) -> str:
    """Enhanced preprocessing with schema context and chart handling"""
    q = query.lower()
    
    # Check if this is a chart query
    chart_keywords = ['chart', 'plot', 'graph', 'visuali', 'pie', 'bar', 'histogram', 'scatter', 'line chart']
    is_chart_query = any(keyword in q for keyword in chart_keywords)
    
    if llm_type == "OLLAMA":
        schema_context = f"""
Schema Info:
- Business Metrics: {schema.get('business_metrics', [])}
- Group-by Columns: {schema.get('groupby_candidates', [])}
- Date Columns: {schema.get('date_columns', [])}
"""
        return f"""You are a helpful data analyst. Answer the user's question about the dataset using python/pandas.

{schema_context}
Question: {query}

DataFrame info: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {list(df.columns)}

Provide a clear, concise answer. Use the python_repl_ast tool to analyze the data when needed."""

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

Use the python_repl_ast tool for complex pandas operations."""
    return query

# ---------- Pandas helpers ----------

def try_direct_pandas_operations(query: str, df: pd.DataFrame) -> Optional[str]:
    """Direct pandas operations without LLM"""
    q = query.lower()
    
    if "describe" in q:
        return df.describe(include='all', datetime_is_numeric=True).to_string()
    
    if "info" in q:
        mem_mb = df.memory_usage(deep=True).sum() / 1024**2
        return f"Dataset Info:\n{df.dtypes.to_string()}\n\nMemory usage: {mem_mb:.2f} MB"
    
    if "nunique" in q or "unique" in q:
        return f"Unique values per column:\n{df.nunique(dropna=True).to_string()}"

    if "columns" in q:
        cols = [f"{c} ({df[c].dtype})" for c in df.columns]
        return "Columns:\n" + "\n".join(cols)
    
    return None

def run_pandas_agent(query: str, df: pd.DataFrame, gemini_llm=None, ollama_llm=None) -> Optional[str]:
    """
    Execute query using a Pandas agent with consistent response cleaning.
    Prefers Gemini if available, otherwise Ollama.
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
        
        # Execute with timeout protection
        with st.spinner(f"{agent_type} processing..."):
            res = agent.invoke({"input": query})
            output = safe_extract_output(res)
            
            # Clean response if it came from Gemini
            if agent_type == "Gemini":
                output = clean_gemini_response(output)
            
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

# ---------- Execution strategies with fallback ----------

def enhanced_fallback_analysis(df: pd.DataFrame, query: str, schema: Dict) -> str:
    """Enhanced fallback using schema information"""
    try:
        heuristics = DynamicHeuristics(schema)
        result = heuristics.try_heuristic_analysis(query, df)
        if result:
            return result
        return generate_helpful_response(df, query, schema)
    except Exception as e:
        return f"Analysis error: {stringify_exception(e)}\n\nDataset: {df.shape[0]} rows, {df.shape[1]} cols"

def generate_helpful_response(df: pd.DataFrame, query: str, schema: Dict) -> str:
    """Generate helpful response using schema context"""
    sample = df.head(2).to_string(index=False, max_cols=8)
    business_metrics = schema.get('business_metrics', [])
    groupby_cols = schema.get('groupby_candidates', [])
    
    return f"""🤔 **Query:** "{query}"

**💡 What I can analyze:**

**⚡ Simple (Heuristics + Local):**
- Show first N rows, describe data, column info
- Basic aggregates: total, average, count
- Simple filtering by date or category

**🔄 Medium (Ollama + Fallbacks):**
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

def execute_simple_strategy(query: str, df: pd.DataFrame, schema: Dict, ollama_llm) -> Tuple[str, str]:
    """Simple queries: Heuristics → Pandas Direct → Ollama (with proper agent checking)"""
    start_time = time.time()
    
    # 1. Try dynamic heuristics first
    heuristics = DynamicHeuristics(schema)
    heuristic_result = heuristics.try_heuristic_analysis(query, df)
    
    if heuristic_result:
        execution_time = time.time() - start_time
        tracker.log_query(query, "SIMPLE", "heuristic", True, execution_time)
        return heuristic_result, "heuristic"
    
    # 2. Try Pandas Direct
    try:
        result = try_direct_pandas_operations(query, df)
        if result:
            execution_time = time.time() - start_time
            tracker.log_query(query, "SIMPLE", "pandas_direct", True, execution_time)
            return f"Direct Pandas Analysis:\n\n{result}", "pandas_direct"
    except Exception:
        pass
    
    # 3. Try Ollama agent - CHECK IF AGENT IS ACTUALLY CREATED
    if ollama_llm:
        try:
            agent = create_ollama_agent(df, ollama_llm)
            if agent:  # Only proceed if agent was successfully created
                simple_prompt = f"Analyze the data and answer: {query}"
                try:
                    res = agent.invoke({"input": simple_prompt})
                    output = safe_extract_output(res)
                    if output.strip() and len(output) > 20:
                        execution_time = time.time() - start_time
                        tracker.log_query(query, "SIMPLE", "ollama", True, execution_time)
                        return f"Ollama Analysis:\n\n{output}", "ollama"
                except (OutputParserException, ValueError) as e:
                    st.info(f"Ollama parsing error: {stringify_exception(e)}")
            else:
                st.info("Ollama agent creation failed → Skipping to fallback")
        except Exception as e:
            st.info(f"Ollama failed: {stringify_exception(e)}")
    
    # 4. Final fallback
    execution_time = time.time() - start_time
    tracker.log_query(query, "SIMPLE", "fallback", False, execution_time)
    return generate_helpful_response(df, query, schema), "fallback"

# ---------- Fixed Medium Strategy ----------

def execute_medium_strategy(query: str, df: pd.DataFrame, schema: Dict, ollama_llm, gemini_llm) -> Tuple[str, str]:
    """Medium queries with proper agent validation"""
    start_time = time.time()
    
    # 1. Try Ollama first - CHECK IF AGENT IS CREATED
    if ollama_llm:
        try:
            agent = create_ollama_agent(df, ollama_llm)
            if agent:  # Only proceed if agent exists
                simple_prompt = f"Using the dataframe 'df', {query}. Show the result."
                try:
                    res = agent.invoke({"input": simple_prompt})
                    output = safe_extract_output(res)
                    
                    if (output.strip() and 
                        len(output) > 30 and 
                        not any(err in output.lower() for err in ["error", "failed", "cannot"])):
                        
                        execution_time = time.time() - start_time
                        tracker.log_query(query, "MEDIUM", "ollama", True, execution_time)
                        return f"Ollama Analysis:\n\n{output}", "ollama"
                except (OutputParserException, ValueError) as e:
                    st.info(f"Ollama parsing failed: {stringify_exception(e)}")
            else:
                st.info("Ollama agent creation failed → Trying Pandas Agent...")
        except Exception as e:
            st.info(f"Ollama setup failed: {stringify_exception(e)} → Trying Pandas Agent...")

    # 2. Try Pandas Agent
    try:
        pandas_res = run_pandas_agent(query, df, gemini_llm=gemini_llm, ollama_llm=ollama_llm)
        if pandas_res and len(pandas_res.strip()) > 20:
            pandas_res = clean_gemini_response(pandas_res)
            execution_time = time.time() - start_time
            tracker.log_query(query, "MEDIUM", "pandas-agent", True, execution_time)
            return f"Pandas Agent Analysis:\n\n{pandas_res}", "pandas-agent"
    except Exception:
        st.info("Pandas Agent failed → Using enhanced fallback...")

    # 3. Final fallback
    execution_time = time.time() - start_time
    tracker.log_query(query, "MEDIUM", "fallback", False, execution_time)
    return enhanced_fallback_analysis(df, query, schema), "fallback"

# ---------- Add Debug Function for Ollama Testing ----------

def test_ollama_connection(ollama_llm) -> bool:
    """Test if Ollama is actually working"""
    if not ollama_llm:
        return False
    try:
        test_response = ollama_llm.invoke("Say 'TEST SUCCESS' if you can read this.")
        response_text = safe_extract_output(test_response)
        return "TEST SUCCESS" in response_text.upper()
    except Exception as e:
        st.warning(f"Ollama connection test failed: {stringify_exception(e)}")
        return False

# ---------- Updated Execute Complex Strategy (with chart validation) ----------

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
    
    # 3. Final fallback → Enhanced heuristics / helper text
    execution_time = time.time() - start_time
    tracker.log_query(query, "COMPLEX", "fallback", False, execution_time)
    return enhanced_fallback_analysis(df, query, schema), "fallback"


def execute_hybrid_strategy(query: str, df: pd.DataFrame, schema: Dict, ollama_llm, gemini_llm) -> str:
    """Main router with schema-aware execution"""
    complexity = classify_query_complexity(query, schema)
    st.info(f"🎯 Query classified as **{complexity}** | Schema-aware routing")
    
    if complexity == "SIMPLE":
        st.info("⚡ Route: Heuristics → Pandas → Ollama")
        result, path = execute_simple_strategy(query, df, schema, ollama_llm)
    elif complexity == "MEDIUM":
        st.info("🔄 Route: Ollama → Pandas Agent → Gemini")
        result, path = execute_medium_strategy(query, df, schema, ollama_llm, gemini_llm)
    else:  # COMPLEX
        st.info("💎 Route: Gemini → Pandas Agent → Enhanced fallbacks")
        result, path = execute_complex_strategy(query, df, schema, gemini_llm)
    
    # Add execution path info to result
    path_emoji = {
        "heuristic": "⚡", "pandas_direct": "🐼", "ollama": "🦙", 
        "gemini": "💎", "fallback": "🔧", "pandas-agent": "🐼"
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
st.caption("Dynamic schema detection • Smart heuristics • Multi-layer fallbacks • Audit tracking")

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
            st.metric("⚡ Heuristics", stats['heuristic_hits'])
            st.metric("🦙 Ollama", stats['ollama_calls'])
        with col2:
            st.metric("💎 Gemini", stats['gemini_calls']) 
            st.metric("🔧 Fallbacks", stats['fallback_hits'])
        
        st.metric("🐼 Pandas (Direct)", stats['pandas_direct'])
        st.metric("🐼 Pandas (Agent)", stats['pandas_agent'])
        
        # Efficiency metrics
        heuristic_rate = (stats['heuristic_hits'] / stats['total_queries']) * 100
        st.metric("Efficiency", f"{heuristic_rate:.1f}%", help="% queries handled by fast heuristics")
        
        # Success rate
        successful_queries = sum([
            stats['heuristic_hits'], stats['ollama_calls'], 
            stats['gemini_calls'], stats['pandas_direct'], stats['pandas_agent']
        ])
        success_rate = (successful_queries / stats['total_queries']) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%", help="% queries with meaningful responses")
    else:
        st.info("No queries yet")
    
    # Performance tips
    st.markdown("---")
    st.markdown("### ⚡ Optimization Tips")
    st.write("- Schema auto-detected for smart routing")
    st.write("- Heuristics handle common queries instantly")
    st.write("- Multi-layer fallbacks ensure reliability")

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
- **SIMPLE** → ⚡ Heuristics (instant) → 🐼 Pandas → 🦙 Ollama  
- **MEDIUM** → 🦙 Ollama (local) → 🐼 Pandas Agent → 💎 Gemini  
- **COMPLEX** → 💎 Gemini (cloud) → 🐼 Pandas Agent → Enhanced fallbacks

**🔄 Multi-Layer Fallbacks:**
1. Fast heuristics using detected schema  
2. Direct pandas operations  
3. Local LLM processing (Ollama)  
4. Cloud LLM reasoning (Gemini)  
5. Enhanced rule-based analysis

**📊 Schema-Driven Optimization:**
- Auto-detects business metrics, dimensions, date columns  
- Routes queries to optimal execution layer  
- Tracks performance for continuous improvement
        """)

    # Dynamic examples based on detected schema
    with st.expander("💡 Schema-Optimized Query Examples"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⚡ SIMPLE (Heuristics)**")
            examples = ["Show first 10 rows", "Dataset shape", "Column information", "describe"]
            if schema.get('business_metrics'):
                examples.append(f"Total {schema['business_metrics'][0]}")
            if schema.get('groupby_candidates'):
                examples.append(f"Top 5 {schema['groupby_candidates'][0]}")
            for ex in examples:
                st.write(f"• {ex}")
        
        with col2:
            st.markdown("**🔄 MEDIUM (Ollama+)**") 
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
        
        with col3:
            st.markdown("**💎 COMPLEX (Gemini)**")
            examples = ["Analyze trends and patterns", "Correlations and insights"]
            if schema.get('primary_business_entities') and len(schema['primary_business_entities']) > 1:
                entities = schema['primary_business_entities'][:2]
                examples.extend([f"Compare {entities[0]} vs {entities[1]}", f"Predict {entities[0]} performance"])
            examples.extend(["Business recommendations", "What-if scenarios"])
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
    options=["Auto (Smart Detection)", "Sales Data", "Knowledge Base"],
    horizontal=True,
    index=0,
    help="Auto mode intelligently routes between data analysis and knowledge base"
)

# Query execution
if query:
    st.markdown("---")
    with st.spinner("🧠 Processing with schema-aware routing..."):
        try:
            if mode == "Knowledge Base":
                response = kb_search(query)
            else:
                # Smart routing logic
                sales_keywords = [
                    "sales", "revenue", "product", "region", "customer",
                    "order", "data", "analyze", "total", "average", "trend",
                    "price", "profit", "margin", "category", "month", "year"
                ]
                is_data_query = (mode == "Sales Data") or any(k in query.lower() for k in sales_keywords)
                
                if is_data_query and df is not None and schema:
                    response = execute_hybrid_strategy(query, df, schema, ollama_llm, gemini_llm)
                else:
                    response = kb_search(query)

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
                    "heuristic": "⚡", "ollama": "🦙", "gemini": "💎",
                    "fallback": "🔧", "pandas_direct": "🐼", "pandas-agent": "🐼"
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
        efficiency = (stats['heuristic_hits'] / stats['total_queries']) * 100
        st.caption(f"⚡ **Efficiency:** {efficiency:.1f}%")