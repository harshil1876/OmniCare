import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.schema import OutputParserException
import pandas as pd
import os
import re

# Initialize the LLM with parameters optimized for agent efficiency
llm = OllamaLLM(
    model="llama2:latest",
    temperature=0,  # Most deterministic for consistent results
    num_predict=400,  # Enough tokens for detailed analysis but not too much
    top_p=0.9,
    repeat_penalty=1.1,
    # Optimize for faster response times
    num_ctx=2048,  # Reasonable context window
    top_k=20,  # Reduce top_k for faster generation
)

def preprocess_query_for_agent(query, df):
    """Enhanced preprocessing with explicit ReAct format guidance"""
    query_lower = query.lower()
    
    # Add explicit format instructions to guide the LLM
    format_instruction = """
IMPORTANT: You must use the python_repl_ast tool to execute code. Follow this format:

Action: python_repl_ast
Action Input: df.head()

"""
    
    # Handle visualization requests
    if any(viz_word in query_lower for viz_word in ["plot", "graph", "chart", "visualize", "show"]):
        return f"{format_instruction}Create a visualization to answer: {query}. Use python_repl_ast tool with matplotlib or pandas plotting."
    
    # Handle analysis requests
    elif any(analysis_word in query_lower for analysis_word in ["analyze", "analysis", "insights", "trends", "patterns"]):
        return f"{format_instruction}Perform analysis to answer: {query}. Use python_repl_ast tool with pandas operations."
    
    # Handle summary requests
    elif any(summary_word in query_lower for summary_word in ["summary", "summarize", "overview", "describe"]):
        return f"{format_instruction}Provide summary for: {query}. Use python_repl_ast tool with pandas describe() and other functions."
    
    # Handle specific date queries with explicit code example
    elif "total sales" in query_lower and any(date in query for date in ["2023", "2024", "20"]):
        import re
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', query)
        if dates:
            return f"""{format_instruction}Calculate total sales for {dates[0]}. Use python_repl_ast tool like this example:

Action: python_repl_ast  
Action Input: df[df['Date'] == '{dates[0]}']['Sales'].sum()
"""
    
    # Handle general aggregation with examples
    elif "total" in query_lower:
        return f"""{format_instruction}Calculate totals to answer: {query}. Use python_repl_ast tool with pandas sum() function."""
    
    elif "average" in query_lower or "mean" in query_lower:
        return f"""{format_instruction}Calculate average to answer: {query}. Use python_repl_ast tool with pandas mean() function."""
    
    # For simple queries, provide very explicit guidance
    else:
        return f"""{format_instruction}Answer this question: {query}

Use the python_repl_ast tool to execute pandas code. Example:
Action: python_repl_ast
Action Input: df.head()
"""

def create_robust_agent_with_retries(df, max_retries=1):
    """Create agent with explicit ReAct format instructions"""
    try:
        st.write("🔧 Creating intelligent pandas agent with ReAct guidance...")
        
        # Create custom prompt template that emphasizes tool usage
        from langchain.prompts import PromptTemplate
        
        # Try to create agent with explicit tool guidance
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
            max_iterations=50,  # Reasonable number of iterations
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors="You must use the python_repl_ast tool to execute code. Format: Action: python_repl_ast\nAction Input: [your pandas code]",
            return_intermediate_steps=True,
        )
        
        st.success("✅ Agent ready for analysis!")
        return agent
        
    except Exception as e:
        # Fallback to simplest configuration
        try:
            st.write("🔧 Using simplified agent configuration...")
            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                verbose=False,
                allow_dangerous_code=True,
                max_iterations=25,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            st.success("✅ Basic agent created!")
            return agent
        except Exception as e2:
            st.error(f"❌ Agent creation failed: {str(e2)}")
            return None

def execute_agent_with_fallback(agent, query, df, max_retries=1):
    """Execute agent with single focused attempt and quick fallback"""
    
    try:
        st.write("🔍 Processing with AI agent...")
        
        # Single focused attempt with the preprocessed query
        response = agent.invoke({"input": query})
        
        # Extract and analyze the response
        if isinstance(response, dict):
            result = response.get("output", str(response))
            
            # Check intermediate steps for useful content
            if "intermediate_steps" in response and (not result or "not a valid tool" in result):
                steps = response.get("intermediate_steps", [])
                for step in steps:
                    if len(step) > 1:
                        step_output = str(step[1])
                        if len(step_output) > 20 and "not a valid tool" not in step_output:
                            result = step_output
                            break
        else:
            result = str(response)
        
        # Check if we got a meaningful result
        if result and len(result.strip()) > 15:
            # Check for tool format errors
            if "not a valid tool" in result.lower() or "try one of [python_repl_ast]" in result.lower():
                st.write("🔄 LLM needs guidance on tool format, switching to direct analysis...")
                return None  # Trigger fallback
            
            # Check for other issues
            elif any(error in result.lower() for error in ["error", "failed", "could not"]):
                st.write("🔄 Agent encountered technical issues, using reliable analysis...")
                return None  # Trigger fallback
            
            # We have a potentially good result
            else:
                st.success("✅ AI analysis complete!")
                return result
        
        # If we reach here, no meaningful result was obtained
        st.write("🔄 Agent response incomplete, using direct analysis...")
        return None
        
    except Exception as e:
        st.write(f"🔄 Agent processing needs adjustment, using reliable fallback...")
        return None

def enhanced_fallback_analysis(df, query):
    """Enhanced fallback that works with any kind of data structure"""
    try:
        query_lower = query.lower()
        
        # Dynamically detect column names and types
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        date_columns = []
        
        # Try to detect date columns
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(), errors='raise')
                date_columns.append(col)
            except:
                continue
        
        # Handle visualization requests
        if any(viz_word in query_lower for viz_word in ["plot", "graph", "chart", "visualize"]):
            return handle_visualization_request(df, query, numeric_columns, text_columns)
        
        # Handle analysis requests
        elif any(analysis_word in query_lower for analysis_word in ["analyze", "analysis", "insights", "trends"]):
            return handle_analysis_request(df, query, numeric_columns, text_columns, date_columns)
        
        # Handle summary requests
        elif any(summary_word in query_lower for summary_word in ["summary", "summarize", "overview"]):
            return handle_summary_request(df, numeric_columns, text_columns, date_columns)
        
        # Handle date-specific queries (works with any date column)
        elif any(date_word in query_lower for date_word in ["on", "date", "day", "2023", "2024"]) and date_columns:
            return handle_dynamic_date_query(df, query, query_lower, date_columns, numeric_columns)
        
        # Handle aggregation queries (works with any numeric column)
        elif any(agg_word in query_lower for agg_word in ["total", "sum", "average", "mean", "max", "min"]) and numeric_columns:
            return handle_dynamic_aggregation_query(df, query, query_lower, numeric_columns)
        
        # Handle basic data exploration
        elif any(word in query_lower for word in ["head", "first", "top"]):
            n = extract_number_from_query(query) or 5
            return f"First {n} rows of data:\n\n{df.head(n).to_string(index=False)}"
        
        elif "shape" in query_lower or "size" in query_lower:
            return f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns"
        
        elif "columns" in query_lower:
            col_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                unique_count = df[col].nunique()
                col_info.append(f"{col} ({col_type}, {unique_count} unique values)")
            return f"Dataset columns:\n" + "\n".join([f"• {info}" for info in col_info])
        
        elif "describe" in query_lower:
            description = f"Dataset Overview:\n\n"
            description += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
            
            if numeric_columns:
                description += f"Numeric columns summary:\n{df[numeric_columns].describe().to_string()}\n\n"
            
            if text_columns:
                description += f"Text columns info:\n"
                for col in text_columns:
                    description += f"• {col}: {df[col].nunique()} unique values\n"
            
            return description
        
        # Generic column-based search
        else:
            return handle_generic_query(df, query, numeric_columns, text_columns, date_columns)
        
    except Exception as e:
        return f"Error in analysis: {str(e)}\n\nDataset info:\n- Columns: {', '.join(df.columns)}\n- Shape: {df.shape}"

def handle_visualization_request(df, query, numeric_columns, text_columns):
    """Handle visualization requests"""
    viz_suggestion = f"📊 **Visualization Analysis for:** {query}\n\n"
    
    if numeric_columns:
        viz_suggestion += f"**Suggested visualizations:**\n"
        viz_suggestion += f"• Histogram of {numeric_columns[0]} values\n"
        viz_suggestion += f"• Box plot to show distribution\n"
        
        if len(numeric_columns) > 1:
            viz_suggestion += f"• Scatter plot between {numeric_columns[0]} and {numeric_columns[1]}\n"
        
        if text_columns:
            viz_suggestion += f"• Bar chart of {numeric_columns[0]} by {text_columns[0]}\n"
    
    viz_suggestion += f"\n**Sample data for visualization:**\n{df.head().to_string(index=False)}"
    
    return viz_suggestion

def handle_analysis_request(df, query, numeric_columns, text_columns, date_columns):
    """Handle analysis requests"""
    analysis = f"📈 **Data Analysis for:** {query}\n\n"
    
    # Basic statistics
    if numeric_columns:
        analysis += f"**Numeric Analysis:**\n"
        for col in numeric_columns:
            analysis += f"• {col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, Range={df[col].min()}-{df[col].max()}\n"
    
    # Categorical analysis
    if text_columns:
        analysis += f"\n**Categorical Analysis:**\n"
        for col in text_columns:
            top_values = df[col].value_counts().head(3)
            analysis += f"• {col}: {df[col].nunique()} categories, Top: {', '.join([f'{k}({v})' for k,v in top_values.items()])}\n"
    
    # Time-based analysis if date columns exist
    if date_columns:
        analysis += f"\n**Time-based Analysis:**\n"
        for col in date_columns:
            date_range = f"{df[col].min()} to {df[col].max()}"
            analysis += f"• {col}: Date range {date_range}\n"
    
    return analysis

def handle_summary_request(df, numeric_columns, text_columns, date_columns):
    """Handle summary requests"""
    summary = f"📋 **Data Summary**\n\n"
    summary += f"**Dataset Overview:**\n"
    summary += f"• Total records: {df.shape[0]:,}\n"
    summary += f"• Total columns: {df.shape[1]}\n"
    summary += f"• Missing values: {df.isnull().sum().sum()}\n\n"
    
    summary += f"**Column Breakdown:**\n"
    summary += f"• Numeric columns ({len(numeric_columns)}): {', '.join(numeric_columns) if numeric_columns else 'None'}\n"
    summary += f"• Text columns ({len(text_columns)}): {', '.join(text_columns) if text_columns else 'None'}\n"
    summary += f"• Date columns ({len(date_columns)}): {', '.join(date_columns) if date_columns else 'None'}\n\n"
    
    if numeric_columns:
        summary += f"**Key Statistics:**\n{df[numeric_columns].describe().to_string()}\n"
    
    return summary

def handle_dynamic_date_query(df, query, query_lower, date_columns, numeric_columns):
    """Handle date queries with any date column"""
    import re
    
    # Extract date from query
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', query)
    if not dates:
        return "Could not extract a valid date. Please use format like '2023-01-02'"
    
    found_date = dates[0]
    
    # Use the first date column found
    date_col = date_columns[0]
    
    try:
        # Convert and filter
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col]).dt.strftime('%Y-%m-%d')
        filtered_df = df_copy[df_copy[date_col] == found_date]
        
        if filtered_df.empty:
            return f"No data found for {found_date} in {date_col} column.\nAvailable dates: {', '.join(df_copy[date_col].unique()[:5])}"
        
        result = f"📅 **Data for {found_date}:**\n\n"
        
        if "total" in query_lower and numeric_columns:
            for col in numeric_columns:
                total = filtered_df[col].sum()
                result += f"• Total {col}: {total}\n"
        elif "average" in query_lower and numeric_columns:
            for col in numeric_columns:
                avg = filtered_df[col].mean()
                result += f"• Average {col}: {avg:.2f}\n"
        
        result += f"\n**Detailed records:**\n{filtered_df.to_string(index=False)}"
        return result
        
    except Exception as e:
        return f"Error processing date query: {str(e)}"

def handle_dynamic_aggregation_query(df, query, query_lower, numeric_columns):
    """Handle aggregation queries with any numeric column"""
    result = f"📊 **Aggregation Analysis:**\n\n"
    
    if "total" in query_lower or "sum" in query_lower:
        for col in numeric_columns:
            total = df[col].sum()
            result += f"• Total {col}: {total:,}\n"
    
    if "average" in query_lower or "mean" in query_lower:
        for col in numeric_columns:
            avg = df[col].mean()
            result += f"• Average {col}: {avg:.2f}\n"
    
    if "max" in query_lower:
        for col in numeric_columns:
            maximum = df[col].max()
            result += f"• Maximum {col}: {maximum}\n"
    
    if "min" in query_lower:
        for col in numeric_columns:
            minimum = df[col].min()
            result += f"• Minimum {col}: {minimum}\n"
    
    return result

def handle_generic_query(df, query, numeric_columns, text_columns, date_columns):
    """Handle any other query by providing helpful information"""
    return f"""🤔 **Analysis for:** "{query}"

I can help you analyze this dataset. Here's what's available:

**📊 Dataset Structure:**
• {df.shape[0]:,} rows × {df.shape[1]} columns
• Numeric columns: {', '.join(numeric_columns) if numeric_columns else 'None'}
• Text columns: {', '.join(text_columns) if text_columns else 'None'}
• Date columns: {', '.join(date_columns) if date_columns else 'None'}

**💡 Try asking:**
• "Analyze the data" or "Show me insights"
• "Total [numeric column]" or "Average [numeric column]"
• "Show data for [specific date]" (if date columns exist)
• "Create a summary" or "Describe the data"
• "Plot/graph [column name]"

**🔍 Sample data:**
{df.head(2).to_string(index=False)}
"""

def handle_date_query(df, query, query_lower):
    """Handle date-specific queries with better parsing"""
    if 'Date' not in df.columns:
        return "No 'Date' column found in the dataset."
    
    # Extract date from query
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        r'\d{2}/\d{2}/\d{4}'   # DD/MM/YYYY
    ]
    
    found_date = None
    for pattern in date_patterns:
        match = re.search(pattern, query)
        if match:
            found_date = match.group()
            break
    
    if not found_date:
        return "Could not extract a valid date from your query. Please use format like '2023-01-02'"
    
    try:
        # Convert date column to string for comparison
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.strftime('%Y-%m-%d')
        
        # Normalize the found date to YYYY-MM-DD format
        if '/' in found_date:
            found_date = found_date.replace('/', '-')
        
        # Filter for the specific date
        filtered_df = df_copy[df_copy['Date'] == found_date]
        
        if filtered_df.empty:
            return f"No data found for date: {found_date}\n\nAvailable dates: {', '.join(df_copy['Date'].unique())}"
        
        # Determine what analysis to perform
        if "total sales" in query_lower or "sum" in query_lower:
            if 'Sales' in filtered_df.columns:
                total = filtered_df['Sales'].sum()
                return f"Total sales on {found_date}: {total}\n\nDetailed records:\n{filtered_df.to_string(index=False)}"
        elif "average sales" in query_lower or "mean" in query_lower:
            if 'Sales' in filtered_df.columns:
                avg = filtered_df['Sales'].mean()
                return f"Average sales on {found_date}: {avg:.2f}\n\nDetailed records:\n{filtered_df.to_string(index=False)}"
        else:
            return f"Data for {found_date}:\n{filtered_df.to_string(index=False)}"
            
    except Exception as e:
        return f"Error processing date query: {str(e)}"

def handle_product_query(df, query, query_lower):
    """Handle product-related queries"""
    if 'Product' not in df.columns:
        return "No 'Product' column found in the dataset."
    
    # Check for specific sales value
    numbers = re.findall(r'\d+', query)
    if numbers and 'Sales' in df.columns:
        value = int(numbers[0])
        result = df[df['Sales'] == value]
        if not result.empty:
            products = result['Product'].tolist()
            return f"Product(s) with sales = {value}: {', '.join(map(str, products))}\n\nFull records:\n{result.to_string(index=False)}"
        else:
            return f"No products found with sales = {value}"
    else:
        # Show all products and sales
        if 'Sales' in df.columns:
            return f"All products and their sales:\n{df[['Product', 'Sales']].to_string(index=False)}"
        else:
            return f"All products:\n{df['Product'].tolist()}"

def handle_region_query(df, query_lower):
    """Handle region-related queries"""
    if 'Region' not in df.columns:
        return "No 'Region' column found in the dataset."
    
    return f"Unique regions: {', '.join(df['Region'].unique())}\n\nRegion distribution:\n{df['Region'].value_counts().to_string()}"

def extract_number_from_query(query):
    """Extract number from query string"""
    numbers = re.findall(r'\d+', query)
    return int(numbers[0]) if numbers else None

def execute_safe_pandas(df, query_lower):
    """Execute safe pandas operations"""
    safe_operations = ['head', 'tail', 'describe', 'info', 'shape', 'columns', 'dtypes', 'mean', 'sum', 'count', 'max', 'min']
    if any(op in query_lower for op in safe_operations):
        try:
            result = eval(query_lower)
            return f"Result:\n{str(result)}"
        except Exception as e:
            return f"Error executing pandas code: {str(e)}"
    else:
        return "For security reasons, only basic pandas operations are allowed in direct execution."

def generate_helpful_default_response(df, query):
    """Generate a helpful default response when query doesn't match patterns"""
    return f"""I couldn't understand your specific query: "{query}"

Here's what I can help you with:

**Basic Data Operations:**
- "show first 5 rows" or "head of data"
- "describe data" or "data summary"
- "show columns" or "data shape"

**Sales Analysis:**
- "total sales" or "sum of sales"
- "average sales" or "mean sales"

**Date-Specific Queries:**
- "total sales on 2023-01-02"
- "show data for 2023-01-01"

**Product Analysis:**
- "product with sales 100"
- "show all products and sales"

**Current dataset overview:**
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}

Sample data:
{df.head(2).to_string()}"""

# Main agent execution function
def run_agent(query):
    """Main function to run the agent with improved error handling"""
    
    # Check if the sales data file exists
    if not os.path.exists("data/sales_data.csv"):
        st.error("sales_data.csv not found in /data folder!")
        return "Error: Data file not found. Please ensure 'sales_data.csv' exists in the 'data' folder."
            
    # Load the sales data
    try:
        df = pd.read_csv("data/sales_data.csv")
        st.info(f"✅ Loaded data successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            st.write("**Columns:**", list(df.columns))
            st.write("**First few rows:**")
            st.dataframe(df.head())
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return "Error loading data. Please check your CSV file format."
        
    # Determine query type
    sales_keywords = ["sales", "revenue", "profit", "data", "df", "dataframe", "analyze", "total", "average", "sum", "mean"]
    is_sales_query = any(keyword in query.lower() for keyword in sales_keywords)
        
    # Load the ChromaDB knowledge base
    vector_store = None
    if os.path.exists("./chroma_db"):
        try:
            embeddings = HuggingFaceEmbeddings()
            vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            st.info("📚 Knowledge base loaded")
        except Exception as e:
            st.warning(f"⚠️ Error loading knowledge base: {str(e)}")
            
    if is_sales_query:
        st.info("🔍 Routing to sales data analysis...")
        
        # Enable agent for intelligent analysis with Ollama LLM
        query_lower = query.lower()
        
        # Use agent for most queries, with smart fallback for when it fails
        use_agent = True  # Enable agent to use Ollama LLM
        
        if use_agent:
            st.info("🤖 Using Ollama LLaMA 2 agent for intelligent analysis...")
            
            # Create and execute agent
            pandas_agent = create_robust_agent_with_retries(df)
            
            if pandas_agent:
                processed_query = preprocess_query_for_agent(query, df)
                agent_result = execute_agent_with_fallback(pandas_agent, processed_query, df)
                
                if agent_result:
                    return agent_result
            
            # If agent fails, use enhanced fallback as backup
            st.info("🔄 Agent encountered issues, using enhanced fallback analysis...")
            
        else:
            st.info("⚡ Using direct analysis...")
            
        return enhanced_fallback_analysis(df, query)
        
    else:
        st.info("🔍 Routing to knowledge base search...")
        if vector_store:
            try:
                docs = vector_store.similarity_search(query, k=3)
                if docs:
                    response = "📖 **Based on the knowledge base:**\n\n"
                    for i, doc in enumerate(docs, 1):
                        response += f"**{i}.** {doc.page_content}\n\n"
                    return response
                else:
                    return "❓ I couldn't find any relevant information in the knowledge base for your query."
            except Exception as e:
                return f"❌ Error searching knowledge base: {str(e)}"
        else:
            return "📝 No knowledge base available. Please upload a document in the Knowledge Base page to create one."

# Streamlit UI
st.set_page_config(page_title="AI Agents", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Analysis Agents")
st.markdown("*Interact with intelligent AI agents to get insights from your data and knowledge base.*")

# Add example queries in a more organized way
with st.expander("💡 Example Queries"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Sales Data Analysis:**
        - `Show me the first 5 rows of data`
        - `What is the total sales?`
        - `What is the average sales?`
        - `Total sales on 2023-01-02`
        - `Product with sales 100`
        - `Show all regions`
        """)
    
    with col2:
        st.markdown("""
        **🔍 Data Exploration:**
        - `Describe the data`
        - `Show data summary` 
        - `What columns do we have?`
        - `Show sales data for date 2023-01-01`
        - `Find products and their sales`
        - `Any questions about uploaded documents`
        """)

# Main input
query = st.text_input(
    "🗣️ **Ask the AI agent a question:**", 
    placeholder="e.g., 'What is the total sales on 2023-01-02?' or 'Show me the product data'",
    help="Type your question about the data or knowledge base"
)

# Execute query
if query:
    with st.spinner("🧠 AI agent is analyzing your request..."):
        try:
            response = run_agent(query)
            
            # Display response with better formatting
            st.markdown("---")
            st.markdown("### 📝 Response:")
            
            if isinstance(response, str) and len(response) > 800:
                st.text_area("Detailed Response:", response, height=300)
            else:
                st.markdown(response)
                
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.info("💡 Try a simpler query or check your data file format.")

# Enhanced sidebar with system status
with st.sidebar:
    st.markdown("## 🔧 System Status")
    
    # Check data file
    if os.path.exists("data/sales_data.csv"):
        st.success("✅ Sales data loaded")
        try:
            df = pd.read_csv("data/sales_data.csv")
            st.markdown(f"**Rows:** {df.shape[0]}")
            st.markdown(f"**Columns:** {df.shape[1]}")
            st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")
        except:
            st.warning("⚠️ Data file exists but couldn't read it")
    else:
        st.error("❌ Sales data not found")
        st.info("Place your `sales_data.csv` file in the `data/` folder")
    
    # Check knowledge base
    if os.path.exists("./chroma_db"):
        st.success("✅ Knowledge base available")
    else:
        st.info("ℹ️ No knowledge base found")
        st.caption("Upload documents in Knowledge Base page")
        
    # LLM status
    st.success("✅ Ollama LLM configured")
    st.caption("Model: llama2:latest")
    
    # Add helpful tips
    st.markdown("---")
    st.markdown("## 💡 Tips")
    st.markdown("""
    - Be specific in your queries
    - Use exact column names when possible
    - Try different phrasings if one doesn't work
    - Check data preview for column names
    """)