# Consolidated Excel Query Engine

An Excel data processing and querying engine that combines traditional data extraction with AI-powered analysis.

## 🚀 Core Functions

### 1. `process_file(file_path: str) -> Dict[str, Any]`
Processes Excel files and returns a structured data representation.

**Parameters:**
- `file_path`: Path to the Excel file to process

**Returns:**
- Structured data representation of the Excel file

**Example:**
```python
from main import process_file

# Process an Excel file
representation = process_file("data/Consolidated Plan 2023-2024.xlsm")
print(f"Processed {len(representation)} sheets")
```

### 2. `excel_query(query: str, file_rep: Dict[str, Any]) -> str`
Queries the data representation and returns comprehensive answers with AI analysis.

**Parameters:**
- `query`: The query string to execute
- `file_rep`: The data representation from `process_file()`

**Returns:**
- Comprehensive answer to the query with AI insights

**Example:**
```python
from main import excel_query

# Query the data
answer = excel_query("What is MXD's Gross Profit in Jan 2022?", representation)
print(answer)
```

## 🏗️ Architecture

The engine uses a **Combined Query Engine** that:
1. **First attempts traditional query parsing** using pattern matching and semantic search
2. **Falls back to AI-powered analysis** when traditional methods have low confidence
3. **Always provides AI enhancement** to analyze results and provide business insights

## 🔧 Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Variables
Set your OpenAI API key for full AI functionality:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
OPENAI_API_KEY=your-api-key-here
```

## 📁 Project Structure

```
sapien-task/
├── main.py                          # Main entry point with core functions
├── src/                             # Source code package
│   ├── __init__.py                 # Package initialization
│   ├── core/                       # Core functionality
│   │   ├── __init__.py            # Core package init
│   │   ├── excel_processor.py     # Excel file processing engine
│   │   ├── query.py               # Traditional query engine
│   │   └── combined_query_engine.py # Combined traditional + AI query engine
│   ├── llm/                       # LLM integration
│   │   ├── __init__.py            # LLM package init
│   │   └── llm_integration.py     # AI/LLM integration components
│   ├── graph/                     # Graph-based functionality
│   │   ├── __init__.py            # Graph package init
│   │   ├── hierarchical_graph.py  # Hierarchical graph builder
│   │   ├── hierarchical_extraction.py # Hierarchical data extraction
│   │   └── knowledge_graph.py     # Knowledge graph functionality
│   └── utils/                     # Utilities
│       ├── __init__.py            # Utils package init
│       └── config.py              # Configuration settings
├── examples/                       # Example scripts
│   ├── __init__.py                # Examples package init
│   └── example_usage.py           # Usage examples
├── data/                          # Data files
│   ├── excel_queries.txt          # Sample queries
│   └── Consolidated Plan 2023-2024.xlsm # Sample Excel file
├── docs/                          # Documentation
│   ├── README.md                  # This file
│   ├── SECURITY.md                # Security documentation
│   └── setup_env.sh               # Environment setup script
├── cache/                         # Cache and generated files
├── requirements.txt                # Python dependencies
└── .env                           # Environment variables (create this)
```

## 🎯 Usage Examples

### Basic Usage
```python
from main import process_file, excel_query

# 1. Process your Excel file
representation = process_file("data/your_file.xlsx")

# 2. Query the data
answer = excel_query("What is the revenue for Q1?", representation)
print(answer)
```

### Advanced Usage
```python
from main import process_file, excel_query, save_representation, load_representation

# Process and save for later use
representation = process_file("large_file.xlsx")
save_representation(representation, "my_data.json")

# Load and query later
representation = load_representation("my_data.json")
answer = excel_query("Complex business question?", representation)
```

## 🔍 Query Examples

The engine handles various types of queries:
- **Financial metrics**: "What is MXD's Gross Profit in Jan 2022?"
- **Trend analysis**: "What direction is 2023 EBITDA vs Revenue going?"
- **Comparative analysis**: "How do the companies compare in Q4?"
- **Forecast analysis**: "What's wrong with Branch's forecasts?"
- **Business insights**: "Describe the trajectory of all companies over 2022-2023"

## 🧠 AI Features

- **Intelligent Routing**: Automatically chooses between traditional and AI methods
- **Confidence Scoring**: Provides confidence levels for each query result
- **Business Insights**: AI analyzes data and provides actionable business intelligence
- **Cost Controls**: Built-in API usage management and rate limiting
- **Fallback Handling**: Gracefully handles complex queries that traditional methods can't process

## 📊 Output Format

Each query returns structured information including:
- Query method used (traditional vs AI)
- Confidence score
- Source data sheets
- Key data points found
- AI analysis and insights

## 🚨 Troubleshooting

### API Key Issues
```bash
# Check if API key is loaded
echo $OPENAI_API_KEY

# Load from .env file
source .env
```

### Performance
- Large Excel files may take time to process initially
- Data representation is cached in JSON for faster subsequent queries
- AI features require internet connection and valid API key
