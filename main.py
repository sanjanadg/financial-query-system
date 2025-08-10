#!/usr/bin/env python3
"""
Consolidated Excel Query Engine
Main deliverable with two core functions:
1. process_file(file_path) - processes Excel files and returns data representation
2. excel_query(query: str, file_rep: any) - queries the data representation and returns answers
"""

import os
import json
from typing import Dict, Any, Optional
from excel_processor import ExcelProcessor
from combined_query_engine import CombinedQueryEngine

def get_api_key() -> Optional[str]:
    """
    Get OpenAI API key using multiple fallback methods.
    
    Returns:
        Optional[str]: The API key if found, None otherwise
    """
    # Method 1: Direct environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key.strip()
    
    # Method 2: Try to load from .env file
    try:
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENAI_API_KEY=') and not line.startswith('#'):
                        api_key = line.split('=', 1)[1].strip()
                        if api_key:
                            # Set it in environment for future use
                            os.environ['OPENAI_API_KEY'] = api_key
                            return api_key
    except Exception:
        pass
    
    # Method 3: Check for common alternative names
    alternative_names = ['OPENAI_KEY', 'AI_API_KEY', 'GPT_API_KEY']
    for alt_name in alternative_names:
        api_key = os.getenv(alt_name)
        if api_key:
            return api_key.strip()
    
    return None

def process_file(file_path: str) -> Dict[str, Any]:
    """
    Process an Excel file and return a structured data representation.
    Intelligently loads existing cached data or processes the file if needed.
    
    Args:
        file_path (str): Path to the Excel file to process
        
    Returns:
        Dict[str, Any]: Structured data representation of the Excel file
        
    Example:
        >>> representation = process_file("Consolidated Plan 2023-2024.xlsm")
        >>> print(f"Processed {len(representation)} sheets")
    """
    try:
        # First, try to load existing cached representation (unless force_reprocess is True)
        if os.path.exists("data_representation.json"):
            print(f"Found existing data representation, loading cached data...")
            cached_representation = load_representation()
            if cached_representation:
                print(f"Loaded cached representation with {len(cached_representation)} sheets")
                return cached_representation
        
        # If no cached data exists or force_reprocess is True, process the file
        print(f"Processing file: {file_path}")
        
        # Initialize the Excel processor
        processor = ExcelProcessor()
        
        # Process the file
        representation = processor.process_file(file_path)
        
        if representation:
            print(f"Successfully processed file with {len(representation)} sheets")
            # Save the representation for future use
            save_representation(representation)
            return representation
        else:
            print("Failed to process file")
            return {}
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return {}

def excel_query(query: str, data_representation: Dict[str, Any]) -> str:
    """
    Execute a query against the data representation and return a formatted answer.
    
    Args:
        query (str): The query to execute
        data_representation (Dict[str, Any]): The data representation from process_file
        
    Returns:
        str: Formatted answer to the query
    """
    try:
        # Get API key for LLM features
        api_key = get_api_key()
        
        # Initialize the combined query engine
        engine = CombinedQueryEngine(api_key=api_key)
        
        print(f"Processing query: {query}")
        
        # Execute the query
        result = engine.query_file(data_representation, query)
        
        if not result:
            return "Query failed - no results returned"
        
        # Extract key information
        query_method = result.get('query_method', 'unknown')
        confidence = result.get('confidence', 0.0)
        source_sheets = result.get('source_sheets', [])
        data_points = result.get('data_points', [])
        
        # Build response
        response_parts = []
        
        # Header
        response_parts.append(f"**Query Results**")
        response_parts.append(f"Method: {query_method.replace('_', ' ').title()}")
        response_parts.append(f"Confidence: {confidence:.1%}")
        
        # Clean up source sheets display
        unique_sheets = list(set(source_sheets)) if source_sheets else []
        sheet_display = ', '.join(unique_sheets[:3]) if len(unique_sheets) > 3 else ', '.join(unique_sheets)
        if len(unique_sheets) > 3:
            sheet_display += f" (+{len(unique_sheets) - 3} more)"
        response_parts.append(f"Source: {sheet_display}")
        response_parts.append(f"Data Points: {len(data_points)}")
        response_parts.append("")
        
        # Data summary
        if data_points:
            response_parts.append("**Key Data:**")
            for i, point in enumerate(data_points[:5], 1):  # Show only first 5
                if 'metric' in point and 'time_period' in point and 'value' in point:
                    metric = point['metric'].split(' - ')[-1] if ' - ' in point['metric'] else point['metric']
                    response_parts.append(f"{i}. {point['time_period']}: ${point['value']:,.2f} ({metric})")
                else:
                    response_parts.append(f"{i}. {point}")
            
            if len(data_points) > 5:
                response_parts.append(f"... and {len(data_points) - 5} more data points")
            response_parts.append("")
        
        # LLM analysis if available
        if result.get('llm_fallback_used') or result.get('query_method') == 'traditional_enhanced' or (api_key and engine.llm_available):
            response_parts.append("**AI Analysis**")
            response_parts.append("-" * 40)
            if 'answer' in result and result['answer'] != result.get('traditional_result', {}).get('answer', ''):
                response_parts.append(result['answer'])
            else:
                response_parts.append("AI analysis provided additional insights and context.")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"Error executing query: {e}"

def save_representation(representation: Dict[str, Any]) -> None:
    """Save the data representation to a JSON file."""
    try:
        with open("data_representation.json", "w") as f:
            json.dump(representation, f, indent=2)
    except Exception as e:
        print(f"Error saving representation: {e}")

def load_representation() -> Optional[Dict[str, Any]]:
    """Load the data representation from a JSON file."""
    try:
        with open("data_representation.json", "r") as f:
            representation = json.load(f)
            print(f"Loaded data representation from: {file_path}")
            return representation
    except Exception as e:
        print(f"Error loading representation: {e}")
        return None

# Main execution - demonstrates the two core functions
if __name__ == "__main__":
    print("Consolidated Excel Query Engine")
    print("=" * 50)
    print("Two main functions:")
    print("• process_file(file_path) -> data_representation")
    print("• excel_query(query, data_representation) -> answer")
    print("=" * 50)
    
    # Process the Excel file directly
    excel_file = "Consolidated Plan 2023-2024.xlsm"
    if os.path.exists(excel_file):
        print(f"\nProcessing Excel file: {excel_file}")
        representation = process_file(excel_file)
        if not representation:
            print("Failed to process Excel file")
            exit(1)
    else:
        print(f"Excel file not found: {excel_file}")
        print("Please ensure the Excel file is in the current directory")
        exit(1)
    
    # Run a few example queries to demonstrate functionality
    example_queries = [
        "What is MXD's Gross Profit in Jan 2022?",
        "What is MXD's Shipping Income in Oct 2022?",
        "What is MXD's cost of direct labor for each month in 2022?"
    ]
    
    print(f"\nRunning {len(example_queries)} example queries...")
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Query {i}/{len(example_queries)} ---")
        answer = excel_query(query, representation)
        print(answer)
        print("-" * 40)
    
    # API key status
    api_key = get_api_key()
    if api_key:
        print(f"\nAPI Key: Active")
    else:
        print("\nAPI Key: Not found - LLM features disabled")
    
    print("\nDemo completed! Use process_file() and excel_query() functions directly.")
