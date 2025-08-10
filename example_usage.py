#!/usr/bin/env python3
# Example usage of the consolidated Excel Query Engine
# Two main functions:
# - process_file(file_path) -> data_representation
# - excel_query(query, data_representation) -> answer

from main import process_file, excel_query

def read_queries_from_file(filename: str) -> list:
    """Read queries from the specified file."""
    try:
        with open(filename, 'r') as file:
            queries = []
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    # Remove the numbering (e.g., "1. ", "2. ") if present
                    if line[0].isdigit() and '. ' in line:
                        query = line.split('. ', 1)[1]
                    else:
                        query = line
                    queries.append(query)
            return queries
    except FileNotFoundError:
        print(f"Query file not found: {filename}")
        return []
    except Exception as e:
        print(f"Error reading query file: {e}")
        return []

def main():
    print("Excel Query Engine - Comprehensive Examples")
    print("=" * 50)
    
    # Process the Excel file (will use cached data if available)
    excel_file = "Consolidated Plan 2023-2024.xlsm"
    representation = process_file(excel_file)
    
    if not representation:
        print("Failed to process Excel file")
        return
    
    # Read all queries from the file
    queries = read_queries_from_file("excel_queries.txt")
    
    if not queries:
        print("No queries found in excel_queries.txt")
        return
    
    print(f"\nRunning all {len(queries)} queries from excel_queries.txt...")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}/{len(queries)} ---")
        print(f"Query: {query}")
        print("-" * 40)
        answer = excel_query(query, representation)
        print(answer)
        print("=" * 50)
    
    print("\nAll queries completed! Use process_file() and excel_query() functions directly.")

if __name__ == "__main__":
    main()
