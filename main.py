import pandas as pd
import openpyxl
import sys
import os
sys.path.append('..')
from excel_processor1 import ExcelProcessor as ep


def process_file(file_path: str) -> any:
    """
    Processes the file and returns a representation of the file
    file_path: The path to the file to be processed
    Returns: A representation of the file
    """
    representation_file = "data_representation.json"
    
    # Step 1: Process the Excel file
    print("\nStep 1: Processing Excel file...")
    print("-" * 30)
    
    # Create an instance of the Excel processor
    processor = ep()
    
    # Check if we already have a processed representation
    if os.path.exists(representation_file):
        print(f"Found existing representation file: {representation_file}")
        use_existing = input("Use existing representation? (y/n): ").lower().strip()
        
        if use_existing == 'y':
            representation = ep.load_representation(representation_file)
            if representation is None:
                print("Failed to load existing representation. Processing file again...")
                representation = processor.process_file(file_path)
        else:
            representation = processor.process_file(file_path)
            ep.save_representation(representation, representation_file)
    else:
        representation = processor.process_file(file_path)
        ep.save_representation(representation, representation_file)

    return representation
    

def excel_query(query: str, file_rep: any) -> str:
    """
    query: The string query to be executed on the file
    file_rep: The processed filed represented (from process_file)
    """
    # For now, use the processor's query_file method
    processor = ep()
    return processor.query_file(file_rep, query)

def main():
    """
    Main function to run the program
    """
    # Process the Excel file
    file_rep = process_file("/Users/sanjanad/sapien-take-home-final/Consolidated Plan 2023-2024.xlsm")
    
    # Check if knowledge graph was built
    if hasattr(file_rep, 'knowledge_graph') and file_rep.knowledge_graph is not None:
        kg = file_rep.knowledge_graph
        summary = kg.get_graph_summary()
        print(f"\nKnowledge Graph Summary:")
        print(f"  - Total nodes: {summary['total_nodes']}")
        print(f"  - Total edges: {summary['total_edges']}")
        print(f"  - Has embeddings: {summary['has_embeddings']}")
        print(f"  - Has search index: {summary['has_index']}")
    
    # Test with real-world financial queries from excel_queries.txt
    test_queries = [
        "What is MXD's Gross Profit in Jan 2022?",
        "What is MXD's Shipping Income in Oct 2022?",
        "What is MXD's cost of direct labor for each month in 2022?",
        "What percent of MXD's costs are indirect? Which month had the highest percentage?",
        "What percent of HEC's operating expenses are from insurance in total for 2021?",
        "What is Branch's advertising forecasts for each month in 2024?",
        "What direction is 2023 EBITDA vs Revenue vs FCF going for All companies per month?",
        "Explain the debt schedules of each company. Where do they differ?",
        "What's wrong with Branch's forecasts?",
        "Describe the trajectory of all the companies over 2022 and 2023, explain why 2023 Q4 budget is the way it is, and determine whether or not we will hit 2024 forecasts using your own predictions."
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        result = excel_query(query, file_rep)
        print(f"Result: {result}")
    
    # Test knowledge graph similarity search
    if hasattr(file_rep, 'knowledge_graph') and file_rep.knowledge_graph is not None:
        print(f"\n--- Testing Knowledge Graph Similarity Search ---")
        kg = file_rep.knowledge_graph
        
        test_similarity_query = "revenue"
        similar_nodes = kg.find_similar_nodes(test_similarity_query, top_k=3)
        print(f"Similar nodes for '{test_similarity_query}':")
        for node_id, similarity in similar_nodes:
            node_data = kg.graph.nodes[node_id]
            node_type = node_data.get('type', 'Unknown')
            print(f"  - {node_id} ({node_type}, similarity: {similarity:.3f})")

if __name__ == "__main__":

    main()