#!/usr/bin/env python3
"""
Hierarchical Graph Builder for Excel Data
Creates a tree structure: Workbook → Sheets → Tables → Columns → Rows → Cells
with context at each level for effective querying.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from .knowledge_graph import KnowledgeGraphBuilder
import faiss
import numpy as np
import re
import pandas as pd
import json
import os

class HierarchicalGraphBuilder:
    """Builds a hierarchical knowledge graph with proper tree structure and context."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the hierarchical graph builder."""
        self.graph = nx.DiGraph()  # Directed graph for hierarchy
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.node_embeddings = {}
        self.embedding_index = None
        self.workbook_name = None
        
    def build_hierarchical_graph(self, file_rep) -> 'HierarchicalGraphBuilder':
        """Build the complete hierarchical graph from Excel data representation."""
        print("Building hierarchical knowledge graph...")
        
        # Step 1: Create root workbook node
        self.workbook_name = file_rep.source_info.get('file_path', 'Unknown Workbook').split('/')[-1]
        self._create_workbook_node()
        
        # Step 2: Create sheet nodes and connect to workbook
        sheet_nodes = self._create_sheet_nodes(file_rep)
        
        # Step 3: For each sheet, create table/region nodes
        for sheet_name, sheet_data in file_rep.sheets_data.items():
            table_nodes = self._create_table_nodes(sheet_name, sheet_data)
            
            # Step 4: For each table, create column and row nodes
            for table_id in table_nodes:
                column_nodes = self._create_column_nodes(table_id, sheet_data)
                row_nodes = self._create_row_nodes(table_id, sheet_data)
                
                # Step 5: Create cell nodes and connect to columns/rows
                self._create_cell_nodes(table_id, sheet_data, file_rep)
        
        # Step 6: Add contextual relationships
        self._add_contextual_relationships(file_rep)
        
        # Step 7: Build embeddings and search index
        self._build_embeddings_and_index()
        
        print(f"Hierarchical graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self
        
    def _create_workbook_node(self):
        """Create the root workbook node."""
        workbook_id = f"workbook_{self.workbook_name}"
        workbook_data = {
            'type': 'workbook',
            'name': self.workbook_name,
            'context': {
                'total_sheets': 0,  # Will be updated
                'file_type': 'excel',
                'description': f"Excel workbook containing financial data"
                # make this more detailed using an LLM?
            }
        }
        
        self.graph.add_node(workbook_id, **workbook_data)
        print(f"Created workbook node: {workbook_id}")
        
    def _create_sheet_nodes(self, file_rep) -> List[str]:
        """Create sheet nodes and connect to workbook."""
        sheet_nodes = []
        workbook_id = f"workbook_{self.workbook_name}"
        
        for sheet_name in file_rep.source_info.get('sheets', []):
            sheet_id = f"sheet_{sheet_name}"
            
            # Get sheet data
            sheet_data = file_rep.sheets_data.get(sheet_name)
            if sheet_data is not None:
                total_rows = len(sheet_data)
                total_cols = len(sheet_data.columns) if hasattr(sheet_data, 'columns') else 0
            else:
                total_rows = 0
                total_cols = 0
            
            # Determine sheet type based on name
            sheet_type = self._classify_sheet_type(sheet_name)
            
            sheet_data = {
                'type': 'sheet',
                'name': sheet_name,
                'context': {
                    'total_rows': total_rows,
                    'total_columns': total_cols,
                    'sheet_type': sheet_type,
                    'description': f"{sheet_type} sheet with {total_rows} rows and {total_cols} columns"
                }
            }
            
            self.graph.add_node(sheet_id, **sheet_data)
            self.graph.add_edge(workbook_id, sheet_id, edge_type='contains')
            sheet_nodes.append(sheet_id)
            
        # Update workbook context
        self.graph.nodes[workbook_id]['context']['total_sheets'] = len(sheet_nodes)
        
        print(f"Created {len(sheet_nodes)} sheet nodes")
        return sheet_nodes
        
    def _classify_sheet_type(self, sheet_name: str) -> str:
        """Classify sheet type based on name."""
        sheet_lower = sheet_name.lower()
        
        if 'p&l' in sheet_lower or 'profit' in sheet_lower:
            return 'profit_loss'
        elif 'bs' in sheet_lower or 'balance' in sheet_lower:
            return 'balance_sheet'
        elif 'debt' in sheet_lower:
            return 'debt_schedule'
        elif 'financial' in sheet_lower:
            return 'financial_statement'
        elif 'summary' in sheet_lower:
            return 'summary'
        elif 'operations' in sheet_lower:
            return 'operations'
        elif 'sales' in sheet_lower or 'marketing' in sheet_lower:
            return 'sales_marketing'
        elif 'admin' in sheet_lower or 'finance' in sheet_lower:
            return 'admin_finance'
        else:
            return 'general'
            
    def _create_table_nodes(self, sheet_name: str, sheet_data) -> List[str]:
        """Create table/region nodes within a sheet."""
        table_nodes = []
        sheet_id = f"sheet_{sheet_name}"
        
        # For now, create one main table per sheet
        # In the future, this could detect multiple tables
        table_id = f"table_{sheet_name}_main"
        
        # Analyze table structure
        if hasattr(sheet_data, 'columns') and len(sheet_data.columns) > 0:
            # Detect header row
            header_row = 0
            for idx, row in sheet_data.iterrows():
                if idx == 0:  # First row is usually header
                    header_row = idx
                    break
            
            # Detect data regions
            data_start_row = header_row + 1
            data_end_row = len(sheet_data) - 1
            
            table_data = {
                'type': 'table',
                'name': f"Main Table in {sheet_name}",
                'context': {
                    'header_row': header_row,
                    'data_start_row': data_start_row,
                    'data_end_row': data_end_row,
                    'total_columns': len(sheet_data.columns),
                    'description': f"Main data table in {sheet_name} with {len(sheet_data.columns)} columns"
                }
            }
            
            self.graph.add_node(table_id, **table_data)
            self.graph.add_edge(sheet_id, table_id, edge_type='contains')
            table_nodes.append(table_id)
        
        print(f"Created {len(table_nodes)} table nodes for {sheet_name}")
        return table_nodes
        
    def _create_column_nodes(self, table_id: str, sheet_data) -> List[str]:
        """Create column nodes within a table."""
        column_nodes = []
        
        if not hasattr(sheet_data, 'columns'):
            return column_nodes
            
        for col_idx, col_name in enumerate(sheet_data.columns):
            column_id = f"column_{table_id}_{col_idx}"
            
            # Analyze column data
            col_data = sheet_data[col_name].dropna()
            numeric_count = sum(1 for val in col_data if isinstance(val, (int, float)))
            text_count = len(col_data) - numeric_count
            
            # Determine column type
            if numeric_count > text_count:
                col_type = 'numeric'
                if numeric_count > 0:
                    # Filter numeric values
                    numeric_values = [val for val in col_data if isinstance(val, (int, float)) and pd.notna(val)]
                    if numeric_values:
                        col_stats = {
                            'min': float(min(numeric_values)),
                            'max': float(max(numeric_values)),
                            'sum': float(sum(numeric_values))
                        }
                    else:
                        col_stats = {'min': 0, 'max': 0, 'sum': 0}
                else:
                    col_stats = {'min': 0, 'max': 0, 'sum': 0}
            else:
                col_type = 'text'
                col_stats = {}
            
            # Detect if column contains time periods
            time_period = None
            if isinstance(col_name, str):
                for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                    if month in col_name:
                        time_period = month
                        break
                        
            column_data = {
                'type': 'column',
                'name': str(col_name),
                'context': {
                    'column_index': col_idx,
                    'column_type': col_type,
                    'numeric_count': numeric_count,
                    'text_count': text_count,
                    'time_period': time_period,
                    'statistics': col_stats,
                    'description': f"{col_type} column '{col_name}' with {numeric_count} numeric and {text_count} text values"
                }
            }
            
            self.graph.add_node(column_id, **column_data)
            self.graph.add_edge(table_id, column_id, edge_type='contains')
            column_nodes.append(column_id)
            
        print(f"Created {len(column_nodes)} column nodes for {table_id}")
        return column_nodes
        
    def _create_row_nodes(self, table_id: str, sheet_data) -> List[str]:
        """Create row nodes within a table."""
        row_nodes = []
        
        # Create row nodes for data rows (skip header)
        for row_idx in range(1, len(sheet_data)):
            row_id = f"row_{table_id}_{row_idx}"
            
            # Analyze row data
            row_data = sheet_data.iloc[row_idx]
            numeric_count = sum(1 for val in row_data if isinstance(val, (int, float)) and val != 0)
            total_value = sum(val for val in row_data if isinstance(val, (int, float)))
            
            # Detect row type based on content
            row_type = 'data'
            row_label = str(row_data.iloc[0]) if len(row_data) > 0 else f"Row {row_idx}"
            
            # Check if this is a company row
            companies = ['MXD', 'HEC', 'Branch']
            for company in companies:
                if company.lower() in str(row_label).lower():
                    row_type = 'company'
                    break
                    
            row_data_dict = {
                'type': 'row',
                'name': row_label,
                'context': {
                    'row_index': row_idx,
                    'row_type': row_type,
                    'numeric_count': numeric_count,
                    'total_value': total_value,
                    'description': f"{row_type} row '{row_label}' with {numeric_count} numeric values totaling ${total_value:,.2f}"
                }
            }
            
            self.graph.add_node(row_id, **row_data_dict)
            self.graph.add_edge(table_id, row_id, edge_type='contains')
            row_nodes.append(row_id)
            
        print(f"Created {len(row_nodes)} row nodes for {table_id}")
        return row_nodes
        
    def _create_cell_nodes(self, table_id: str, sheet_data, file_rep):
        """Create cell nodes and connect to columns and rows."""
        cell_count = 0
        
        for row_idx in range(len(sheet_data)):
            row_id = f"row_{table_id}_{row_idx}"
            
            for col_idx, col_name in enumerate(sheet_data.columns):
                cell_value = sheet_data.iloc[row_idx, col_idx]
                
                # Only create nodes for non-empty cells
                if pd.notna(cell_value) and cell_value != '':
                    cell_id = f"cell_{table_id}_{row_idx}_{col_idx}"
                    
                    # Determine cell type
                    if isinstance(cell_value, (int, float)):
                        cell_type = 'numeric'
                        cell_stats = {'value': float(cell_value)}
                    else:
                        cell_type = 'text'
                        cell_stats = {'text': str(cell_value)}
                    
                    # Get context from atomic units
                    cell_context = self._get_cell_context(file_rep, table_id, row_idx, col_idx, cell_value)
                    
                    cell_data = {
                        'type': 'cell',
                        'name': f"Cell at R{row_idx}C{col_idx}",
                        'content': cell_value,
                        'context': {
                            'row_index': row_idx,
                            'column_index': col_idx,
                            'column_name': str(col_name),
                            'cell_type': cell_type,
                            'cell_stats': cell_stats,
                            **cell_context,
                            'description': f"{cell_type} cell containing {cell_value}"
                        }
                    }
                    
                    self.graph.add_node(cell_id, **cell_data)
                    
                    # Connect to column and row
                    column_id = f"column_{table_id}_{col_idx}"
                    if self.graph.has_node(column_id):
                        self.graph.add_edge(column_id, cell_id, edge_type='contains')
                    if self.graph.has_node(row_id):
                        self.graph.add_edge(row_id, cell_id, edge_type='contains')
                        
                    cell_count += 1
                    
        print(f"Created {cell_count} cell nodes for {table_id}")
        
    def _get_cell_context(self, file_rep, table_id: str, row_idx: int, col_idx: int, cell_value) -> Dict:
        """Get contextual information for a cell from atomic units."""
        context = {}
        
        # Find matching atomic unit
        for unit in file_rep.get_all_atomic_units():
            if (unit.unit_type == 'cell' and 
                unit.location.get('row_index') == row_idx and 
                unit.location.get('column_index') == col_idx):
                
                # Copy context from atomic unit
                context.update(unit.context)
                break
                
        return context
        
    def _add_contextual_relationships(self, file_rep):
        """Add contextual relationships between nodes."""
        print("Adding contextual relationships...")
        
        # Collect all edges to add first, then add them
        edges_to_add = []
        
        # Add company relationships
        companies = ['MXD', 'HEC', 'Branch']
        for company in companies:
            company_id = f"company_{company}"
            
            # Find all nodes related to this company
            for node_id, node_data in list(self.graph.nodes(data=True)):
                if node_data.get('type') == 'cell':
                    context = node_data.get('context', {})
                    if context.get('company') == company:
                        edges_to_add.append((company_id, node_id, 'related_to'))
                        
        # Add metric relationships
        metrics = ['revenue', 'profit', 'cost', 'expense', 'debt', 'ebitda', 'fcf']
        for metric in metrics:
            metric_id = f"metric_{metric}"
            
            # Find all nodes related to this metric
            for node_id, node_data in list(self.graph.nodes(data=True)):
                if node_data.get('type') == 'cell':
                    context = node_data.get('context', {})
                    if context.get('metric') == metric:
                        edges_to_add.append((metric_id, node_id, 'related_to'))
                        
        # Add time period relationships
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in months:
            month_id = f"time_{month}"
            
            # Find all nodes related to this time period
            for node_id, node_data in list(self.graph.nodes(data=True)):
                if node_data.get('type') == 'cell':
                    context = node_data.get('context', {})
                    if context.get('time_period') == month:
                        edges_to_add.append((month_id, node_id, 'related_to'))
        
        # Add all edges at once
        for source, target, edge_type in edges_to_add:
            self.graph.add_edge(source, target, edge_type=edge_type)
                        
        print(f"Added {len(edges_to_add)} contextual relationships")
        
    def _build_embeddings_and_index(self):
        """Build embeddings for all nodes and create search index."""
        print("Building embeddings and search index...")
        
        # Create embeddings for all nodes
        for node_id, node_data in self.graph.nodes(data=True):
            description = self._create_node_description(node_data)
            embedding = self.embedding_model.encode(description)
            self.node_embeddings[node_id] = embedding
            
        # Build FAISS index
        if self.node_embeddings:
            embeddings_array = np.array(list(self.node_embeddings.values()))
            self.embedding_index = faiss.IndexFlatL2(embeddings_array.shape[1])
            self.embedding_index.add(embeddings_array.astype('float32'))
            
        print(f"Built embeddings for {len(self.node_embeddings)} nodes")
        
    def _create_node_description(self, node_data: Dict) -> str:
        """Create a description for a node for embedding."""
        node_type = node_data.get('type', 'unknown')
        name = node_data.get('name', 'unknown')
        context = node_data.get('context', {})
        content = node_data.get('content', '')
        
        desc_parts = [
            f"Type: {node_type}",
            f"Name: {name}"
        ]
        
        if content:
            desc_parts.append(f"Content: {content}")
            
        if context:
            context_str = ", ".join([f"{k}: {v}" for k, v in context.items() if k != 'description'])
            desc_parts.append(f"Context: {context_str}")
            
        return " | ".join(desc_parts)
        
    def find_similar_nodes(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar nodes using embeddings."""
        if not self.embedding_index or not self.node_embeddings:
            return []
            
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Search
        distances, indices = self.embedding_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(top_k, len(self.node_embeddings))
        )
        
        # Convert to node IDs and similarities
        node_ids = list(self.node_embeddings.keys())
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(node_ids):
                node_id = node_ids[idx]
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                results.append((node_id, similarity))
                
        return results
        
    def get_node_neighbors(self, node_id: str, max_depth: int = 2) -> List[str]:
        """Get neighboring nodes up to a specified depth."""
        neighbors = []
        visited = set()
        
        def dfs(current_id, depth):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)
            
            # Get neighbors
            for neighbor in self.graph.neighbors(current_id):
                if neighbor not in visited:
                    neighbors.append(neighbor)
                    dfs(neighbor, depth + 1)
                    
        dfs(node_id, 0)
        return neighbors
        
    def get_hierarchical_path(self, node_id: str) -> List[str]:
        """Get the hierarchical path from root to a node."""
        path = []
        current = node_id
        
        while current:
            path.append(current)
            # Find parent
            parents = list(self.graph.predecessors(current))
            current = parents[0] if parents else None
            
        return list(reversed(path))
        
    def get_graph_summary(self) -> Dict:
        """Get summary statistics of the graph."""
        node_types = {}
        edge_types = {}
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        for source, target, edge_data in self.graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'has_embeddings': len(self.node_embeddings) > 0,
            'has_index': self.embedding_index is not None
        }
    
    def save_representation(self, file_path: str) -> bool:
        """
        Save the hierarchical graph representation to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data for serialization
            graph_data = {
                'workbook_name': self.workbook_name,
                'nodes': {},
                'edges': [],
                'node_embeddings': {},
                'metadata': {
                    'total_nodes': self.graph.number_of_nodes(),
                    'total_edges': self.graph.number_of_edges(),
                    'has_embeddings': len(self.node_embeddings) > 0,
                    'has_index': self.embedding_index is not None
                }
            }
            
            # Convert nodes to serializable format
            for node_id, node_data in self.graph.nodes(data=True):
                # Convert numpy arrays and other non-serializable types
                serializable_node_data = {}
                for key, value in node_data.items():
                    if isinstance(value, np.ndarray):
                        serializable_node_data[key] = value.tolist()
                    elif isinstance(value, pd.DataFrame):
                        serializable_node_data[key] = value.to_dict('records')
                    elif isinstance(value, pd.Series):
                        serializable_node_data[key] = value.to_dict()
                    else:
                        serializable_node_data[key] = value
                
                graph_data['nodes'][node_id] = serializable_node_data
            
            # Convert edges to serializable format
            for source, target, edge_data in self.graph.edges(data=True):
                serializable_edge_data = {}
                for key, value in edge_data.items():
                    if isinstance(value, np.ndarray):
                        serializable_edge_data[key] = value.tolist()
                    else:
                        serializable_edge_data[key] = value
                
                graph_data['edges'].append({
                    'source': source,
                    'target': target,
                    'data': serializable_edge_data
                })
            
            # Convert embeddings to serializable format
            for node_id, embedding in self.node_embeddings.items():
                if isinstance(embedding, np.ndarray):
                    graph_data['node_embeddings'][node_id] = embedding.tolist()
                else:
                    graph_data['node_embeddings'][node_id] = embedding
            
            # Save to JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"Hierarchical graph representation saved to: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving hierarchical graph representation: {e}")
            return False
    
    def load_representation(self, file_path: str) -> bool:
        """
        Load the hierarchical graph representation from a JSON file.
        
        Args:
            file_path: Path to the JSON file to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False
            
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Clear existing graph
            self.graph.clear()
            self.node_embeddings.clear()
            self.embedding_index = None
            
            # Restore workbook name
            self.workbook_name = graph_data.get('workbook_name', 'Unknown Workbook')
            
            # Restore nodes
            for node_id, node_data in graph_data.get('nodes', {}).items():
                # Convert lists back to numpy arrays where appropriate
                restored_node_data = {}
                for key, value in node_data.items():
                    if key == 'embeddings' and isinstance(value, list):
                        restored_node_data[key] = np.array(value)
                    elif key == 'data' and isinstance(value, list):
                        # Try to convert back to DataFrame if it was originally a DataFrame
                        try:
                            restored_node_data[key] = pd.DataFrame(value)
                        except:
                            restored_node_data[key] = value
                    else:
                        restored_node_data[key] = value
                
                self.graph.add_node(node_id, **restored_node_data)
            
            # Restore edges
            for edge_info in graph_data.get('edges', []):
                source = edge_info['source']
                target = edge_info['target']
                edge_data = edge_info.get('data', {})
                
                # Convert lists back to numpy arrays where appropriate
                restored_edge_data = {}
                for key, value in edge_data.items():
                    if isinstance(value, list) and key in ['weights', 'similarities']:
                        restored_edge_data[key] = np.array(value)
                    else:
                        restored_edge_data[key] = value
                
                self.graph.add_edge(source, target, **restored_edge_data)
            
            # Restore embeddings
            for node_id, embedding in graph_data.get('node_embeddings', {}).items():
                if isinstance(embedding, list):
                    self.node_embeddings[node_id] = np.array(embedding)
                else:
                    self.node_embeddings[node_id] = embedding
            
            # Rebuild embedding index if embeddings exist
            if self.node_embeddings:
                self._rebuild_embedding_index()
            
            print(f"Hierarchical graph representation loaded from: {file_path}")
            print(f"Loaded {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            print(f"Error loading hierarchical graph representation: {e}")
            return False
    
    def _rebuild_embedding_index(self):
        """Rebuild the FAISS index from loaded embeddings."""
        if not self.node_embeddings:
            return
        
        try:
            # Get embedding dimension from first embedding
            first_embedding = next(iter(self.node_embeddings.values()))
            if isinstance(first_embedding, np.ndarray):
                dimension = first_embedding.shape[0]
            else:
                dimension = len(first_embedding)
            
            # Create new index
            self.embedding_index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            embeddings_list = []
            for node_id, embedding in self.node_embeddings.items():
                if isinstance(embedding, np.ndarray):
                    embeddings_list.append(embedding.astype('float32'))
                else:
                    embeddings_list.append(np.array(embedding, dtype='float32'))
            
            if embeddings_list:
                embeddings_array = np.vstack(embeddings_list)
                self.embedding_index.add(embeddings_array)
                
            print(f"Rebuilt embedding index with {len(self.node_embeddings)} embeddings")
            
        except Exception as e:
            print(f"Error rebuilding embedding index: {e}")
            self.embedding_index = None
