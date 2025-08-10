import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class KnowledgeGraphBuilder:
    """Builds a knowledge graph representation of Excel data with embeddings."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the knowledge graph builder.
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
        self.node_descriptions = {}
        self.embedding_index = None
        self.node_to_index = {}
        self.index_to_node = {}
        
    def create_node_description(self, node_type: str, node_data: Dict) -> str:
        """Create a natural language description of a node for embedding."""
        if node_type == "company":
            company_name = node_data.get('name', 'Unknown')
            return f"Company {company_name} with financial data including revenue, expenses, and performance metrics"
        
        elif node_type == "metric":
            metric_name = node_data.get('name', 'Unknown')
            metric_type = node_data.get('type', 'financial')
            return f"Financial metric {metric_name} of type {metric_type} used for analysis and reporting"
        
        elif node_type == "time_period":
            period = node_data.get('period', 'Unknown')
            year = node_data.get('year', '')
            return f"Time period {period} {year} for financial data and reporting"
        
        elif node_type == "sheet":
            sheet_name = node_data.get('name', 'Unknown')
            content_type = node_data.get('content_type', 'financial')
            return f"Excel sheet {sheet_name} containing {content_type} data and calculations"
        
        elif node_type == "data_point":
            value = node_data.get('value', 0)
            metric = node_data.get('metric', 'Unknown')
            company = node_data.get('company', 'Unknown')
            period = node_data.get('period', 'Unknown')
            return f"Data point: {company} {metric} value {value} for period {period}"
        
        elif node_type == "formula":
            formula = node_data.get('formula', 'Unknown')
            cell = node_data.get('cell', 'Unknown')
            return f"Excel formula {formula} in cell {cell} for calculations"
        
        else:
            return f"Node of type {node_type} with data: {str(node_data)[:100]}"
    
    def add_node(self, node_id: str, node_type: str, node_data: Dict, 
                 description: str = None) -> str:
        """
        Add a node to the knowledge graph with embedding.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (company, metric, time_period, etc.)
            node_data: Dictionary containing node data
            description: Optional custom description for embedding
            
        Returns:
            The node ID
        """
        # Create description if not provided
        if description is None:
            description = self.create_node_description(node_type, node_data)
        
        # Add node to graph
        self.graph.add_node(node_id, 
                           type=node_type, 
                           data=node_data, 
                           description=description)
        
        # Create embedding
        embedding = self.embedding_model.encode(description)
        self.node_embeddings[node_id] = embedding
        self.node_descriptions[node_id] = description
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str, 
                 edge_data: Dict = None) -> None:
        """
        Add an edge between nodes in the knowledge graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            edge_data: Additional edge data
        """
        if edge_data is None:
            edge_data = {}
        
        self.graph.add_edge(source_id, target_id, 
                           type=edge_type, 
                           data=edge_data)
    
    def build_company_nodes(self, companies: List[str]) -> List[str]:
        """Build nodes for each company."""
        company_nodes = []
        for company in companies:
            node_id = f"company_{company}"
            node_data = {
                'name': company,
                'type': 'company',
                'created_at': datetime.now().isoformat()
            }
            self.add_node(node_id, 'company', node_data)
            company_nodes.append(node_id)
        return company_nodes
    
    def build_metric_nodes(self, metrics: List[str]) -> List[str]:
        """Build nodes for financial metrics."""
        metric_nodes = []
        for metric in metrics:
            node_id = f"metric_{metric.replace(' ', '_')}"
            node_data = {
                'name': metric,
                'type': 'financial_metric',
                'category': self.categorize_metric(metric)
            }
            self.add_node(node_id, 'metric', node_data)
            metric_nodes.append(node_id)
        return metric_nodes
    
    def categorize_metric(self, metric: str) -> str:
        """Categorize a financial metric."""
        metric_lower = metric.lower()
        
        if any(word in metric_lower for word in ['revenue', 'income', 'sales']):
            return 'revenue'
        elif any(word in metric_lower for word in ['cost', 'expense', 'expenditure']):
            return 'expense'
        elif any(word in metric_lower for word in ['profit', 'margin', 'ebitda']):
            return 'profitability'
        elif any(word in metric_lower for word in ['cash', 'flow', 'fcf']):
            return 'cash_flow'
        elif any(word in metric_lower for word in ['debt', 'loan', 'liability']):
            return 'debt'
        else:
            return 'other'
    
    def build_time_period_nodes(self, years: List[int], months: List[str]) -> List[str]:
        """Build nodes for time periods."""
        time_nodes = []
        
        # Add year nodes
        for year in years:
            node_id = f"year_{year}"
            node_data = {
                'year': year,
                'type': 'year',
                'period': 'annual'
            }
            self.add_node(node_id, 'time_period', node_data)
            time_nodes.append(node_id)
        
        # Add month nodes
        for month in months:
            node_id = f"month_{month}"
            node_data = {
                'month': month,
                'type': 'month',
                'period': 'monthly'
            }
            self.add_node(node_id, 'time_period', node_data)
            time_nodes.append(node_id)
        
        return time_nodes
    
    def build_sheet_nodes(self, sheets_data: Dict) -> List[str]:
        """Build nodes for Excel sheets."""
        sheet_nodes = []
        
        for sheet_name, sheet_data in sheets_data.items():
            node_id = f"sheet_{sheet_name.replace(' ', '_')}"
            
            # Determine content type
            content_type = 'financial'
            if 'debt' in sheet_name.lower():
                content_type = 'debt_schedule'
            elif 'forecast' in sheet_name.lower():
                content_type = 'forecast'
            
            node_data = {
                'name': sheet_name,
                'content_type': content_type,
                'rows': len(sheet_data.get('data', pd.DataFrame())),
                'columns': len(sheet_data.get('data', pd.DataFrame()).columns) if len(sheet_data.get('data', pd.DataFrame())) > 0 else 0
            }
            
            self.add_node(node_id, 'sheet', node_data)
            sheet_nodes.append(node_id)
        
        return sheet_nodes
    
    def build_data_point_nodes(self, monthly_data: Dict) -> List[str]:
        """Build nodes for individual data points."""
        data_nodes = []
        
        for period_key, period_data in monthly_data.items():
            # Parse period key (e.g., "Jan_2023")
            if '_' in period_key:
                month, year = period_key.split('_', 1)
            else:
                month, year = 'Unknown', 'Unknown'
            
            for item_name, value in period_data.items():
                if isinstance(value, (int, float)):
                    node_id = f"data_{period_key}_{item_name.replace(' ', '_')}"
                    
                    node_data = {
                        'value': value,
                        'period': period_key,
                        'month': month,
                        'year': year,
                        'item_name': item_name,
                        'type': 'data_point'
                    }
                    
                    self.add_node(node_id, 'data_point', node_data)
                    data_nodes.append(node_id)
        
        return data_nodes
    
    def build_formula_nodes(self, formulas: Dict) -> List[str]:
        """Build nodes for Excel formulas."""
        formula_nodes = []
        
        for cell_ref, formula in formulas.items():
            node_id = f"formula_{cell_ref.replace(':', '_')}"
            
            node_data = {
                'formula': formula,
                'cell': cell_ref,
                'type': 'formula'
            }
            
            self.add_node(node_id, 'formula', node_data)
            formula_nodes.append(node_id)
        
        return formula_nodes
    
    def build_relationships(self, company_nodes: List[str], metric_nodes: List[str], 
                          time_nodes: List[str], sheet_nodes: List[str], 
                          data_nodes: List[str], formula_nodes: List[str]) -> None:
        """Build relationships between nodes."""
        
        # Connect companies to metrics
        for company_node in company_nodes:
            for metric_node in metric_nodes:
                self.add_edge(company_node, metric_node, 'has_metric')
        
        # Connect companies to sheets
        for company_node in company_nodes:
            company_name = self.graph.nodes[company_node]['data']['name']
            for sheet_node in sheet_nodes:
                sheet_name = self.graph.nodes[sheet_node]['data']['name']
                if company_name.lower() in sheet_name.lower():
                    self.add_edge(company_node, sheet_node, 'has_sheet')
        
        # Connect data points to companies and metrics
        for data_node in data_nodes:
            data_info = self.graph.nodes[data_node]['data']
            
            # Find matching company
            for company_node in company_nodes:
                company_name = self.graph.nodes[company_node]['data']['name']
                if company_name.lower() in str(data_info).lower():
                    self.add_edge(data_node, company_node, 'belongs_to_company')
                    break
            
            # Find matching metric
            for metric_node in metric_nodes:
                metric_name = self.graph.nodes[metric_node]['data']['name']
                if metric_name.lower() in str(data_info).lower():
                    self.add_edge(data_node, metric_node, 'represents_metric')
                    break
        
        # Connect formulas to sheets
        for formula_node in formula_nodes:
            for sheet_node in sheet_nodes:
                self.add_edge(sheet_node, formula_node, 'contains_formula')
    
    def build_index(self) -> None:
        """Build FAISS index for similarity search."""
        if not self.node_embeddings:
            return
        
        # Convert embeddings to numpy array
        embeddings_list = list(self.node_embeddings.values())
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.embedding_index = faiss.IndexFlatL2(dimension)
        self.embedding_index.add(embeddings_array)
        
        # Create mapping
        node_ids = list(self.node_embeddings.keys())
        self.node_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        self.index_to_node = {idx: node_id for idx, node_id in enumerate(node_ids)}
    
    def find_similar_nodes(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find nodes similar to a query using embeddings.
        
        Args:
            query: Query string
            top_k: Number of similar nodes to return
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if self.embedding_index is None:
            self.build_index()
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search for similar nodes
        distances, indices = self.embedding_index.search(query_embedding, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.index_to_node):
                node_id = self.index_to_node[idx]
                similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                results.append((node_id, similarity_score))
        
        return results
    
    def get_node_neighbors(self, node_id: str, max_depth: int = 2) -> List[str]:
        """
        Get neighboring nodes up to a certain depth.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of neighboring node IDs
        """
        if node_id not in self.graph:
            return []
        
        neighbors = set()
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_node, depth = queue.pop(0)
            
            if current_node in visited or depth > max_depth:
                continue
            
            visited.add(current_node)
            neighbors.add(current_node)
            
            if depth < max_depth:
                # Add neighbors
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
                
                # Add predecessors
                for predecessor in self.graph.predecessors(current_node):
                    if predecessor not in visited:
                        queue.append((predecessor, depth + 1))
        
        return list(neighbors)
    
    def get_graph_summary(self) -> Dict:
        """Get a summary of the knowledge graph."""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(self.graph.nodes(data='type')),
            'edge_types': {(source, target): data.get('type', 'unknown') for source, target, data in self.graph.edges(data=True)},
            'has_embeddings': len(self.node_embeddings) > 0,
            'has_index': self.embedding_index is not None
        }
    
    def save_graph(self, filepath: str) -> None:
        """Save the knowledge graph to a file."""
        graph_data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': dict(self.graph.edges(data=True)),
            'node_embeddings': {k: v.tolist() for k, v in self.node_embeddings.items()},
            'node_descriptions': self.node_descriptions
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load_graph(self, filepath: str) -> None:
        """Load the knowledge graph from a file."""
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        # Rebuild graph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for node_id, node_data in graph_data['nodes'].items():
            self.graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge, edge_data in graph_data['edges'].items():
            source, target = edge
            self.graph.add_edge(source, target, **edge_data)
        
        # Restore embeddings
        self.node_embeddings = {k: np.array(v) for k, v in graph_data['node_embeddings'].items()}
        self.node_descriptions = graph_data['node_descriptions']
        
        # Rebuild index
        self.build_index()
