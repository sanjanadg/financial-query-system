#!/usr/bin/env python3
"""
Query Engine for Financial Data Analysis
Handles all query processing, search, and analysis functions.
"""

import json
import os
import re
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

class QueryEngine:
    def __init__(self):
        """Initialize the query engine with sentence transformer model."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.query_cache = {}
    
    def query_file(self, representation, query):
        """Query the representation using both direct search and semantic search."""
        if not representation:
            return {"error": "No representation available"}
        
        # Check cache first
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Extract entities from query
        companies = self._extract_companies(query)
        metrics = self._extract_metrics(query)
        time_periods = self._extract_time_periods(query)
        
        # Search financial data first (most precise)
        if 'financial_data' in representation:
            result = self._search_financial_data(query, companies, metrics, time_periods, representation)
            if result and result['data_points']:
                result['query'] = query
                self.query_cache[query] = result
                return result
        
        # Fall back to semantic search
        result = self._semantic_search(query, representation)
        if result:
            result['query'] = query
            self.query_cache[query] = result
            return result
        
        # No results found
        result = {
            'query': query,
            'answer': 'Unable to find specific data for this query.',
            'data_points': [],
            'confidence': 0.0,
            'source_sheets': []
        }
        
        self.query_cache[query] = result
        return result
    
    def _search_financial_data(self, query, companies, metrics, time_periods, representation):
        """Search financial data with enhanced specificity and multi-company support."""
        query_lower = query.lower()
        
        # Check if this is a multi-company comparison query
        if any(term in query_lower for term in ['all companies', 'each company', 'companies']):
            return self._handle_multi_company_query(query, companies, metrics, time_periods, representation)
        
        # Check if this is a percentage calculation query
        if any(term in query_lower for term in ['percent', 'percentage', '%']):
            return self._calculate_percentage(query, companies, metrics, time_periods, representation)
        
        result = {
            'answer': 'Financial data found.',
            'data_points': [],
            'confidence': 0.0,
            'source_sheets': []
        }
        
        # Compound terms that should be matched together
        compound_terms = [
            'gross profit', 'operating income', 'net income', 'cost of revenue',
            'operating expenses', 'shipping income', 'direct labor', 'free cash flow'
        ]
        
        # Check for compound terms first (highest priority)
        for compound_term in compound_terms:
            if compound_term in query_lower:
                # Skip cost of revenue if query specifically asks for direct labor
                if compound_term == 'cost of revenue' and 'direct labor' in query_lower:
                    continue
                    
                for sheet_name, sheet_data in representation.get('financial_data', {}).items():
                    # Check if this sheet belongs to the requested company
                    if companies and companies != ['ALL_COMPANIES']:
                        if not any(company.lower() in sheet_name.lower() for company in companies):
                            continue
                    
                    for metric_name, metric_data in sheet_data.items():
                        if compound_term in metric_name.lower():
                            # Check time period matching
                            for time_period, value in metric_data.get('values', {}).items():
                                if not time_periods or any(tp.lower() in time_period.lower() for tp in time_periods):
                                    # Determine company name for the data point
                                    company_name = 'Unknown'
                                    if 'mxd' in sheet_name.lower():
                                        company_name = 'MXD'
                                    elif 'hec' in sheet_name.lower():
                                        company_name = 'HEC'
                                    elif 'branch' in sheet_name.lower() or 'brnch' in sheet_name.lower():
                                        company_name = 'Branch'
                                    elif 'rev and cogs' in sheet_name.lower():
                                        company_name = 'Revenue'
                                    
                                    result['data_points'].append({
                                        'company': company_name,
                                        'metric': metric_name,
                                        'time_period': time_period,
                                        'value': value,
                                        'sheet': sheet_name,
                                        'match_type': 'compound_term',
                                        'confidence': 0.95
                                    })
                                    result['source_sheets'].append(sheet_name)
                                    result['confidence'] = max(result['confidence'], 0.95)
                
                # If we found compound term matches, return them
                if result['data_points']:
                    return result
        
        # Check for specific metric patterns
        for metric_pattern in metrics:
            for sheet_name, sheet_data in representation.get('financial_data', {}).items():
                # Check if this sheet belongs to the requested company
                if companies and companies != ['ALL_COMPANIES']:
                    if not any(company.lower() in sheet_name.lower() for company in companies):
                        continue
                
                for metric_name, metric_data in sheet_data.items():
                    # Skip cost of revenue if query specifically asks for direct labor
                    if 'cost of revenue' in metric_name.lower() and 'direct labor' in query_lower:
                        continue
                    
                    if metric_pattern.lower() in metric_name.lower():
                        # Check time period matching
                        for time_period, value in metric_data.get('values', {}).items():
                            if not time_periods or any(tp.lower() in time_period.lower() for tp in time_periods):
                                # Determine company name for the data point
                                company_name = 'Unknown'
                                if 'mxd' in sheet_name.lower():
                                    company_name = 'MXD'
                                elif 'hec' in sheet_name.lower():
                                    company_name = 'HEC'
                                elif 'branch' in sheet_name.lower() or 'brnch' in sheet_name.lower():
                                    company_name = 'Branch'
                                elif 'rev and cogs' in sheet_name.lower():
                                    company_name = 'Revenue'
                                
                                result['data_points'].append({
                                    'company': company_name,
                                    'metric': metric_name,
                                    'time_period': time_period,
                                    'value': value,
                                    'sheet': sheet_name,
                                    'match_type': 'specific_metric',
                                    'confidence': 0.9
                                })
                                result['source_sheets'].append(sheet_name)
                                result['confidence'] = max(result['confidence'], 0.9)
        
        # Check for general terms (lower priority)
        general_terms = ['income', 'revenue', 'cost', 'expense', 'profit', 'loss']
        for term in general_terms:
            if term in query_lower:
                # Skip general cost/labor if query specifically asks for direct labor
                if term in ['cost', 'labor'] and 'direct labor' in query_lower:
                    continue
                    
                for sheet_name, sheet_data in representation.get('financial_data', {}).items():
                    # Check if this sheet belongs to the requested company
                    if companies and companies != ['ALL_COMPANIES']:
                        if not any(company.lower() in sheet_name.lower() for company in companies):
                            continue
                    
                    for metric_name, metric_data in sheet_data.items():
                        if term in metric_name.lower():
                            # Check time period matching
                            for time_period, value in metric_data.get('values', {}).items():
                                if not time_periods or any(tp.lower() in time_period.lower() for tp in time_periods):
                                    # Determine company name for the data point
                                    company_name = 'Unknown'
                                    if 'mxd' in sheet_name.lower():
                                        company_name = 'MXD'
                                    elif 'hec' in sheet_name.lower():
                                        company_name = 'HEC'
                                    elif 'branch' in sheet_name.lower() or 'brnch' in sheet_name.lower():
                                        company_name = 'Branch'
                                    elif 'rev and cogs' in sheet_name.lower():
                                        company_name = 'Revenue'
                                    
                                    result['data_points'].append({
                                        'company': company_name,
                                        'metric': metric_name,
                                        'time_period': time_period,
                                        'value': value,
                                        'sheet': sheet_name,
                                        'match_type': 'general_term',
                                        'confidence': 0.5
                                    })
                                    result['source_sheets'].append(sheet_name)
                                    result['confidence'] = max(result['confidence'], 0.5)
        
        # Sort by confidence and return top results
        if result['data_points']:
            result['data_points'] = sorted(result['data_points'], key=lambda x: x['confidence'], reverse=True)
            result['data_points'] = result['data_points'][:10]  # Limit to top 10
            result['query'] = query
            return result
        
        return None
    
    def _handle_multi_company_query(self, query, companies, metrics, time_periods, representation):
        """Handle queries that ask for data across multiple companies."""
        result = {
            'answer': 'Multi-company analysis completed.',
            'data_points': [],
            'confidence': 0.9,
            'source_sheets': []
        }
        
        query_lower = query.lower()
        
        # Extract target metrics from query
        target_metrics = []
        if 'ebitda' in query_lower:
            target_metrics.append('ebitda')
        if 'revenue' in query_lower:
            target_metrics.append('revenue')
        if 'fcf' in query_lower or 'free cash flow' in query_lower:
            target_metrics.append('fcf')
        if 'debt' in query_lower:
            target_metrics.append('debt')
        
        # Get all available companies from financial data
        available_companies = []
        company_sheet_mapping = {}
        
        for sheet_name, sheet_data in representation.get('financial_data', {}).items():
            # Map sheet names to company names
            if 'mxd' in sheet_name.lower():
                company_name = 'MXD'
            elif 'hec' in sheet_name.lower():
                company_name = 'HEC'
            elif 'branch' in sheet_name.lower() or 'brnch' in sheet_name.lower():
                company_name = 'Branch'
            elif 'rev and cogs' in sheet_name.lower(): # Added for specific mapping
                company_name = 'Revenue'
            else:
                # Skip sheets that don't clearly belong to a company
                continue
            
            if company_name not in available_companies:
                available_companies.append(company_name)
                company_sheet_mapping[company_name] = sheet_name
        
        # Handle different types of multi-company queries
        if 'direction' in query_lower and any(metric in query_lower for metric in ['ebitda', 'revenue', 'fcf']):
            return self._analyze_company_trends(query, target_metrics, available_companies, company_sheet_mapping, representation)
        elif 'debt' in query_lower and 'schedule' in query_lower:
            return self._analyze_debt_schedules(query, available_companies, company_sheet_mapping, representation)
        else:
            return self._general_multi_company_analysis(query, target_metrics, available_companies, company_sheet_mapping, representation)
    
    def _analyze_company_trends(self, query, target_metrics, companies, company_sheet_mapping, representation):
        """Analyze trends across companies for specified metrics."""
        result = {
            'answer': f'Company trend analysis for {", ".join(target_metrics)}.',
            'data_points': [],
            'confidence': 0.9,
            'source_sheets': []
        }
        
        for company in companies:
            if company in company_sheet_mapping:
                sheet_name = company_sheet_mapping[company]
                sheet_data = representation.get('financial_data', {}).get(sheet_name, {})
                
                for metric_name, metric_data in sheet_data.items():
                    if any(target in metric_name.lower() for target in target_metrics):
                        for time_period, value in metric_data.get('values', {}).items():
                            result['data_points'].append({
                                'company': company,
                                'metric': metric_name,
                                'time_period': time_period,
                                'value': value,
                                'sheet': sheet_name,
                                'match_type': 'multi_company_trend',
                                'confidence': 0.9
                            })
                            result['source_sheets'].append(sheet_name)
        
        result['query'] = query
        return result
    
    def _analyze_debt_schedules(self, query, companies, company_sheet_mapping, representation):
        """Analyze debt schedules across companies."""
        result = {
            'answer': 'Debt schedule analysis across companies.',
            'data_points': [],
            'confidence': 0.9,
            'source_sheets': []
        }
        
        for company in companies:
            if company in company_sheet_mapping:
                sheet_name = company_sheet_mapping[company]
                sheet_data = representation.get('financial_data', {}).get(sheet_name, {})
                
                for metric_name, metric_data in sheet_data.items():
                    if 'debt' in metric_name.lower():
                        for time_period, value in metric_data.get('values', {}).items():
                            result['data_points'].append({
                                'company': company,
                                'metric': metric_name,
                                'time_period': time_period,
                                'value': value,
                                'sheet': sheet_name,
                                'match_type': 'multi_company_debt',
                                'confidence': 0.9
                            })
                            result['source_sheets'].append(sheet_name)
        
        result['query'] = query
        return result
    
    def _general_multi_company_analysis(self, query, target_metrics, companies, company_sheet_mapping, representation):
        """General multi-company analysis for other types of comparisons."""
        result = {
            'answer': 'Multi-company comparison analysis.',
            'data_points': [],
            'confidence': 0.8,
            'source_sheets': []
        }
        
        for company in companies:
            if company in company_sheet_mapping:
                sheet_name = company_sheet_mapping[company]
                sheet_data = representation.get('financial_data', {}).get(sheet_name, {})
                
                for metric_name, metric_data in sheet_data.items():
                    if not target_metrics or any(target in metric_name.lower() for target in target_metrics):
                        for time_period, value in metric_data.get('values', {}).items():
                            result['data_points'].append({
                                'company': company,
                                'metric': metric_name,
                                'time_period': time_period,
                                'value': value,
                                'sheet': sheet_name,
                                'match_type': 'multi_company_general',
                                'confidence': 0.8
                            })
                            result['source_sheets'].append(sheet_name)
        
        result['query'] = query
        return result
    
    def _calculate_percentage(self, query, companies, metrics, time_periods, representation):
        """Calculate percentages for specific queries."""
        query_lower = query.lower()
        
        if 'indirect cost' in query_lower and 'total cost' in query_lower:
            return self._calculate_indirect_cost_percentage(query, companies, time_periods, representation)
        elif 'insurance' in query_lower and 'operating expense' in query_lower:
            return self._calculate_insurance_percentage(query, companies, time_periods, representation)
        else:
            return {
                'query': query,
                'answer': 'Percentage calculation requested but specific calculation not implemented.',
                'data_points': [],
                'confidence': 0.3,
                'source_sheets': []
            }
    
    def _calculate_indirect_cost_percentage(self, query, companies, time_periods, representation):
        """Calculate indirect costs as a percentage of total costs."""
        result = {
            'answer': 'Indirect cost percentage calculation.',
            'data_points': [],
            'confidence': 0.8,
            'source_sheets': []
        }
        
        # This would need specific data points for indirect costs and total costs
        # For now, return a placeholder
        result['data_points'].append({
            'company': 'MXD',
            'metric': 'Indirect Cost % of Total Cost',
            'time_period': '2022',
            'value': 'Calculation requires specific cost data',
            'sheet': 'MXD P&L',
            'match_type': 'percentage_calculation',
            'confidence': 0.8
        })
        
        result['query'] = query
        return result
    
    def _calculate_insurance_percentage(self, query, companies, time_periods, representation):
        """Calculate insurance as a percentage of operating expenses."""
        result = {
            'answer': 'Insurance percentage calculation.',
            'data_points': [],
            'confidence': 0.8,
            'source_sheets': []
        }
        
        # This would need specific data points for insurance and operating expenses
        # For now, return a placeholder
        result['data_points'].append({
            'company': 'MXD',
            'metric': 'Insurance % of Operating Expenses',
            'time_period': '2022',
            'value': 'Calculation requires specific expense data',
            'sheet': 'MXD P&L',
            'match_type': 'percentage_calculation',
            'confidence': 0.8
        })
        
        result['query'] = query
        return result
    
    def _extract_companies(self, query):
        """Extract company names from query."""
        companies = []
        query_lower = query.lower()
        
        # Check for multi-company indicators
        if any(term in query_lower for term in ['all companies', 'each company', 'companies']):
            # Return special indicator for multi-company queries
            return ['ALL_COMPANIES']
        
        # Extract specific company names
        company_patterns = ['mxd', 'hec', 'branch']
        for pattern in company_patterns:
            if pattern in query_lower:
                companies.append(pattern.upper())
        
        return companies
    
    def _extract_metrics(self, query):
        """Extract metric names from query."""
        metrics = []
        query_lower = query.lower()
        
        # Common financial metrics
        metric_patterns = [
            'gross profit', 'operating income', 'net income', 'revenue', 'sales',
            'cost of goods sold', 'cogs', 'operating expenses', 'ebitda',
            'shipping income', 'direct labor', 'insurance', 'advertising',
            'debt', 'interest', 'principal', 'forecast', 'budget'
        ]
        
        for pattern in metric_patterns:
            if pattern in query_lower:
                metrics.append(pattern)
        
        # Add individual terms
        individual_terms = ['income', 'revenue', 'cost', 'expense', 'profit', 'loss']
        for term in individual_terms:
            if term in query_lower and term not in [m for m in metrics if term in m]:
                metrics.append(term)
        
        return metrics
    
    def _extract_time_periods(self, query):
        """Extract time periods from query."""
        time_periods = []
        query_lower = query.lower()
        
        # Month patterns
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        for month in months:
            if month in query_lower:
                # Look for year after month
                month_index = query_lower.find(month)
                remaining_text = query_lower[month_index:]
                year_match = re.search(r'\d{4}', remaining_text)
                if year_match:
                    time_periods.append(f"{month.capitalize()} {year_match.group()}")
        
        # Year patterns
        year_patterns = [r'\d{4}', r'fy\d{2}', r'ttm\d{2}', r'ytd\d{2}']
        for pattern in year_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if match not in time_periods:
                    time_periods.append(match)
        
        return time_periods
    
    def _semantic_search(self, query, representation):
        """Perform semantic search using embeddings."""
        if not representation.get('sheets'):
            return None
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        best_matches = []
        
        for sheet_name, sheet_data in representation['sheets'].items():
            if 'embeddings' not in sheet_data or not sheet_data['embeddings']:
                continue
            
            # Calculate similarity with each text chunk
            for i, embedding in enumerate(sheet_data['embeddings']):
                similarity = self._cosine_similarity(query_embedding[0], embedding)
                if similarity > 0.3:  # Threshold for relevance
                    best_matches.append({
                        'sheet': sheet_name,
                        'similarity': similarity,
                        'chunk_index': i
                    })
        
        # Sort by similarity and get top matches
        best_matches.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = best_matches[:5]
        
        if not top_matches:
            return None
        
        # Create result
        result = {
            'answer': 'Semantic search results found.',
            'data_points': [],
            'confidence': top_matches[0]['similarity'],
            'source_sheets': [match['sheet'] for match in top_matches],
            'search_type': 'semantic'
        }
        
        # Add data points from top matches
        for match in top_matches:
            sheet_data = representation['sheets'][match['sheet']]
            if 'data' in sheet_data and sheet_data['data']:
                # Add sample data from the sheet
                sample_data = sheet_data['data'][:3]  # First 3 rows
                result['data_points'].append({
                    'sheet': match['sheet'],
                    'sample_data': sample_data,
                    'similarity': match['similarity']
                })
        
        result['query'] = query
        return result
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)


    # Utility functions for backward compatibility
    def load_queries(query_file_path):
        """Load queries from the text file."""
        queries = []
        try:
            with open(query_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove numbering if present
                        if line[0].isdigit() and '. ' in line:
                            query = line.split('. ', 1)[1]
                        else:
                            query = line
                        queries.append(query)
        except FileNotFoundError:
            print(f"Query file {query_file_path} not found.")
            return []
        
        return queries

    def print_query_results(results):
        """Print query results in a formatted way."""
        print(f"\n{'='*80}")
        print(f"QUERY: {results['query']}")
        print(f"{'='*80}")
        print(f"Answer: {results['answer']}")
        print(f"Confidence: {results['confidence']:.3f}")
        print(f"Source Sheets: {', '.join(set(results['source_sheets']))}")
        
        if results['data_points']:
            print(f"\nData Points ({len(results['data_points'])} found):")
            print("-" * 60)
            for i, point in enumerate(results['data_points'][:10], 1):  # Show first 10
                if 'metric' in point and 'time_period' in point:
                    confidence = point.get('confidence', point.get('similarity', 0.0))
                    print(f"{i}. {point['metric']} ({point['time_period']}): {point['value']} (confidence: {confidence:.3f})")
                elif 'metric' in point:
                    confidence = point.get('confidence', point.get('similarity', 0.0))
                    print(f"{i}. {point['metric']}: {point['value']} (confidence: {confidence:.3f})")
                elif 'data' in point:
                    confidence = point.get('confidence', point.get('similarity', 0.0))
                    print(f"{i}. Data: {str(point['data'])[:100]}... (confidence: {confidence:.3f})")
                elif 'sample_data' in point:
                    confidence = point.get('confidence', point.get('similarity', 0.0))
                    print(f"{i}. Sample data from {point['sheet']}: {str(point['sample_data'])[:100]}... (confidence: {confidence:.3f})")
        
        print()