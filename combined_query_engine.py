#!/usr/bin/env python3
"""
Combined Query Engine
Combines traditional query.py functionality with LLM query enhancement as a fallback.
Tries traditional queries first, then falls back to LLM if needed.
"""

import json
import os
from typing import Dict, Any, Optional
from query import QueryEngine
from llm_integration import LLMQueryEnhancer, LLMResponseGenerator, create_llm_config

class CombinedQueryEngine:
    """
    A query engine that combines traditional pattern-based queries with LLM enhancement.
    Tries traditional queries first, then falls back to LLM if needed.
    """
    
    def __init__(self, llm_api_key: Optional[str] = None, 
                 llm_model: str = "gpt-4o-mini",
                 confidence_threshold: float = 0.7,
                 enable_llm_fallback: bool = True):
        """
        Initialize the combined query engine.
        
        Args:
            llm_api_key: OpenAI API key for LLM fallback
            llm_model: LLM model to use for fallback
            confidence_threshold: Minimum confidence for traditional query results
            enable_llm_fallback: Whether to enable LLM fallback
        """
        # Initialize traditional query engine
        self.traditional_engine = QueryEngine()
        
        # Initialize LLM components if available
        self.enable_llm_fallback = enable_llm_fallback
        self.confidence_threshold = confidence_threshold
        
        if enable_llm_fallback and (llm_api_key or os.getenv('OPENAI_API_KEY')):
            try:
                self.llm_config = create_llm_config(
                    api_key=llm_api_key,
                    model=llm_model,
                    daily_budget=5.0,
                    monthly_budget=50.0
                )
                self.llm_enhancer = LLMQueryEnhancer(self.llm_config)
                self.llm_response_generator = LLMResponseGenerator(self.llm_config)
                self.llm_available = True
                print(f"LLM integration enabled with model: {llm_model}")
            except Exception as e:
                print(f"Failed to initialize LLM integration: {e}")
                self.llm_available = False
        else:
            self.llm_available = False
            print("LLM integration disabled - no API key or fallback disabled")
    
    def query_file(self, representation: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Query the file representation using combined approach.
        
        Args:
            representation: The processed file representation
            query: The query string
            
        Returns:
            Query results with metadata about which approach was used
        """
        # Step 1: Try traditional query engine first
        traditional_result = self.traditional_engine.query_file(representation, query)
        
        # Always enhance with LLM if available, regardless of traditional engine success
        if self.llm_available and self.enable_llm_fallback:
            llm_result = self._query_with_llm(representation, query, traditional_result)
            if llm_result:
                llm_result['query_method'] = 'traditional_enhanced' if self._is_traditional_query_successful(traditional_result) else 'llm_fallback'
                llm_result['llm_fallback_used'] = not self._is_traditional_query_successful(traditional_result)
                llm_result['traditional_confidence'] = traditional_result.get('confidence', 0.0)
                return llm_result
        
        # Return traditional result (either enhanced or not)
        if self._is_traditional_query_successful(traditional_result):
            traditional_result['query_method'] = 'traditional'
            traditional_result['llm_fallback_used'] = False
            return traditional_result
        else:
            traditional_result['query_method'] = 'traditional_low_confidence'
            traditional_result['llm_fallback_used'] = False
            return traditional_result
    
    def _is_traditional_query_successful(self, result: Dict[str, Any]) -> bool:
        """
        Determine if the traditional query result is considered successful.
        
        Args:
            result: The query result from traditional engine
            
        Returns:
            True if the result is considered successful
        """
        # Check if we have meaningful data
        if not result or 'error' in result:
            return False
        
        # Check confidence threshold
        confidence = result.get('confidence', 0.0)
        if confidence < self.confidence_threshold:
            return False
        
        # Check if we have data points
        data_points = result.get('data_points', [])
        if not data_points:
            return False
        
        # Check if data points have meaningful information
        meaningful_data = False
        for point in data_points:
            if 'value' in point and point['value']:
                meaningful_data = True
                break
        
        return meaningful_data
    
    def _query_with_llm(self, representation: Dict[str, Any], 
                        query: str, 
                        traditional_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to enhance the query understanding and generate a response.
        
        Args:
            representation: The processed file representation
            query: The original query string
            traditional_result: The result from traditional query engine
            
        Returns:
            Enhanced LLM result or None if failed
        """
        try:
            # Create context for LLM
            file_context = self._create_file_context(representation)
            
            # Enhance query understanding
            enhanced_understanding = self.llm_enhancer.enhance_query_understanding(query, file_context)
            
            if not enhanced_understanding:
                return None
            
            # Generate response using LLM
            llm_response = self.llm_response_generator.generate_response(
                query, traditional_result, enhanced_understanding
            )
            
            # Create comprehensive result
            result = {
                'query': query,
                'answer': llm_response,
                'data_points': traditional_result.get('data_points', []),
                'confidence': enhanced_understanding.get('confidence', 0.8),
                'source_sheets': traditional_result.get('source_sheets', []),
                'enhanced_understanding': enhanced_understanding,
                'llm_response': llm_response,
                'traditional_result': traditional_result
            }
            
            return result
            
        except Exception as e:
            print(f"  LLM query processing failed: {e}")
            return None
    
    def _create_file_context(self, representation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create context information for LLM queries.
        
        Args:
            representation: The processed file representation
            
        Returns:
            Context dictionary for LLM
        """
        context = {
            'companies': [],
            'metrics': [],
            'time_periods': [],
            'sheets': []
        }
        
        # Extract company information
        if 'financial_data' in representation:
            for sheet_name in representation['financial_data'].keys():
                if 'mxd' in sheet_name.lower():
                    context['companies'].append('MXD')
                elif 'hec' in sheet_name.lower():
                    context['companies'].append('HEC')
                elif 'branch' in sheet_name.lower() or 'brnch' in sheet_name.lower():
                    context['companies'].append('Branch')
                elif 'rev and cogs' in sheet_name.lower():
                    context['companies'].append('Revenue')
        
        # Remove duplicates
        context['companies'] = list(set(context['companies']))
        
        # Extract available metrics
        if 'financial_data' in representation:
            for sheet_data in representation['financial_data'].values():
                for metric_name in sheet_data.keys():
                    if metric_name not in context['metrics']:
                        context['metrics'].append(metric_name)
        
        # Extract time periods
        if 'financial_data' in representation:
            for sheet_data in representation['financial_data'].values():
                for metric_data in sheet_data.values():
                    if 'values' in metric_data:
                        for time_period in metric_data['values'].keys():
                            if time_period not in context['time_periods']:
                                context['time_periods'].append(time_period)
        
        # Extract sheet names
        if 'sheets' in representation:
            context['sheets'] = list(representation['sheets'].keys())
        
        return context
    
    def get_query_method_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the query methods used.
        
        Returns:
            Summary of query methods and performance
        """
        return {
            'traditional_engine_available': True,
            'llm_fallback_available': self.llm_available,
            'confidence_threshold': self.confidence_threshold,
            'llm_model': self.llm_config.model if hasattr(self, 'llm_config') else None,
            'enable_llm_fallback': self.enable_llm_fallback
        }
    
    def print_query_results(self, results: Dict[str, Any]):
        """
        Print query results with enhanced formatting for combined approach.
        
        Args:
            results: The query results
        """
        print(f"\n{'='*80}")
        print(f"QUERY: {results['query']}")
        print(f"{'='*80}")
        
        # Show query method used
        query_method = results.get('query_method', 'unknown')
        llm_used = results.get('llm_fallback_used', False)
        
        if query_method == 'traditional':
            print(f"Query Method: Traditional Engine (No LLM fallback needed)")
        elif query_method == 'llm_fallback':
            print(f"Query Method: LLM Fallback (Traditional failed)")
            if 'traditional_confidence' in results:
                print(f"   Traditional confidence: {results['traditional_confidence']:.3f}")
        elif query_method == 'traditional_low_confidence':
            print(f"Query Method: Traditional Engine (Low confidence, LLM fallback failed)")
        
        print(f"Answer: {results['answer']}")
        print(f"Confidence: {results['confidence']:.3f}")
        print(f"Source Sheets: {', '.join(set(results['source_sheets']))}")
        
        # Show enhanced understanding if available
        if 'enhanced_understanding' in results:
            enhanced = results['enhanced_understanding']
            print(f"\nEnhanced Understanding:")
            print(f"   Intent: {enhanced.get('intent', 'unknown')}")
            print(f"   Analysis Strategy: {enhanced.get('analysis_strategy', 'unknown')}")
            print(f"   Refined Query: {enhanced.get('refined_query', 'N/A')}")
        
        # Show data points
        if results.get('data_points'):
            print(f"\nData Points ({len(results['data_points'])} found):")
            print("-" * 60)
            for i, point in enumerate(results['data_points'][:10], 1):  # Show first 10
                if 'metric' in point and 'time_period' in point:
                    confidence = point.get('confidence', point.get('similarity', 0.0))
                    company = point.get('company', 'Unknown')
                    print(f"{i}. {company} - {point['metric']} ({point['time_period']}): {point['value']} (confidence: {confidence:.3f})")
                elif 'metric' in point:
                    confidence = point.get('confidence', point.get('similarity', 0.0))
                    print(f"{i}. {point['metric']}: {point['value']} (confidence: {confidence:.3f})")
                elif 'sample_data' in point:
                    confidence = point.get('confidence', point.get('similarity', 0.0))
                    print(f"{i}. Sample data from {point['sheet']}: {str(point['sample_data'])[:100]}... (confidence: {confidence:.3f})")
        
        print()


def main():
    """
    Example usage of the combined query engine.
    """
    # Check if we have the required data representation
    representation_file = "data_representation.json"
    
    if not os.path.exists(representation_file):
        print(f"Data representation file not found: {representation_file}")
        print("Please run the excel processor first to create the data representation.")
        return
    
    # Load the data representation
    try:
        with open(representation_file, 'r') as f:
            representation = json.load(f)
        print(f"Loaded data representation from {representation_file}")
    except Exception as e:
        print(f"Failed to load data representation: {e}")
        return
    
    # Initialize the combined query engine
    # You can set your OpenAI API key here or use environment variable
    llm_api_key = os.getenv('OPENAI_API_KEY')  # or set directly: "your-api-key-here"
    
    engine = CombinedQueryEngine(
        llm_api_key=llm_api_key,
        llm_model="gpt-4o-mini",
        confidence_threshold=0.7,
        enable_llm_fallback=True
    )
    
    # Show engine configuration
    config = engine.get_query_method_summary()
    print(f"\nCombined Query Engine Configuration:")
    print(f"   Traditional Engine: {'Available' if config['traditional_engine_available'] else 'Not Available'}")
    print(f"   LLM Fallback: {'Available' if config['llm_fallback_available'] else 'Not Available'}")
    print(f"   Confidence Threshold: {config['confidence_threshold']}")
    print(f"   LLM Model: {config['llm_model'] or 'Not Available'}")
    
    # Test queries
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
    
    print(f"\nTesting {len(test_queries)} queries with combined approach...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}/{len(test_queries)}")
        print(f"{'='*60}")
        
        try:
            result = engine.query_file(representation, query)
            engine.print_query_results(result)
        except Exception as e:
            print(f"Query failed with error: {e}")
        
        # Add a small delay between queries to avoid overwhelming the system
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()
