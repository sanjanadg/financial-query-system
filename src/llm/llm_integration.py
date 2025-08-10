#!/usr/bin/env python3
"""
LLM Integration for Enhanced Query Understanding and Graph Traversal
"""

import os
import json
import openai
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ..graph.hierarchical_graph import HierarchicalGraphBuilder
from ..graph.hierarchical_extraction import ExcelDataRepresentation

@dataclass
class CostLimits:
    """Cost control limits for OpenAI API usage."""
    daily_budget_usd: float = 5.0  # Daily spending limit
    monthly_budget_usd: float = 50.0  # Monthly spending limit
    max_requests_per_minute: int = 10  # Rate limiting
    max_requests_per_hour: int = 100  # Hourly rate limiting
    max_tokens_per_request: int = 2000  # Token limit per request
    cost_per_1k_tokens: Dict[str, float] = field(default_factory=lambda: {
        "gpt-4o": 0.005,
        "gpt-4o-mini": 0.00015,
        "gpt-3.5-turbo": 0.0005
    })

@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    model: str = "gpt-4o-mini"  # Changed default to cheaper model
    temperature: float = 0.1
    max_tokens: int = 1000
    api_key: Optional[str] = None
    cost_limits: CostLimits = field(default_factory=CostLimits)
    enable_cost_tracking: bool = True
    enable_rate_limiting: bool = True

class CostTracker:
    """Tracks OpenAI API usage and costs."""
    
    def __init__(self, cost_limits: CostLimits):
        self.cost_limits = cost_limits
        self.daily_costs = {}
        self.monthly_costs = {}
        self.request_history = []
        self.lock = threading.Lock()
        
    def track_request(self, model: str, tokens_used: int, cost_usd: float):
        """Track a single API request."""
        with self.lock:
            timestamp = datetime.now()
            date_key = timestamp.strftime("%Y-%m-%d")
            month_key = timestamp.strftime("%Y-%m")
            
            # Track daily costs
            if date_key not in self.daily_costs:
                self.daily_costs[date_key] = 0.0
            self.daily_costs[date_key] += cost_usd
            
            # Track monthly costs
            if month_key not in self.monthly_costs:
                self.monthly_costs[month_key] = 0.0
            self.monthly_costs[month_key] += cost_usd
            
            # Track request history
            self.request_history.append({
                'timestamp': timestamp,
                'model': model,
                'tokens': tokens_used,
                'cost': cost_usd
            })
            
            # Keep only last 1000 requests in memory
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]
    
    def can_make_request(self, model: str, estimated_tokens: int) -> Tuple[bool, str]:
        """Check if a request can be made within cost limits."""
        with self.lock:
            today = datetime.now().strftime("%Y-%m-%d")
            this_month = datetime.now().strftime("%Y-%m")
            
            # Check daily budget
            daily_cost = self.daily_costs.get(today, 0.0)
            estimated_cost = (estimated_tokens / 1000) * self.cost_limits.cost_per_1k_tokens.get(model, 0.001)
            
            if daily_cost + estimated_cost > self.cost_limits.daily_budget_usd:
                return False, f"Daily budget exceeded. Current: ${daily_cost:.2f}, Limit: ${self.cost_limits.daily_budget_usd}"
            
            # Check monthly budget
            monthly_cost = self.monthly_costs.get(this_month, 0.0)
            if monthly_cost + estimated_cost > self.cost_limits.monthly_budget_usd:
                return False, f"Monthly budget exceeded. Current: ${monthly_cost:.2f}, Limit: ${self.cost_limits.monthly_budget_usd}"
            
            return True, "OK"
    
    def get_usage_summary(self) -> Dict:
        """Get summary of API usage and costs."""
        with self.lock:
            today = datetime.now().strftime("%Y-%m-%d")
            this_month = datetime.now().strftime("%Y-%m")
            
            return {
                'daily_cost': self.daily_costs.get(today, 0.0),
                'monthly_cost': self.monthly_costs.get(this_month, 0.0),
                'daily_limit': self.cost_limits.daily_budget_usd,
                'monthly_limit': self.cost_limits.monthly_budget_usd,
                'total_requests': len(self.request_history),
                'recent_requests': self.request_history[-10:] if self.request_history else []
            }

class RateLimiter:
    """Rate limiting for API requests."""
    
    def __init__(self, max_per_minute: int, max_per_hour: int):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.minute_requests = []
        self.hour_requests = []
        self.lock = threading.Lock()
    
    def can_make_request(self) -> Tuple[bool, float]:
        """Check if a request can be made within rate limits."""
        with self.lock:
            now = time.time()
            
            # Clean old requests
            self.minute_requests = [req for req in self.minute_requests if now - req < 60]
            self.hour_requests = [req for req in self.hour_requests if now - req < 3600]
            
            # Check limits
            if len(self.minute_requests) >= self.max_per_minute:
                wait_time = 60 - (now - self.minute_requests[0])
                return False, max(0, wait_time)
            
            if len(self.hour_requests) >= self.max_per_hour:
                wait_time = 3600 - (now - self.hour_requests[0])
                return False, max(0, wait_time)
            
            return True, 0.0
    
    def record_request(self):
        """Record a new request."""
        with self.lock:
            now = time.time()
            self.minute_requests.append(now)
            self.hour_requests.append(now)

class LLMQueryEnhancer:
    """Uses LLM to enhance query understanding."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.cost_tracker = CostTracker(config.cost_limits) if config.enable_cost_tracking else None
        self.rate_limiter = RateLimiter(
            config.cost_limits.max_requests_per_minute,
            config.cost_limits.max_requests_per_hour
        ) if config.enable_rate_limiting else None
        
        # Initialize OpenAI client
        if config.api_key:
            self.client = openai.OpenAI(api_key=config.api_key)
        elif os.getenv('OPENAI_API_KEY'):
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            print("Warning: No OpenAI API key found. LLM query enhancement will be disabled.")
            self.llm_available = False
            return
        
        self.llm_available = True
        self.query_cache = {}
    
    def enhance_query_understanding(self, query: str, file_context: Dict) -> Dict[str, Any]:
        """Use LLM to enhance query understanding."""
        if not self.llm_available:
            return self._fallback_query_parsing(query)
        
        cache_key = f"enhancement_{hash(query)}_{hash(str(file_context))}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Check rate limiting
        if self.rate_limiter:
            can_proceed, wait_time = self.rate_limiter.can_make_request()
            if not can_proceed:
                print(f"Rate limit reached. Please wait {wait_time:.1f} seconds.")
                return self._fallback_query_parsing(query)
        
        # Check cost limits
        if self.cost_tracker:
            estimated_tokens = self.config.max_tokens + len(query) // 4  # Rough estimate
            can_proceed, message = self.cost_tracker.can_make_request(self.config.model, estimated_tokens)
            if not can_proceed:
                print(f"Cost limit reached: {message}")
                return self._fallback_query_parsing(query)
        
        try:
            # Create enhancement prompt
            system_prompt = """Analyze the user's query and enhance it with better understanding.

IMPORTANT: You must respond with ONLY valid JSON. No additional text, explanations, or formatting.

Return a JSON response with:
- intent: string (specific_value, trend, percentage, comparison, etc.)
- entities: object with companies, metrics, time_periods, values arrays
- refined_query: string with improved query
- analysis_strategy: string describing approach
- confidence: float 0-1
- requires_clarification: boolean
- clarification_questions: array of strings if needed

Example response format:
{"intent": "trend", "entities": {"companies": [], "metrics": ["revenue"], "time_periods": [2023], "values": []}, "refined_query": "Show revenue trends for 2023", "analysis_strategy": "time_series_analysis", "confidence": 0.8, "requires_clarification": false, "clarification_questions": []}"""

            user_prompt = f"""Query: {query}

File Context:
{self._create_context_prompt(file_context)}

Enhance this query understanding."""

            # Record request start time for rate limiting
            if self.rate_limiter:
                self.rate_limiter.record_request()

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Track costs
            if self.cost_tracker:
                tokens_used = response.usage.total_tokens
                cost_usd = (tokens_used / 1000) * self.cost_tracker.cost_limits.cost_per_1k_tokens.get(self.config.model, 0.001)
                self.cost_tracker.track_request(self.config.model, tokens_used, cost_usd)
                print(f"API call cost: ${cost_usd:.4f} ({tokens_used} tokens)")
            
            # Parse LLM response
            try:
                enhanced_understanding = json.loads(response.choices[0].message.content)
                
                # Cache the result
                self.query_cache[cache_key] = enhanced_understanding
                
                return enhanced_understanding
            except json.JSONDecodeError as e:
                print(f"LLM returned invalid JSON: {response.choices[0].message.content}")
                print(f"JSON error: {e}")
                return self._fallback_query_parsing(query)
            
        except Exception as e:
            print(f"LLM query enhancement failed: {e}")
            return self._fallback_query_parsing(query)
    
    def _create_context_prompt(self, file_context: Dict) -> str:
        """Create context prompt for the LLM."""
        context_parts = []
        
        if 'companies' in file_context:
            context_parts.append(f"Available companies: {', '.join(file_context['companies'])}")
        
        if 'metrics' in file_context:
            context_parts.append(f"Available metrics: {', '.join(file_context['metrics'])}")
        
        if 'time_periods' in file_context:
            context_parts.append(f"Available time periods: {', '.join(file_context['time_periods'])}")
        
        if 'sheets' in file_context:
            context_parts.append(f"Available sheets: {', '.join(file_context['sheets'])}")
        
        return "\n".join(context_parts)
    
    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """Fallback to basic query parsing when LLM is not available."""
        return {
            "intent": "general",
            "entities": {
                "companies": [],
                "metrics": [],
                "time_periods": [],
                "values": []
            },
            "refined_query": query,
            "analysis_strategy": "basic_pattern_matching",
            "confidence": 0.5,
            "requires_clarification": False,
            "clarification_questions": []
        }

class LLMGraphTraverser:
    """Uses LLM to intelligently traverse the knowledge graph. 
    *Note that this is not currently in place and was an explored concept but not integrated into the main codebase.
    """
    def __init__(self, config: LLMConfig):
        self.config = config
        self.cost_tracker = CostTracker(config.cost_limits) if config.enable_cost_tracking else None
        self.rate_limiter = RateLimiter(
            config.cost_limits.max_requests_per_minute,
            config.cost_limits.max_requests_per_hour
        ) if config.enable_rate_limiting else None
        
        # Initialize OpenAI client
        if config.api_key:
            self.client = openai.OpenAI(api_key=config.api_key)
        elif os.getenv('OPENAI_API_KEY'):
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            print("Warning: No OpenAI API key found. LLM graph traversal will be disabled.")
            self.llm_available = False
            return
        
        self.llm_available = True
        self.traversal_cache = {}
    
    def intelligent_graph_traversal(self, query: str, kg: HierarchicalGraphBuilder, 
                                  initial_nodes: List[str]) -> Dict[str, Any]:
        """Use LLM to intelligently traverse the graph."""
        if not self.llm_available:
            return self._fallback_traversal(kg, initial_nodes)
        
        cache_key = f"traversal_{hash(query)}_{hash(str(initial_nodes))}"
        if cache_key in self.traversal_cache:
            return self.traversal_cache[cache_key]
        
        # Check rate limiting
        if self.rate_limiter:
            can_proceed, wait_time = self.rate_limiter.can_make_request()
            if not can_proceed:
                print(f"Rate limit reached. Please wait {wait_time:.1f} seconds.")
                return self._fallback_traversal(kg, initial_nodes)
        
        # Check cost limits
        if self.cost_tracker:
            estimated_tokens = self.config.max_tokens + len(query) // 4  # Rough estimate
            can_proceed, message = self.cost_tracker.can_make_request(self.config.model, estimated_tokens)
            if not can_proceed:
                print(f"Cost limit reached: {message}")
                return self._fallback_traversal(kg, initial_nodes)
        
        try:
            # Create traversal strategy prompt
            system_prompt = """You are an expert at navigating knowledge graphs. Analyze the query and suggest the best traversal strategy.

IMPORTANT: You must respond with ONLY valid JSON. No additional text, explanations, or formatting.

Return a JSON response with:
- relevant_node_ids: array of node IDs to focus on
- traversal_paths: array of traversal paths to follow
- reasoning: string explaining the strategy

Example response format:
{"relevant_node_ids": ["node1", "node2"], "traversal_paths": ["path1", "path2"], "reasoning": "Focus on revenue-related nodes and follow time-based paths"}

Note: If you don't have specific node IDs, use empty arrays and focus on the reasoning."""

            user_prompt = f"Query: {query}\nInitial nodes: {initial_nodes}\n\nWhat's the best traversal strategy?"

            # Record request start time for rate limiting
            if self.rate_limiter:
                self.rate_limiter.record_request()

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Track costs
            if self.cost_tracker:
                tokens_used = response.usage.total_tokens
                cost_usd = (tokens_used / 1000) * self.cost_tracker.cost_limits.cost_per_1k_tokens.get(self.config.model, 0.001)
                self.cost_tracker.track_request(self.config.model, tokens_used, cost_usd)
                print(f"API call cost: ${cost_usd:.4f} ({tokens_used} tokens)")
            
            # Parse LLM response
            try:
                traversal_strategy = json.loads(response.choices[0].message.content)
                
                # Execute the strategy
                result = self._execute_traversal_strategy(kg, traversal_strategy, initial_nodes)
                
                # Cache the result
                self.traversal_cache[cache_key] = result
                
                return result
            except json.JSONDecodeError as e:
                print(f"LLM returned invalid JSON: {response.choices[0].message.content}")
                print(f"JSON error: {e}")
                return self._fallback_traversal(kg, initial_nodes)
            
        except Exception as e:
            print(f"LLM graph traversal failed: {e}")
            return self._fallback_traversal(kg, initial_nodes)
    
    def _get_graph_context(self, kg: HierarchicalGraphBuilder, node_ids: List[str]) -> str:
        """Get context about the graph and nodes."""
        context_parts = []
        
        # Graph summary
        summary = kg.get_graph_summary()
        context_parts.append(f"Graph has {summary['total_nodes']} nodes and {summary['total_edges']} edges")
        
        # Node types
        node_types = {}
        for node_id in kg.graph.nodes():
            node_type = kg.graph.nodes[node_id].get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        context_parts.append(f"Node types: {dict(node_types)}")
        
        # Initial nodes info
        for node_id in node_ids[:5]:  # Limit to first 5
            if node_id in kg.graph.nodes:
                node_data = kg.graph.nodes[node_id]
                context_parts.append(f"Node {node_id}: {node_data.get('type', 'unknown')} - {node_data.get('name', 'unnamed')}")
        
        return "\n".join(context_parts)
    
    def _execute_traversal_strategy(self, kg: HierarchicalGraphBuilder, 
                                  strategy: Dict, initial_nodes: List[str]) -> Dict[str, Any]:
        """Execute the LLM-suggested traversal strategy."""
        relevant_nodes = strategy.get('relevant_node_ids', initial_nodes)
        traversal_paths = strategy.get('traversal_paths', [])
        
        # Find the suggested nodes
        found_nodes = []
        for node_id in relevant_nodes:
            if node_id in kg.graph.nodes:
                found_nodes.append(node_id)
        
        # Get node data
        node_data = {}
        for node_id in found_nodes:
            node_data[node_id] = kg.graph.nodes[node_id]
        
        return {
            'strategy': strategy,
            'found_nodes': found_nodes,
            'node_data': node_data,
            'traversal_paths': traversal_paths,
            'confidence': strategy.get('confidence', 0.5)
        }
    
    def _fallback_traversal(self, kg: HierarchicalGraphBuilder, initial_nodes: List[str]) -> Dict[str, Any]:
        """Fallback traversal when LLM is not available."""
        return {
            'strategy': {'reasoning': 'Fallback traversal'},
            'found_nodes': initial_nodes,
            'node_data': {node_id: kg.graph.nodes[node_id] for node_id in initial_nodes if node_id in kg.graph.nodes},
            'traversal_paths': [],
            'confidence': 0.3
        }

class LLMResponseGenerator:
    """Uses LLM to generate natural language responses."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.cost_tracker = CostTracker(config.cost_limits) if config.enable_cost_tracking else None
        self.rate_limiter = RateLimiter(
            config.cost_limits.max_requests_per_minute,
            config.cost_limits.max_requests_per_hour
        ) if config.enable_rate_limiting else None
        
        # Initialize OpenAI client
        if config.api_key:
            self.client = openai.OpenAI(api_key=config.api_key)
        elif os.getenv('OPENAI_API_KEY'):
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            print("Warning: No OpenAI API key found. LLM response generation will be disabled.")
            self.llm_available = False
            return
        
        self.llm_available = True
        self.response_cache = {}
    
    def generate_response(self, query: str, data_results: Dict, 
                         enhanced_understanding: Dict) -> str:
        """Generate a natural language response using LLM."""
        if not self.llm_available:
            return self._fallback_response(query, data_results)
        
        cache_key = f"response_{hash(query)}_{hash(str(data_results))}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Check rate limiting
        if self.rate_limiter:
            can_proceed, wait_time = self.rate_limiter.can_make_request()
            if not can_proceed:
                print(f"Rate limit reached. Please wait {wait_time:.1f} seconds.")
                return self._fallback_response(query, data_results)
        
        # Check cost limits
        if self.cost_tracker:
            estimated_tokens = self.config.max_tokens + len(query) // 4  # Rough estimate
            can_proceed, message = self.cost_tracker.can_make_request(self.config.model, estimated_tokens)
            if not can_proceed:
                print(f"Cost limit reached: {message}")
                return self._fallback_response(f"Cost limit reached: {message}")
        
        try:
            # Create response generation prompt
            system_prompt = """You are an expert financial analyst. Generate clear, professional responses to financial data queries.

Your response should:
1. Answer the question directly and clearly
2. Provide relevant data and insights
3. Use appropriate financial terminology
4. Be concise but comprehensive
5. Include relevant context when helpful

Format your response in a professional, easy-to-read manner."""

            user_prompt = f"""Query: {query}

Enhanced Understanding: {json.dumps(enhanced_understanding, indent=2)}

Data Results: {json.dumps(data_results, indent=2)}

Please generate a comprehensive response to this query."""

            # Record request start time for rate limiting
            if self.rate_limiter:
                self.rate_limiter.record_request()

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Track costs
            if self.cost_tracker:
                tokens_used = response.usage.total_tokens
                cost_usd = (tokens_used / 1000) * self.cost_tracker.cost_limits.cost_per_1k_tokens.get(self.config.model, 0.001)
                self.cost_tracker.track_request(self.config.model, tokens_used, cost_usd)
                print(f"API call cost: ${cost_usd:.4f} ({tokens_used} tokens)")
            
            generated_response = response.choices[0].message.content
            
            # Cache the result
            self.response_cache[cache_key] = generated_response
            
            return generated_response
            
        except Exception as e:
            print(f"LLM response generation failed: {e}")
            return self._fallback_response(query, data_results)
    
    def _fallback_response(self, query: str, data_results: Dict) -> str:
        """Fallback response when LLM is not available."""
        return f"Query: {query}\n\nData found: {len(data_results.get('found_nodes', []))} relevant nodes\n\nPlease enable LLM integration for detailed responses."

def create_llm_config(api_key: Optional[str] = None, model: str = "gpt-4o-mini", 
                     daily_budget: float = 5.0, monthly_budget: float = 50.0,
                     enable_cost_tracking: bool = True, enable_rate_limiting: bool = True) -> LLMConfig:
    """Create LLM configuration with cost controls."""
    cost_limits = CostLimits(
        daily_budget_usd=daily_budget,
        monthly_budget_usd=monthly_budget
    )
    
    return LLMConfig(
        api_key=api_key,
        model=model,
        temperature=0.1,
        max_tokens=1000,
        cost_limits=cost_limits,
        enable_cost_tracking=enable_cost_tracking,
        enable_rate_limiting=enable_rate_limiting
    )
