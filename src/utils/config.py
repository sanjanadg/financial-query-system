#!/usr/bin/env python3
"""
Configuration file for API key limits and cost controls
"""

import os
from typing import Optional

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Set this in your environment variables
OPENAI_MODEL = "gpt-4o-mini"  # Cheaper model for cost control

# Cost Control Limits
DAILY_BUDGET_USD = 5.0      # Maximum daily spending limit
MONTHLY_BUDGET_USD = 50.0   # Maximum monthly spending limit

# Rate Limiting
MAX_REQUESTS_PER_MINUTE = 10   # Maximum API calls per minute
MAX_REQUESTS_PER_HOUR = 100    # Maximum API calls per hour
MAX_TOKENS_PER_REQUEST = 2000  # Maximum tokens per request

# Feature Toggles
ENABLE_COST_TRACKING = True     # Track and log all API costs
ENABLE_RATE_LIMITING = True     # Enforce rate limits
ENABLE_LLM_FEATURES = True      # Enable/disable all LLM features

# Cost per 1K tokens (OpenAI pricing as of 2024)
COST_PER_1K_TOKENS = {
    "gpt-4o": 0.005,           # $0.005 per 1K tokens
    "gpt-4o-mini": 0.00015,    # $0.00015 per 1K tokens (much cheaper!)
    "gpt-3.5-turbo": 0.0005,   # $0.0005 per 1K tokens
}

# Cache Settings
CACHE_ENABLED = True
CACHE_EXPIRY_HOURS = 24  # Cache responses for 24 hours

# Logging
LOG_LEVEL = "INFO"
LOG_COSTS = True  # Log all cost information
LOG_REQUESTS = True  # Log all API requests

def get_cost_estimate(tokens: int, model: str = OPENAI_MODEL) -> float:
    """Estimate cost for a given number of tokens."""
    cost_per_1k = COST_PER_1K_TOKENS.get(model, 0.001)
    return (tokens / 1000) * cost_per_1k

def get_daily_cost_limit() -> float:
    """Get daily cost limit from environment or default."""
    return float(os.getenv('OPENAI_DAILY_BUDGET', DAILY_BUDGET_USD))

def get_monthly_cost_limit() -> float:
    """Get monthly cost limit from environment or default."""
    return float(os.getenv('OPENAI_MONTHLY_BUDGET', MONTHLY_BUDGET_USD))

def is_llm_enabled() -> bool:
    """Check if LLM features are enabled."""
    return ENABLE_LLM_FEATURES and bool(OPENAI_API_KEY)

def get_rate_limits() -> tuple:
    """Get rate limiting configuration."""
    return (
        int(os.getenv('OPENAI_MAX_PER_MINUTE', MAX_REQUESTS_PER_MINUTE)),
        int(os.getenv('OPENAI_MAX_PER_HOUR', MAX_REQUESTS_PER_HOUR))
    )

# Environment variable overrides
if os.getenv('OPENAI_DAILY_BUDGET'):
    DAILY_BUDGET_USD = float(os.getenv('OPENAI_DAILY_BUDGET'))

if os.getenv('OPENAI_MONTHLY_BUDGET'):
    MONTHLY_BUDGET_USD = float(os.getenv('OPENAI_MONTHLY_BUDGET'))

if os.getenv('OPENAI_MODEL'):
    OPENAI_MODEL = os.getenv('OPENAI_MODEL')

if os.getenv('OPENAI_DISABLE_COST_TRACKING'):
    ENABLE_COST_TRACKING = False

if os.getenv('OPENAI_DISABLE_RATE_LIMITING'):
    ENABLE_RATE_LIMITING = False

if os.getenv('OPENAI_DISABLE_LLM'):
    ENABLE_LLM_FEATURES = False
