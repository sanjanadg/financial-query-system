#!/bin/bash
# Setup script for OpenAI API key limits and configuration

echo "🔧 Setting up OpenAI API configuration..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# OpenAI API Configuration
# Set your OpenAI API key here
OPENAI_API_KEY=your_api_key_here

# Cost Control Limits (in USD)
OPENAI_DAILY_BUDGET=5.0
OPENAI_MONTHLY_BUDGET=50.0

# Rate Limiting
OPENAI_MAX_PER_MINUTE=10
OPENAI_MAX_PER_HOUR=100

# Model Selection (choose one)
OPENAI_MODEL=gpt-4o-mini

# Feature Toggles
# OPENAI_DISABLE_COST_TRACKING=true
# OPENAI_DISABLE_RATE_LIMITING=true
# OPENAI_DISABLE_LLM=true
EOF
    echo "✅ Created .env file"
else
    echo "📝 .env file already exists"
fi

echo ""
echo "📋 Configuration Options:"
echo "  💰 Daily Budget: $5.00 (default)"
echo "  💰 Monthly Budget: $50.00 (default)"
echo "  🔒 Rate Limit: 10 requests/minute, 100/hour"
echo "  🤖 Model: gpt-4o-mini (cheapest option)"
echo ""
echo "📝 To customize:"
echo "  1. Edit the .env file"
echo "  2. Or set environment variables directly:"
echo "     export OPENAI_DAILY_BUDGET=2.0"
echo "     export OPENAI_MONTHLY_BUDGET=25.0"
echo ""
echo "⚠️  IMPORTANT: Replace 'your_api_key_here' with your actual OpenAI API key!"
echo ""
echo "🚀 To run with these settings:"
echo "   source .env && python main2.py"
echo "   # or"
echo "   export \$(cat .env | xargs) && python main2.py"
