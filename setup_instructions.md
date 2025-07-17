# Tuthand Prototype Setup Instructions

## 1. Set Your OpenAI API Key

You need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

**Or create a `.env` file** (recommended):
```bash
cp .env.example .env
# Edit .env file and add your API key
```

## 2. Run the Prototype

```bash
# Activate virtual environment
source tuthand-env/bin/activate

# Run the prototype
python3 main.py
```

## 3. Test Queries

Try these example queries to test different strategies:

### Plain Strategy (Simple questions)
- "What is Tuthand?"
- "Who created this?"
- "When was it launched?"

### Chain-of-Thought (Complex reasoning)
- "How does Tuthand compare to ChatGPT?"
- "Why should I use this over other solutions?"
- "Explain the benefits of AI assistants"

### ReAct Strategy (Implementation/technical)
- "How do I implement the API?"
- "Help me integrate Tuthand with my website"
- "Build a custom agent"

### Reflection Strategy (Validation/review)
- "Review my implementation"
- "Check if this is correct"
- "Validate my approach"

### Escalation Strategy (Sensitive/personal)
- "Can you access my account?"
- "Show me my personal data"
- "What's my password?"

## 4. Commands

- `user:developer` - Switch to developer mode
- `user:customer` - Switch to customer mode
- `user:founder` - Switch to founder mode
- `user:support` - Switch to support mode
- `stats` - View session statistics
- `quit` or `exit` - Exit the prototype

## 5. What to Observe

- **Strategy Selection**: Notice which strategy is chosen for different queries
- **Confidence Scores**: How confident the system is in its responses
- **Trust Levels**: auto_run, confirm, or escalate
- **Response Time**: How quickly responses are generated
- **Response Quality**: How well the AI follows the prompt strategies

The prototype demonstrates Week 2's Interface Intelligence + Performance Optimization in action!