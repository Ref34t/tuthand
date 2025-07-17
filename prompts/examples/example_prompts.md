# Tuthand Example Prompts and Test Cases

## User Type Classifications

### 1. **Founder/Creator** - Building or researching AI assistants
### 2. **Customer** - Potential buyer evaluating the product
### 3. **Developer** - Technical implementation and integration
### 4. **Support** - Help with existing implementation

---

## Example Queries by User Type

### Founder/Creator Examples

#### Query 1: Product Positioning
**User**: "How does Tuthand compare to other AI assistant solutions?"

**Strategy**: Chain-of-Thought (Complex comparison)
**Expected Response**:
```
[CONFIDENCE: 82%] [TRUST_LEVEL: confirm]

Let me think through this step by step:

1. First, I need to understand: You're comparing Tuthand to other AI assistant solutions for websites
2. Based on the available information: Tuthand is built as a production-ready, multi-agent system following AI Engineering principles
3. Considering your specific context: As a founder, you likely care about scalability, customization, and enterprise features

Key differentiators:
- **Architecture**: Multi-agent system vs single-agent competitors
- **Trust Model**: Built-in confidence scoring and routing (auto-run/confirm/escalate)
- **Performance**: <2s response time, 99.9% uptime, <$0.05 per interaction
- **Deployment**: Production-ready with Kubernetes, monitoring, auto-scaling

Would you like me to dive deeper into any specific comparison area?

Source: README.md, architecture documentation
```

#### Query 2: Implementation Timeline
**User**: "How long does it take to implement Tuthand on a website?"

**Strategy**: Plain Response (Clear FAQ)
**Expected Response**:
```
[CONFIDENCE: 95%] [TRUST_LEVEL: auto_run]

Tuthand can be implemented in under 30 minutes for basic setup:
- 5 minutes: Add embed script to your website
- 10 minutes: Configure content sources and trust levels
- 15 minutes: Test and customize UI themes

For enterprise deployment with custom agents and integrations, plan 1-2 weeks.

Source: docs/user_guide/getting_started.md
```

### Customer Examples

#### Query 3: Pricing Inquiry
**User**: "What does Tuthand cost per month?"

**Strategy**: Chain-of-Thought (Context-dependent pricing)
**Expected Response**:
```
[CONFIDENCE: 78%] [TRUST_LEVEL: confirm]

Let me think through this step by step:

1. First, I need to understand: You're asking about monthly pricing for Tuthand
2. Based on the available information: Pricing depends on usage volume, features, and deployment type
3. Considering your specific context: Without knowing your traffic and feature needs, I can provide general guidance

General pricing structure:
- **Starter**: $99/month (up to 1,000 interactions)
- **Professional**: $299/month (up to 10,000 interactions)
- **Enterprise**: Custom pricing (unlimited interactions + dedicated support)

Each interaction costs <$0.05 on average, but varies by complexity.

Would you like me to help you estimate costs based on your expected usage?

Source: pricing documentation
```

#### Query 4: Integration Complexity
**User**: "Do I need technical skills to set up Tuthand?"

**Strategy**: Plain Response (Clear capability question)
**Expected Response**:
```
[CONFIDENCE: 96%] [TRUST_LEVEL: auto_run]

No technical skills required for basic setup. Tuthand provides:
- Simple embed script (copy-paste into your website)
- Visual configuration dashboard
- Pre-built templates for common use cases
- Drag-and-drop UI customization

For advanced features (custom agents, API integrations), some technical knowledge helps but isn't required.

Source: docs/user_guide/getting_started.md
```

### Developer Examples

#### Query 5: API Documentation
**User**: "Where can I find the REST API documentation?"

**Strategy**: Plain Response (Direct resource location)
**Expected Response**:
```
[CONFIDENCE: 98%] [TRUST_LEVEL: auto_run]

API documentation is available at:
- **Main docs**: /docs/developer_guide/api_reference.md
- **Interactive playground**: Available in your dashboard under "API Explorer"
- **SDK libraries**: Python, JavaScript, and Go clients available

Key endpoints:
- POST /api/v1/chat - Send messages
- GET /api/v1/context - Retrieve conversation context
- POST /api/v1/agents - Create custom agents

Source: docs/developer_guide/api_reference.md
```

#### Query 6: Custom Agent Development
**User**: "How do I create a custom agent for my specific domain?"

**Strategy**: ReAct (Multi-step technical process)
**Expected Response**:
```
[CONFIDENCE: 85%] [TRUST_LEVEL: confirm]

**Thought**: This is a technical implementation question about custom agent development.

**Action**: Let me gather the relevant information about agent creation process.

**Observation**: Custom agents can be created using the agent framework with specialized prompts, tools, and knowledge bases.

**Thought**: I should provide a structured approach for domain-specific agent creation.

**Answer**: To create a custom agent for your domain:

1. **Define Agent Role**: Specify expertise area and boundaries
2. **Configure Knowledge Base**: Upload domain-specific content
3. **Set Tools**: Connect relevant APIs and data sources
4. **Train Prompts**: Create domain-specific prompt templates
5. **Test & Iterate**: Use the testing framework to validate responses

The agent framework supports specialization in areas like:
- Sales and lead qualification
- Technical support and troubleshooting
- Product guidance and recommendations

Does this address your question, or would you like me to explore any aspect further?

Sources: docs/developer_guide/extending_agents.md, agent examples
```

### Support Examples

#### Query 7: Performance Issues
**User**: "My Tuthand assistant is responding slowly. How can I fix this?"

**Strategy**: Reflection (Diagnostic process)
**Expected Response**:
```
[CONFIDENCE: 88%] [TRUST_LEVEL: confirm]

**Initial Response**: Slow responses can be caused by several factors including token usage, context size, or network issues.

**Reflection**: Let me double-check this answer:
- Is this information accurate? Yes, these are common performance bottlenecks
- Are there any missing considerations? Should also check caching and model selection
- Could this be interpreted differently? User might need specific debugging steps

**Refined Answer**: To diagnose slow response times:

1. **Check Token Usage**: Review your prompt length and context size
2. **Optimize Caching**: Enable response caching for frequent queries
3. **Review Model Selection**: Consider using faster models for simple queries
4. **Monitor Network**: Check latency to API endpoints

Performance targets:
- Target: <2 seconds average response time
- Alert threshold: >5 seconds indicates issues

Use the performance dashboard to identify specific bottlenecks.

I believe this is accurate, but would you like me to verify any specific details?

Sources: docs/user_guide/troubleshooting.md, performance optimization guide
```

#### Query 8: Sensitive Data Handling
**User**: "A user is asking for their personal account information. How should Tuthand handle this?"

**Strategy**: Escalation (Sensitive data request)
**Expected Response**:
```
[CONFIDENCE: 45%] [TRUST_LEVEL: escalate]

I want to make sure you get the best possible help with this sensitive data question.

**What I understand**: A user is requesting personal account information through the Tuthand assistant.

**Why I'm escalating**: Personal data requests require human verification and proper security protocols that I cannot provide.

Let me connect you with our security team who can provide guidance on:
- Proper data handling procedures
- User verification requirements
- Compliance with privacy regulations (GDPR, CCPA)

Context for handoff: User data request requiring security review and proper authentication
```

---

## Edge Cases and Error Handling

### Ambiguous Query
**User**: "How much does it cost?"

**Strategy**: Plain Response (Clarification request)
**Expected Response**:
```
[CONFIDENCE: 65%] [TRUST_LEVEL: confirm]

Could you clarify what you're looking for? Are you asking about:
- Monthly subscription pricing for Tuthand
- Per-interaction costs
- Implementation and setup fees
- Enterprise licensing options

I'd be happy to provide specific pricing information once I understand your needs better.

Source: pricing documentation
```

### Out-of-Scope Query
**User**: "What's the weather like today?"

**Strategy**: Escalation (Out of domain)
**Expected Response**:
```
[CONFIDENCE: 25%] [TRUST_LEVEL: escalate]

I'm designed to help with questions about Tuthand and AI assistant implementation. For weather information, I'd recommend checking a weather service like Weather.com or your local weather app.

Is there anything about Tuthand I can help you with instead?

Context for handoff: User query outside of Tuthand domain knowledge
```

### Technical Error
**User**: "I'm getting a 500 error when trying to embed the script"

**Strategy**: Escalation (Technical issue requiring support)
**Expected Response**:
```
[CONFIDENCE: 55%] [TRUST_LEVEL: escalate]

I want to make sure you get the best possible help with this technical issue.

**What I understand**: You're encountering a 500 error when embedding the Tuthand script.

**Why I'm escalating**: Server errors require technical diagnosis and may indicate infrastructure issues.

Let me connect you with our technical support team who can:
- Check server status and logs
- Verify your embed code configuration
- Provide immediate technical assistance

Context for handoff: 500 error during script embedding, requires technical investigation
```

---

## Performance Benchmarks

### Response Time Targets
- **Plain Response**: <1 second
- **Chain-of-Thought**: <2 seconds
- **ReAct**: <3 seconds
- **Reflection**: <3 seconds
- **Escalation**: <1 second

### Token Usage Targets
- **Plain Response**: 50-150 tokens
- **Chain-of-Thought**: 150-300 tokens
- **ReAct**: 200-400 tokens
- **Reflection**: 250-400 tokens
- **Escalation**: 75-125 tokens

### Confidence Accuracy
- **Target**: 95% correlation with user satisfaction
- **Measurement**: Confidence score vs user feedback
- **Adjustment**: Threshold tuning based on real-world performance

---

## Testing Validation

### Response Consistency
- Same query should produce similar responses (±5% confidence)
- Trust level should remain consistent for identical contexts
- Source references should be accurate and accessible

### Trust Level Accuracy
- AUTO_RUN responses should have >95% user satisfaction
- CONFIRM responses should have >85% user satisfaction after confirmation
- ESCALATE responses should result in successful human handoff

### Edge Case Handling
- Ambiguous queries should request clarification
- Out-of-scope queries should gracefully redirect
- Technical errors should escalate with appropriate context

### Performance Validation
- Response times should meet target benchmarks
- Token usage should stay within defined budgets
- Confidence scores should correlate with actual accuracy