# FinGPT Deployment Guide

This guide explains how to deploy FinGPT for use with TradeBuddy, including local and cloud deployment options.

## Current Status

TradeBuddy's FinGPT integration includes:
- ✅ **Smart Mock Responses**: Context-aware mock responses for development
- ✅ **Real API Integration**: Ready for local or cloud FinGPT deployments
- ✅ **Automatic Fallback**: Falls back to mock responses if API is unavailable
- ✅ **Configuration Support**: Flexible endpoint and authentication options

## Quick Start (Development)

The system currently uses enhanced mock responses that analyze your market data to provide relevant signals:

```bash
# Select FinGPT in CLI
# System will automatically use mock responses with smart signal generation
```

## Local FinGPT Deployment

### Option 1: Docker Container (Recommended)

1. **Create Local FinGPT Server**:
```bash
# Create directory for FinGPT server
mkdir fingpt-server && cd fingpt-server

# Create simple FastAPI server
cat > server.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Local FinGPT Server")

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False

@app.post("/generate")
async def generate(request: GenerateRequest):
    # Placeholder for actual FinGPT model inference
    # In production, this would load and run the actual FinGPT model
    return {
        "response": "Local FinGPT model response would go here",
        "model": request.model,
        "tokens_used": 150
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "fingpt:v3.2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .
EXPOSE 8000

CMD ["python", "server.py"]
EOF

# Create requirements
cat > requirements.txt << 'EOF'
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.4.0
transformers>=4.30.0
torch>=2.0.0
EOF

# Build and run
docker build -t fingpt-server .
docker run -p 8000:8000 fingpt-server
```

2. **Configure TradeBuddy**:
```bash
# Set environment variables
export FINGPT_API_ENDPOINT="http://localhost:8000/api/fingpt"
export FINGPT_MODEL_VARIANT="v3.2"

# Or configure in code
```

### Option 2: Hugging Face Inference Endpoints

1. **Deploy to Hugging Face**:
   - Go to [Hugging Face Inference Endpoints](https://endpoints.huggingface.co/)
   - Select FinGPT model (e.g., `FinGPT/fingpt-forecaster_dow30_llama2-7b_lora`)
   - Deploy as dedicated endpoint

2. **Configure TradeBuddy**:
```python
# In your session configuration
ai_config = AIModelConfig(
    model_type=AIModelType.FINGPT,
    fingpt_api_endpoint="https://your-endpoint.endpoints.huggingface.cloud",
    fingpt_api_key="your-hf-token",
    fingpt_model_variant="v3.2"
)
```

## Configuration Options

### Environment Variables
```bash
# FinGPT API Configuration
export FINGPT_API_ENDPOINT="http://localhost:8000/api/fingpt"
export FINGPT_API_KEY="your-api-key"  # Optional
export FINGPT_MODEL_VARIANT="v3.2"    # v3.1, v3.2, or v3.3
export FINGPT_TIMEOUT="30"             # Request timeout in seconds
```

### Code Configuration
```python
from src.core.models import AIModelConfig, AIModelType

# Local deployment
ai_config = AIModelConfig(
    model_type=AIModelType.FINGPT,
    fingpt_api_endpoint="http://localhost:8000/api/fingpt",
    fingpt_model_variant="v3.2",
    fingpt_timeout=30,
    fallback_enabled=True
)

# Cloud deployment
ai_config = AIModelConfig(
    model_type=AIModelType.FINGPT,
    fingpt_api_endpoint="https://api.openai.com/v1",  # Or your cloud endpoint
    fingpt_api_key="your-api-key",
    fingpt_model_variant="v3.3",
    fingpt_timeout=60,
    fallback_enabled=True
)
```

## API Specification

The FinGPT client expects the following API format:

### POST /generate
```json
{
  "model": "fingpt:v3.2",
  "prompt": "Your financial analysis prompt...",
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false
}
```

### Response
```json
{
  "response": "Generated financial analysis text...",
  "model": "fingpt:v3.2",
  "tokens_used": 150
}
```

## Error Handling

The system includes robust error handling:

1. **API Connection Failure**: Falls back to smart mock responses
2. **Timeout**: Configurable timeout with fallback
3. **Authentication Error**: Clear error messages with fallback options
4. **Rate Limiting**: Automatic retry with exponential backoff (planned)

## Cost Considerations

### Local Deployment
- **Setup Cost**: ~$0 (uses your hardware)
- **Running Cost**: Electricity + hardware depreciation
- **Pros**: Full control, no API limits, privacy
- **Cons**: Requires GPU, setup complexity

### Cloud Deployment
- **Setup Cost**: ~$0
- **Running Cost**: $0.01-0.10 per request (varies by provider)
- **Pros**: Managed infrastructure, scalability
- **Cons**: Ongoing costs, API limits

### Hybrid Approach (Recommended)
- **Development**: Local deployment or smart mocks
- **Production**: Cloud deployment with local fallback
- **Cost**: Optimized based on usage patterns

## Next Steps

1. **Immediate**: Use enhanced mock responses for development
2. **Short-term**: Set up local Docker container for testing
3. **Production**: Deploy to Hugging Face Inference Endpoints
4. **Advanced**: Implement model ensemble and comparative analysis

## Troubleshooting

### Common Issues

1. **"Connection refused"**: Check if FinGPT server is running on localhost:8000
2. **"Authentication failed"**: Verify API key configuration
3. **"Timeout"**: Increase timeout or check network connectivity
4. **"Model not found"**: Verify model variant and endpoint URL

### Debug Mode
```python
import logging
logging.getLogger("src.analysis.ai_models.fingpt_client").setLevel(logging.DEBUG)
```

## Support

For issues with FinGPT deployment:
1. Check TradeBuddy logs for detailed error messages
2. Verify API endpoint accessibility: `curl http://localhost:8000/health`
3. Review configuration in `AIModelConfig`
4. Use fallback mode for development: `fallback_enabled=True`