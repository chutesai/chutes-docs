# Your First Custom Chute

This guide walks you through building your first completely custom chute from scratch. Unlike templates, you'll learn to build every component yourself, giving you full control and understanding of the platform.

## What We'll Build

We'll create a **sentiment analysis API** that:

- 🧠 **Loads a custom model** (DistilBERT for sentiment analysis)
- 🔍 **Validates inputs** with Pydantic schemas
- 🌐 **Provides REST endpoints** for single and batch processing
- 📊 **Returns structured results** with confidence scores
- 🏗️ **Uses custom Docker image** with optimized dependencies

## Prerequisites

Make sure you've completed:

- ✅ [Installation & Setup](installation)
- ✅ [Quick Start Guide](quickstart) (recommended)
- ✅ [Authentication](authentication)

## Step 1: Plan Your Chute

Before coding, let's plan what we need:

### API Endpoints

- `POST /analyze` - Analyze single text
- `POST /batch` - Analyze multiple texts
- `GET /health` - Health check

### Input/Output

- **Input**: Text string or array of strings
- **Output**: Sentiment label (POSITIVE/NEGATIVE/NEUTRAL) + confidence

### Resources

- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **GPU**: 1x GPU with 8GB VRAM
- **Dependencies**: PyTorch, Transformers, FastAPI, Pydantic

## Step 2: Create Project Structure

Create a new directory for your project:

```bash
mkdir my-first-chute
cd my-first-chute
```

Create the main chute file:

```bash
touch sentiment_chute.py
```

## Step 3: Define Input/Output Schemas

Start by defining your data models with Pydantic:

```python
# sentiment_chute.py
from pydantic import BaseModel, Field, validator
from typing import List
from enum import Enum

class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=50, description="List of texts to analyze")

    @validator('texts')
    def validate_texts(cls, v):
        cleaned_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 5000:
                raise ValueError(f'Text at index {i} is too long (max 5000 characters)')
            cleaned_texts.append(text.strip())
        return cleaned_texts

class SentimentResult(BaseModel):
    text: str
    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float

class BatchSentimentResult(BaseModel):
    results: List[SentimentResult]
    total_texts: int
    total_processing_time: float
    average_confidence: float
```

## Step 4: Build Custom Docker Image

Define a custom Docker image with all necessary dependencies:

```python
# Add to sentiment_chute.py
from chutes.image import Image

# Create optimized image for sentiment analysis
image = (
    Image(username="myuser", name="sentiment-chute", tag="1.0")

    # Start with CUDA-enabled Ubuntu
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")

    # Install Python 3.11
    .with_python("3.11")

    # Install system dependencies
    .run_command("""
        apt-get update && apt-get install -y \\
        git curl wget \\
        && rm -rf /var/lib/apt/lists/*
    """)

    # Install PyTorch with CUDA support
    .run_command("""
        pip install torch torchvision torchaudio \\
        --index-url https://download.pytorch.org/whl/cu121
    """)

    # Install transformers and other ML dependencies
    .run_command("""
        pip install \\
        transformers>=4.30.0 \\
        accelerate>=0.20.0 \\
        tokenizers>=0.13.0 \\
        numpy>=1.24.0 \\
        scikit-learn>=1.3.0
    """)

    # Set up model cache directory
    .with_env("TRANSFORMERS_CACHE", "/app/models")
    .with_env("HF_HOME", "/app/models")
    .run_command("mkdir -p /app/models")

    # Set working directory
    .set_workdir("/app")
)
```

## Step 5: Create the Chute

Now create the main chute with proper initialization:

```python
# Add to sentiment_chute.py
from chutes.chute import Chute, NodeSelector
from fastapi import HTTPException
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Define the chute
chute = Chute(
    username="myuser",  # Replace with your username
    name="sentiment-chute",
    image=image,
    tagline="Advanced sentiment analysis with confidence scoring",
    readme="""
# Sentiment Analysis Chute

A production-ready sentiment analysis service using RoBERTa.

## Features
- High-accuracy sentiment classification
- Confidence scoring for each prediction
- Batch processing support
- GPU acceleration
- Input validation and error handling

## Usage

### Single Text Analysis
```bash
curl -X POST https://myuser-sentiment-chute.chutes.ai/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"text": "I love this new AI service!"}'
```

### Batch Analysis

```bash
curl -X POST https://myuser-sentiment-chute.chutes.ai/batch \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": [
      "This is amazing!",
      "Not very good...",
      "It works okay I guess"
    ]
  }'
```

## Response Format

```json
{
  "text": "I love this new AI service!",
  "sentiment": "POSITIVE",
  "confidence": 0.9847,
  "processing_time": 0.045
}
```
    """,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8,
        include=["rtx4090", "rtx3090", "a100"]  # Prefer these GPUs
    ),
    concurrency=4  # Handle up to 4 concurrent requests

)
```

## Step 6: Add Model Loading

Implement the startup function to load your model:

```python
# Add to sentiment_chute.py
@chute.on_startup()
async def load_model(self):
    """Load the sentiment analysis model and tokenizer."""
    print("🚀 Starting sentiment analysis chute...")

    # Model configuration
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    print(f"📥 Loading model: {model_name}")

    try:
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded successfully")

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("✅ Model loaded successfully")

        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")

        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Label mapping (specific to this model)
        self.label_mapping = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }

        # Warm up the model with a dummy input
        print("🔥 Warming up model...")
        dummy_text = "This is a test."
        await self._predict_sentiment(dummy_text)

        print("✅ Model loaded and ready!")

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e

async def _predict_sentiment(self, text: str) -> tuple[str, float, float]:
    """
    Internal method to predict sentiment.
    Returns: (sentiment_label, confidence, processing_time)
    """
    start_time = time.time()

    try:
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get predicted class and confidence
        predicted_class_id = predictions.argmax().item()
        confidence = predictions[0][predicted_class_id].item()

        # Map to human-readable label
        model_label = self.model.config.id2label[predicted_class_id]
        sentiment_label = self.label_mapping.get(model_label, model_label)

        processing_time = time.time() - start_time

        return sentiment_label, confidence, processing_time

    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment prediction failed: {str(e)}"
        )
```

## Step 7: Implement API Endpoints

Add your API endpoints using the `@chute.cord` decorator:

```python
# Add to sentiment_chute.py
@chute.cord(
    public_api_path="/analyze",
    method="POST",
    input_schema=TextInput,
    output_content_type="application/json"
)
async def analyze_sentiment(self, data: TextInput) -> SentimentResult:
    """Analyze sentiment of a single text."""

    sentiment, confidence, processing_time = await self._predict_sentiment(data.text)

    return SentimentResult(
        text=data.text,
        sentiment=SentimentLabel(sentiment),
        confidence=confidence,
        processing_time=processing_time
    )

@chute.cord(
    public_api_path="/batch",
    method="POST",
    input_schema=BatchTextInput,
    output_content_type="application/json"
)
async def analyze_batch(self, data: BatchTextInput) -> BatchSentimentResult:
    """Analyze sentiment of multiple texts."""

    start_time = time.time()
    results = []
    confidences = []

    for text in data.texts:
        sentiment, confidence, proc_time = await self._predict_sentiment(text)

        results.append(SentimentResult(
            text=text,
            sentiment=SentimentLabel(sentiment),
            confidence=confidence,
            processing_time=proc_time
        ))

        confidences.append(confidence)

    total_processing_time = time.time() - start_time
    average_confidence = np.mean(confidences) if confidences else 0.0

    return BatchSentimentResult(
        results=results,
        total_texts=len(data.texts),
        total_processing_time=total_processing_time,
        average_confidence=average_confidence
    )

@chute.cord(
    public_api_path="/health",
    method="GET",
    output_content_type="application/json"
)
async def health_check(self) -> dict:
    """Health check endpoint."""

    model_loaded = hasattr(self, 'model') and hasattr(self, 'tokenizer')

    # Quick performance test
    if model_loaded:
        try:
            _, _, test_time = await self._predict_sentiment("Test message")
            performance_ok = test_time < 1.0  # Should be under 1 second
        except:
            performance_ok = False
    else:
        performance_ok = False

    return {
        "status": "healthy" if model_loaded and performance_ok else "unhealthy",
        "model_loaded": model_loaded,
        "device": getattr(self, 'device', 'unknown'),
        "performance_ok": performance_ok,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None
    }
```

## Step 8: Add Local Testing

Add a local testing function to verify everything works:

```python
# Add to sentiment_chute.py
if __name__ == "__main__":
    import asyncio

    async def test_locally():
        """Test the chute locally before deploying."""
        print("🧪 Testing chute locally...")

        # Simulate the startup process
        await load_model(chute)

        # Test single analysis
        print("\n📝 Testing single text analysis...")
        test_input = TextInput(text="I absolutely love this new technology!")
        result = await analyze_sentiment(chute, test_input)
        print(f"Input: {result.text}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Processing time: {result.processing_time:.4f}s")

        # Test batch analysis
        print("\n📝 Testing batch analysis...")
        batch_input = BatchTextInput(texts=[
            "This is amazing!",
            "I hate this so much.",
            "It's okay, nothing special.",
            "Absolutely fantastic experience!"
        ])
        batch_result = await analyze_batch(chute, batch_input)
        print(f"Processed {batch_result.total_texts} texts")
        print(f"Average confidence: {batch_result.average_confidence:.4f}")
        print(f"Total time: {batch_result.total_processing_time:.4f}s")

        for i, res in enumerate(batch_result.results):
            print(f"  {i+1}. '{res.text}' -> {res.sentiment} ({res.confidence:.3f})")

        # Test health check
        print("\n🏥 Testing health check...")
        health = await health_check(chute)
        print(f"Status: {health['status']}")
        print(f"Device: {health['device']}")

        print("\n✅ All tests passed! Ready to deploy.")

    # Run local tests
    asyncio.run(test_locally())
```

## Step 9: Complete File

(Refer to the full file structure in Step 8)

## Step 10: Test Locally

Before deploying, test your chute locally:

```bash
python sentiment_chute.py
```

## Step 11: Build and Deploy

### Build the Image

```bash
chutes build sentiment_chute:chute --wait
```

This will:

- 📦 Create your custom Docker image
- 🔧 Install all dependencies
- ⬇️ Download the model
- ✅ Validate the configuration

### Deploy the Chute

```bash
chutes deploy sentiment_chute:chute
```

After successful deployment:

```
✅ Chute deployed successfully!
🌐 Public API: https://myuser-sentiment-chute.chutes.ai
📋 Chute ID: 12345678-1234-5678-9abc-123456789012
```

## Step 12: Test Your Live API

Test your deployed chute:

### Single Text Analysis

```bash
curl -X POST https://myuser-sentiment-chute.chutes.ai/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this new AI service!"}'
```

### Batch Analysis

```bash
curl -X POST https://myuser-sentiment-chute.chutes.ai/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This is amazing technology!",
      "I hate waiting in long lines.",
      "The weather is okay today."
    ]
  }'
```

### Health Check

```bash
curl https://myuser-sentiment-chute.chutes.ai/health
```

## Next Steps

Now that you understand the fundamentals, explore more advanced topics:

### Immediate Next Steps

- **[Streaming Responses](../examples/streaming-responses)** - Add real-time processing
- **[Batch Processing](../examples/batch-processing)** - Optimize for high throughput
- **[Input/Output Schemas](../guides/schemas)** - Advanced validation patterns

### Advanced Topics

- **[Custom Images Guide](../guides/custom-images)** - Advanced Docker configurations
- **[Performance Optimization](../guides/performance)** - Speed up your chutes
- **[Error Handling](../guides/error-handling)** - Robust error management
- **[Best Practices](../guides/best-practices)** - Production deployment patterns

## Troubleshooting

**Build fails with dependency errors?**

- Check Python package versions
- Ensure CUDA compatibility
- Verify base image availability

**Model loading takes too long?**

- Model downloads on first run (normal)
- Consider pre-downloading in Docker image
- Check internet connection during build

**GPU not detected?**

- Verify CUDA installation in image
- Check NodeSelector GPU requirements
- Ensure PyTorch CUDA support

## Getting Help

- 📖 **Documentation**: Continue with advanced guides
- 💬 **Discord**: [Join our community](https://discord.gg/wHrXwWkCRz)
- 🐛 **Issues**: [GitHub Issues](https://github.com/rayonlabs/chutes/issues)
- 📧 **Support**: `support@chutes.ai`

---

🎉 **Congratulations!** You've built your first custom chute from scratch. You now have the foundation to create any AI application you can imagine with Chutes!
