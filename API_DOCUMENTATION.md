# Azure AI Services - Comprehensive API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Azure AI Vision Services](#azure-ai-vision-services)
4. [Azure AI Language Services](#azure-ai-language-services)
5. [Azure Machine Learning Services](#azure-machine-learning-services)
6. [Common Patterns & Best Practices](#common-patterns--best-practices)
7. [Error Handling](#error-handling)
8. [Resource Management](#resource-management)
9. [API Reference](#api-reference)

## Overview

This documentation provides comprehensive guidance for using Azure AI Services. The services covered include:

- **Vision AI**: Image analysis, object detection, face detection, and image captioning
- **Language Services**: Text analytics, sentiment analysis, named entity recognition, question answering
- **Machine Learning**: Automated ML model training and deployment
- **Speech Services**: Speech-to-text and text-to-speech (planned)

### Architecture

```
Azure AI Services
├── Vision Services
│   ├── Image Analysis
│   ├── Object Detection
│   └── Face Detection
├── Language Services
│   ├── Text Analytics
│   ├── Question Answering
│   └── Named Entity Recognition
└── Machine Learning
    ├── Automated ML
    └── Model Deployment
```

## Getting Started

### Prerequisites

- Azure subscription
- Azure CLI or Azure Portal access
- Appropriate permissions to create resources

### Quick Start

1. **Create Azure AI Services Resource**
   ```bash
   az cognitiveservices account create \
     --name "my-ai-services" \
     --resource-group "my-resource-group" \
     --kind "CognitiveServices" \
     --sku "S0" \
     --location "eastus"
   ```

2. **Get Access Keys**
   ```bash
   az cognitiveservices account keys list \
     --name "my-ai-services" \
     --resource-group "my-resource-group"
   ```

### Authentication

All Azure AI Services use key-based authentication or Azure Active Directory.

**Using Access Keys:**
```http
Authorization: Ocp-Apim-Subscription-Key {your-key}
Content-Type: application/json
```

## Azure AI Vision Services

### Image Analysis API

Analyze images to extract information about visual content.

#### Endpoints

| Service | Endpoint | Method |
|---------|----------|--------|
| Image Captioning | `{endpoint}/vision/v3.2/analyze` | POST |
| Object Detection | `{endpoint}/vision/v3.2/detect` | POST |
| Tag Extraction | `{endpoint}/vision/v3.2/tag` | POST |

#### Image Captioning

**Description**: Generate human-readable descriptions of images.

**Request:**
```http
POST {endpoint}/vision/v3.2/analyze?visualFeatures=Description
Content-Type: application/json
Ocp-Apim-Subscription-Key: {key}

{
  "url": "https://example.com/image.jpg"
}
```

**Response:**
```json
{
  "description": {
    "tags": ["person", "outdoor", "sport"],
    "captions": [
      {
        "text": "a person playing tennis on a court",
        "confidence": 0.8456
      }
    ]
  },
  "requestId": "abc123-def456",
  "metadata": {
    "width": 1024,
    "height": 768,
    "format": "Jpeg"
  }
}
```

**Usage Example:**
```python
import requests

endpoint = "https://your-resource.cognitiveservices.azure.com"
key = "your-subscription-key"

url = f"{endpoint}/vision/v3.2/analyze"
headers = {
    "Ocp-Apim-Subscription-Key": key,
    "Content-Type": "application/json"
}
params = {"visualFeatures": "Description"}
data = {"url": "https://example.com/image.jpg"}

response = requests.post(url, headers=headers, params=params, json=data)
result = response.json()
caption = result["description"]["captions"][0]["text"]
print(f"Image description: {caption}")
```

#### Dense Captioning

**Description**: Generate multiple captions for different regions of an image.

**Features:**
- Multiple human-readable captions
- Bounding box coordinates
- Object detection within regions
- Confidence scores for each caption

**Response Structure:**
```json
{
  "denseCaptions": [
    {
      "text": "a person holding a tennis racket",
      "confidence": 0.89,
      "boundingBox": {
        "x": 100,
        "y": 50,
        "w": 200,
        "h": 300
      }
    }
  ]
}
```

#### Object Detection

**Description**: Detect and locate objects within images.

**Request:**
```http
POST {endpoint}/vision/v3.2/detect
Content-Type: application/json
Ocp-Apim-Subscription-Key: {key}

{
  "url": "https://example.com/image.jpg"
}
```

**Response:**
```json
{
  "objects": [
    {
      "rectangle": {
        "x": 185,
        "y": 63,
        "w": 211,
        "h": 284
      },
      "object": "person",
      "confidence": 0.924
    }
  ]
}
```

### Face Detection API

**Description**: Detect human faces in images and return face rectangles and attributes.

#### Endpoints

| Service | Endpoint | Method |
|---------|----------|--------|
| Face Detection | `{endpoint}/face/v1.0/detect` | POST |

#### Basic Face Detection

**Request:**
```http
POST {endpoint}/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true
Content-Type: application/json
Ocp-Apim-Subscription-Key: {key}

{
  "url": "https://example.com/face-image.jpg"
}
```

**Response:**
```json
[
  {
    "faceId": "abc123-def456-ghi789",
    "faceRectangle": {
      "top": 63,
      "left": 185,
      "width": 211,
      "height": 284
    },
    "faceLandmarks": {
      "pupilLeft": {
        "x": 231.5,
        "y": 191.4
      },
      "pupilRight": {
        "x": 329.9,
        "y": 176.5
      },
      "noseTip": {
        "x": 279.9,
        "y": 232.2
      },
      "mouthLeft": {
        "x": 240.7,
        "y": 270.7
      },
      "mouthRight": {
        "x": 347.0,
        "y": 254.2
      }
    }
  }
]
```

#### Face Attributes

**Available Attributes:**
- Age estimation
- Gender detection
- Emotion analysis
- Facial hair detection
- Glasses detection
- Head pose
- Makeup detection
- Accessories detection

**Request with Attributes:**
```http
POST {endpoint}/face/v1.0/detect?returnFaceAttributes=age,gender,emotion,glasses
```

## Azure AI Language Services

### Text Analytics API

Analyze text to extract insights such as sentiment, key phrases, and named entities.

#### Endpoints

| Service | Endpoint | Method |
|---------|----------|--------|
| Sentiment Analysis | `{endpoint}/text/analytics/v3.1/sentiment` | POST |
| Key Phrase Extraction | `{endpoint}/text/analytics/v3.1/keyPhrases` | POST |
| Named Entity Recognition | `{endpoint}/text/analytics/v3.1/entities/recognition/general` | POST |
| Text Summarization | `{endpoint}/text/analytics/v3.1/analyze` | POST |

#### Sentiment Analysis

**Description**: Determine the sentiment (positive, negative, neutral) of text.

**Request:**
```http
POST {endpoint}/text/analytics/v3.1/sentiment
Content-Type: application/json
Ocp-Apim-Subscription-Key: {key}

{
  "documents": [
    {
      "id": "1",
      "language": "en",
      "text": "I love this product! It's amazing."
    }
  ]
}
```

**Response:**
```json
{
  "documents": [
    {
      "id": "1",
      "sentiment": "positive",
      "confidenceScores": {
        "positive": 0.99,
        "neutral": 0.01,
        "negative": 0.0
      },
      "sentences": [
        {
          "sentiment": "positive",
          "confidenceScores": {
            "positive": 0.99,
            "neutral": 0.01,
            "negative": 0.0
          },
          "offset": 0,
          "length": 35,
          "text": "I love this product! It's amazing."
        }
      ]
    }
  ]
}
```

#### Named Entity Recognition (NER)

**Description**: Extract and classify named entities from text.

**Entity Categories:**
- Person
- Location
- Organization
- DateTime
- Quantity
- PersonType
- Event
- Product
- Skill

**Request:**
```http
POST {endpoint}/text/analytics/v3.1/entities/recognition/general
Content-Type: application/json
Ocp-Apim-Subscription-Key: {key}

{
  "documents": [
    {
      "id": "1",
      "language": "en",
      "text": "Microsoft was founded by Bill Gates and Paul Allen in 1975 in Albuquerque."
    }
  ]
}
```

**Response:**
```json
{
  "documents": [
    {
      "id": "1",
      "entities": [
        {
          "text": "Microsoft",
          "category": "Organization",
          "confidenceScore": 0.99,
          "offset": 0,
          "length": 9
        },
        {
          "text": "Bill Gates",
          "category": "Person",
          "confidenceScore": 0.99,
          "offset": 25,
          "length": 10
        },
        {
          "text": "1975",
          "category": "DateTime",
          "subcategory": "DateRange",
          "confidenceScore": 0.8,
          "offset": 52,
          "length": 4
        }
      ]
    }
  ]
}
```

#### Key Phrase Extraction

**Description**: Extract key phrases that capture the main topics in text.

**Request:**
```http
POST {endpoint}/text/analytics/v3.1/keyPhrases
Content-Type: application/json
Ocp-Apim-Subscription-Key: {key}

{
  "documents": [
    {
      "id": "1",
      "language": "en",
      "text": "The concert was amazing. The band played all their hit songs and the crowd was energetic."
    }
  ]
}
```

**Response:**
```json
{
  "documents": [
    {
      "id": "1",
      "keyPhrases": [
        "concert",
        "band",
        "hit songs",
        "crowd"
      ]
    }
  ]
}
```

#### Text Summarization

**Description**: Generate summaries of long text documents.

**Types:**
- **Extractive**: Selects key sentences from the original text
- **Abstractive**: Generates new summary text

**Request:**
```http
POST {endpoint}/text/analytics/v3.1/analyze
Content-Type: application/json
Ocp-Apim-Subscription-Key: {key}

{
  "analysisInput": {
    "documents": [
      {
        "id": "1",
        "language": "en",
        "text": "Long text to be summarized..."
      }
    ]
  },
  "tasks": {
    "extractiveSummarizationTasks": [
      {
        "parameters": {
          "sentenceCount": 3
        }
      }
    ]
  }
}
```

### Question Answering Service

**Description**: Create knowledge bases that can answer questions based on provided content.

#### Creating a Knowledge Base

**Steps:**
1. Create Language resource with Custom Question Answering
2. Create project in Language Studio
3. Add data sources (URLs, files, manual Q&A pairs)
4. Train the knowledge base
5. Test and deploy

#### API Usage

**Request:**
```http
POST {endpoint}/language/:query-knowledgebases?projectName={project}&deploymentName={deployment}
Content-Type: application/json
Ocp-Apim-Subscription-Key: {key}

{
  "top": 3,
  "question": "How can I cancel my reservation?",
  "includeUnstructuredSources": true,
  "confidenceScoreThreshold": 0.5
}
```

**Response:**
```json
{
  "answers": [
    {
      "questions": [
        "How can I cancel my reservation?"
      ],
      "answer": "You can cancel your reservation by calling our customer service or using the online portal.",
      "confidenceScore": 0.92,
      "id": 1,
      "source": "FAQ.docx",
      "metadata": {}
    }
  ]
}
```

## Azure Machine Learning Services

### Automated ML API

**Description**: Automatically build and train machine learning models.

#### Creating an AutoML Job

**Configuration Example:**
```json
{
  "experiment_name": "bike-rental-prediction",
  "task": "regression",
  "primary_metric": "normalized_root_mean_squared_error",
  "target_column_name": "rentals",
  "training_data": {
    "name": "bike-rentals",
    "version": "1"
  },
  "compute_target": "cpu-cluster",
  "training_parameters": {
    "experiment_timeout_minutes": 60,
    "max_trials": 10,
    "max_concurrent_trials": 4,
    "enable_early_stopping": true
  }
}
```

#### Model Deployment

**Real-time Endpoint Configuration:**
```json
{
  "name": "bike-rental-endpoint",
  "description": "Endpoint for bike rental predictions",
  "compute": {
    "instance_type": "Standard_DS2_v2",
    "instance_count": 1
  },
  "environment": {
    "name": "automl-environment",
    "version": "1"
  }
}
```

#### Making Predictions

**Request:**
```http
POST {endpoint}/score
Content-Type: application/json
Authorization: Bearer {token}

{
  "input_data": {
    "columns": [
      "day", "mnth", "year", "season", "holiday", 
      "weekday", "workingday", "weathersit", 
      "temp", "atemp", "hum", "windspeed"
    ],
    "index": [0],
    "data": [[1, 1, 2022, 2, 0, 1, 1, 2, 0.3, 0.3, 0.3, 0.3]]
  }
}
```

**Response:**
```json
{
  "predictions": [334.67]
}
```

## Common Patterns & Best Practices

### Authentication Patterns

#### Using Azure SDK for Python
```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Initialize client
credential = AzureKeyCredential("your-key")
client = TextAnalyticsClient(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    credential=credential
)

# Use the client
documents = ["I love this product!"]
result = client.analyze_sentiment(documents)
```

#### Using Azure Identity (Recommended)
```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = TextAnalyticsClient(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    credential=credential
)
```

### Error Handling Patterns

```python
from azure.core.exceptions import HttpResponseError

try:
    result = client.analyze_sentiment(documents)
except HttpResponseError as e:
    if e.status_code == 429:
        print("Rate limit exceeded. Retry after some time.")
    elif e.status_code == 401:
        print("Authentication failed. Check your credentials.")
    else:
        print(f"Request failed: {e}")
```

### Batch Processing

```python
# Process multiple documents efficiently
documents = [
    {"id": "1", "language": "en", "text": "First document"},
    {"id": "2", "language": "en", "text": "Second document"},
    {"id": "3", "language": "en", "text": "Third document"}
]

# Analyze in batches of 10 (API limit)
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    result = client.analyze_sentiment(batch)
    for doc in result:
        print(f"Document {doc.id}: {doc.sentiment}")
```

### Async Processing

```python
import asyncio
from azure.ai.textanalytics.aio import TextAnalyticsClient

async def analyze_documents_async():
    async with TextAnalyticsClient(endpoint, credential) as client:
        result = await client.analyze_sentiment(documents)
        return result

# Run async operation
result = asyncio.run(analyze_documents_async())
```

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Action |
|------|---------|---------|
| 200 | Success | Process response |
| 400 | Bad Request | Check request format |
| 401 | Unauthorized | Verify credentials |
| 403 | Forbidden | Check permissions |
| 429 | Rate Limited | Implement retry logic |
| 500 | Server Error | Retry request |

### Retry Logic Pattern

```python
import time
import random
from azure.core.exceptions import HttpResponseError

def make_request_with_retry(client, operation, max_retries=3):
    for attempt in range(max_retries):
        try:
            return operation(client)
        except HttpResponseError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise e
```

## Resource Management

### Cost Optimization

1. **Choose Appropriate Pricing Tiers**
   - Free tier: For development and testing
   - Standard tier: For production workloads
   - Pay-as-you-go: For variable workloads

2. **Monitor Usage**
   ```bash
   # Check current usage
   az cognitiveservices account get-usage \
     --name "my-ai-services" \
     --resource-group "my-resource-group"
   ```

3. **Set Up Alerts**
   - Configure billing alerts
   - Monitor API call quotas
   - Track response times

### Security Best Practices

1. **Use Managed Identity**
   ```python
   from azure.identity import ManagedIdentityCredential
   
   credential = ManagedIdentityCredential()
   client = TextAnalyticsClient(endpoint, credential)
   ```

2. **Rotate Keys Regularly**
   ```bash
   # Regenerate keys
   az cognitiveservices account keys regenerate \
     --name "my-ai-services" \
     --resource-group "my-resource-group" \
     --key-name "Key1"
   ```

3. **Use Private Endpoints**
   - Restrict network access
   - Use VNet integration
   - Enable firewall rules

### Monitoring and Diagnostics

#### Enable Diagnostic Logging
```bash
az monitor diagnostic-settings create \
  --resource "my-ai-services" \
  --resource-group "my-resource-group" \
  --name "ai-diagnostics" \
  --logs '[{"category":"Audit","enabled":true}]' \
  --metrics '[{"category":"AllMetrics","enabled":true}]' \
  --storage-account "mystorageaccount"
```

#### Key Metrics to Monitor
- Request count
- Response time
- Error rate
- Token usage
- Quota consumption

## API Reference

### Rate Limits

| Service | Tier | Requests per Second | Requests per Month |
|---------|------|--------------------|--------------------|
| Text Analytics | Free | 5 | 5,000 |
| Text Analytics | Standard | 1,000 | Unlimited |
| Vision | Free | 20 | 5,000 |
| Vision | Standard | 10,000 | Unlimited |
| Face | Free | 20 | 30,000 |
| Face | Standard | 10,000 | Unlimited |

### Regional Availability

| Service | Available Regions |
|---------|------------------|
| Text Analytics | East US, West Europe, Southeast Asia, Australia East |
| Vision | Global |
| Face | East US, West Europe, Southeast Asia |
| Speech | Global |

### SDK Support

| Language | Text Analytics | Vision | Face | Speech |
|----------|---------------|---------|------|--------|
| Python | ✓ | ✓ | ✓ | ✓ |
| C# | ✓ | ✓ | ✓ | ✓ |
| Java | ✓ | ✓ | ✓ | ✓ |
| JavaScript | ✓ | ✓ | ✓ | ✓ |
| Go | ✓ | ✓ | ✓ | ✓ |

### Sample Applications

#### Sentiment Analysis Dashboard
```python
# Complete example for sentiment analysis
import streamlit as st
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def create_sentiment_app():
    st.title("Sentiment Analysis Dashboard")
    
    # Input
    text = st.text_area("Enter text to analyze:")
    
    if st.button("Analyze"):
        if text:
            # Initialize client
            client = TextAnalyticsClient(endpoint, credential)
            
            # Analyze sentiment
            result = client.analyze_sentiment([text])[0]
            
            # Display results
            st.metric("Sentiment", result.sentiment)
            st.metric("Confidence", f"{result.confidence_scores.positive:.2f}")
            
            # Visualize
            scores = {
                "Positive": result.confidence_scores.positive,
                "Neutral": result.confidence_scores.neutral,
                "Negative": result.confidence_scores.negative
            }
            st.bar_chart(scores)

if __name__ == "__main__":
    create_sentiment_app()
```

#### Image Analysis Tool
```python
# Complete example for image analysis
import requests
import json
from PIL import Image
import matplotlib.pyplot as plt

class ImageAnalyzer:
    def __init__(self, endpoint, key):
        self.endpoint = endpoint
        self.key = key
        self.headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/json"
        }
    
    def analyze_image(self, image_url):
        """Analyze image and return comprehensive results"""
        analyze_url = f"{self.endpoint}/vision/v3.2/analyze"
        params = {
            "visualFeatures": "Categories,Description,Color,Objects,Tags",
            "details": "Landmarks,Celebrities"
        }
        
        data = {"url": image_url}
        response = requests.post(
            analyze_url, 
            headers=self.headers, 
            params=params, 
            json=data
        )
        
        return response.json()
    
    def detect_objects(self, image_url):
        """Detect objects in image"""
        detect_url = f"{self.endpoint}/vision/v3.2/detect"
        data = {"url": image_url}
        
        response = requests.post(
            detect_url,
            headers=self.headers,
            json=data
        )
        
        return response.json()
    
    def visualize_results(self, image_url, results):
        """Visualize analysis results"""
        # Load and display image
        img = Image.open(requests.get(image_url, stream=True).raw)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        
        # Add object bounding boxes
        if "objects" in results:
            for obj in results["objects"]:
                rect = obj["rectangle"]
                plt.gca().add_patch(plt.Rectangle(
                    (rect["x"], rect["y"]),
                    rect["w"], rect["h"],
                    fill=False, color="red", linewidth=2
                ))
                plt.text(
                    rect["x"], rect["y"] - 10,
                    f"{obj['object']} ({obj['confidence']:.2f})",
                    color="red", fontsize=12
                )
        
        plt.title(f"Analysis: {results['description']['captions'][0]['text']}")
        plt.axis("off")
        plt.show()

# Usage example
analyzer = ImageAnalyzer(endpoint, key)
results = analyzer.analyze_image("https://example.com/image.jpg")
analyzer.visualize_results("https://example.com/image.jpg", results)
```

---

## Contributing

To contribute to this documentation:

1. Fork the repository
2. Create a feature branch
3. Update documentation
4. Submit a pull request

## License

This documentation is licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:
- Azure AI Services Documentation: https://docs.microsoft.com/azure/cognitive-services/
- Azure Support: https://azure.microsoft.com/support/
- Community Forums: https://docs.microsoft.com/answers/topics/azure-cognitive-services.html