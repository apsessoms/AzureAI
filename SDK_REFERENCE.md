# Azure AI Services SDK Reference Guide

## Overview

This reference guide provides detailed documentation for Azure AI Services SDKs across multiple programming languages. Each service is documented with complete function signatures, parameters, return values, and usage examples.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Vision Services SDK](#vision-services-sdk)
3. [Text Analytics SDK](#text-analytics-sdk)
4. [Question Answering SDK](#question-answering-sdk)
5. [Machine Learning SDK](#machine-learning-sdk)
6. [Common Types & Models](#common-types--models)
7. [Error Classes](#error-classes)

---

## Installation & Setup

### Python SDK Installation

```bash
# Install all Azure AI services
pip install azure-ai-vision azure-ai-textanalytics azure-ai-language-questionanswering azure-cognitiveservices-vision-face

# Or install individually
pip install azure-ai-textanalytics
pip install azure-ai-vision
pip install azure-cognitiveservices-vision-face
pip install azure-ai-language-questionanswering
```

### JavaScript/TypeScript SDK Installation

```bash
npm install @azure/ai-text-analytics @azure/ai-vision @azure/cognitive-services-face @azure/ai-language-questionanswering
```

### C# SDK Installation

```xml
<PackageReference Include="Azure.AI.TextAnalytics" Version="5.3.0" />
<PackageReference Include="Azure.AI.Vision" Version="1.0.0" />
<PackageReference Include="Azure.CognitiveServices.Vision.Face" Version="2.8.0" />
<PackageReference Include="Azure.AI.Language.QuestionAnswering" Version="1.1.0" />
```

---

## Vision Services SDK

### ComputerVisionClient

#### Class: `ComputerVisionClient`

**Description**: Client for Azure Computer Vision API operations.

**Constructor**:
```python
from azure.ai.vision import ComputerVisionClient
from azure.core.credentials import AzureKeyCredential

client = ComputerVisionClient(
    endpoint: str,
    credential: AzureKeyCredential | DefaultAzureCredential
)
```

**Parameters**:
- `endpoint` (str): The endpoint URL for your Computer Vision resource
- `credential` (AzureKeyCredential): Authentication credential

---

#### Method: `analyze_image`

**Description**: Analyze visual content in an image.

**Signature**:
```python
def analyze_image(
    image_url: str = None,
    image_stream: BinaryIO = None,
    visual_features: List[VisualFeatureTypes] = None,
    details: List[Details] = None,
    language: str = "en"
) -> ImageAnalysis
```

**Parameters**:
- `image_url` (str, optional): URL of the image to analyze
- `image_stream` (BinaryIO, optional): Image data as binary stream
- `visual_features` (List[VisualFeatureTypes], optional): Features to analyze
- `details` (List[Details], optional): Additional details to extract
- `language` (str): Language for returned descriptions (default: "en")

**Returns**: `ImageAnalysis` object containing analysis results

**Visual Feature Types**:
- `VisualFeatureTypes.DESCRIPTION` - Image captions
- `VisualFeatureTypes.TAGS` - Content tags
- `VisualFeatureTypes.CATEGORIES` - Image categories
- `VisualFeatureTypes.OBJECTS` - Object detection
- `VisualFeatureTypes.BRANDS` - Brand detection
- `VisualFeatureTypes.FACES` - Face detection
- `VisualFeatureTypes.COLOR` - Color analysis

**Example**:
```python
from azure.ai.vision import ComputerVisionClient, VisualFeatureTypes
from azure.core.credentials import AzureKeyCredential

# Initialize client
client = ComputerVisionClient(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    credential=AzureKeyCredential("your-key")
)

# Analyze image
result = client.analyze_image(
    image_url="https://example.com/image.jpg",
    visual_features=[
        VisualFeatureTypes.DESCRIPTION,
        VisualFeatureTypes.OBJECTS,
        VisualFeatureTypes.TAGS
    ]
)

# Access results
print(f"Description: {result.description.captions[0].text}")
print(f"Objects found: {len(result.objects)}")
for obj in result.objects:
    print(f"- {obj.object_property} (confidence: {obj.confidence:.2f})")
```

**Raises**:
- `HttpResponseError`: When the request fails
- `ValueError`: When invalid parameters are provided

---

#### Method: `detect_objects`

**Description**: Detect objects in an image.

**Signature**:
```python
def detect_objects(
    image_url: str = None,
    image_stream: BinaryIO = None
) -> DetectResult
```

**Parameters**:
- `image_url` (str, optional): URL of the image
- `image_stream` (BinaryIO, optional): Image data as binary stream

**Returns**: `DetectResult` containing detected objects

**Example**:
```python
result = client.detect_objects(image_url="https://example.com/image.jpg")

for obj in result.objects:
    print(f"Object: {obj.object_property}")
    print(f"Confidence: {obj.confidence:.2f}")
    print(f"Bounding box: ({obj.rectangle.x}, {obj.rectangle.y}) "
          f"{obj.rectangle.w}x{obj.rectangle.h}")
```

---

### FaceClient

#### Class: `FaceClient`

**Description**: Client for Azure Face API operations.

**Constructor**:
```python
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

client = FaceClient(
    endpoint: str,
    credentials: CognitiveServicesCredentials
)
```

---

#### Method: `face.detect_with_url`

**Description**: Detect faces in an image from URL.

**Signature**:
```python
def detect_with_url(
    url: str,
    return_face_id: bool = True,
    return_face_landmarks: bool = False,
    return_face_attributes: List[FaceAttributeType] = None,
    recognition_model: str = "recognition_04",
    return_recognition_model: bool = False,
    detection_model: str = "detection_03"
) -> List[DetectedFace]
```

**Parameters**:
- `url` (str): URL of the image
- `return_face_id` (bool): Whether to return face ID
- `return_face_landmarks` (bool): Whether to return face landmarks
- `return_face_attributes` (List[FaceAttributeType]): Face attributes to return
- `recognition_model` (str): Recognition model version
- `detection_model` (str): Detection model version

**Face Attribute Types**:
- `FaceAttributeType.age`
- `FaceAttributeType.gender`
- `FaceAttributeType.emotion`
- `FaceAttributeType.glasses`
- `FaceAttributeType.facial_hair`
- `FaceAttributeType.makeup`
- `FaceAttributeType.accessories`
- `FaceAttributeType.head_pose`

**Returns**: List of `DetectedFace` objects

**Example**:
```python
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials

# Initialize client
credentials = CognitiveServicesCredentials("your-key")
client = FaceClient("https://your-resource.cognitiveservices.azure.com", credentials)

# Detect faces with attributes
faces = client.face.detect_with_url(
    url="https://example.com/face-image.jpg",
    return_face_landmarks=True,
    return_face_attributes=[
        FaceAttributeType.age,
        FaceAttributeType.gender,
        FaceAttributeType.emotion
    ]
)

for face in faces:
    print(f"Face ID: {face.face_id}")
    if face.face_attributes:
        print(f"Age: {face.face_attributes.age}")
        print(f"Gender: {face.face_attributes.gender}")
        print(f"Primary emotion: {max(face.face_attributes.emotion.__dict__.items(), key=lambda x: x[1])[0]}")
    
    if face.face_landmarks:
        print(f"Pupil left: ({face.face_landmarks.pupil_left.x}, {face.face_landmarks.pupil_left.y})")
        print(f"Pupil right: ({face.face_landmarks.pupil_right.x}, {face.face_landmarks.pupil_right.y})")
```

---

## Text Analytics SDK

### TextAnalyticsClient

#### Class: `TextAnalyticsClient`

**Description**: Client for Azure Text Analytics operations.

**Constructor**:
```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    endpoint: str,
    credential: AzureKeyCredential | DefaultAzureCredential
)
```

---

#### Method: `analyze_sentiment`

**Description**: Analyze sentiment of text documents.

**Signature**:
```python
def analyze_sentiment(
    documents: Union[List[str], List[TextDocumentInput], List[Dict[str, str]]],
    show_opinion_mining: bool = False,
    language: str = None,
    model_version: str = None,
    show_stats: bool = False
) -> List[AnalyzeSentimentResult]
```

**Parameters**:
- `documents` (List): Text documents to analyze
- `show_opinion_mining` (bool): Enable opinion mining
- `language` (str): Language of the documents
- `model_version` (str): Version of the model to use
- `show_stats` (bool): Include document statistics

**Returns**: List of `AnalyzeSentimentResult` objects

**Example**:
```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    credential=AzureKeyCredential("your-key")
)

documents = [
    "I love this product! It's amazing.",
    "This is terrible. I hate it.",
    "It's okay, nothing special."
]

results = client.analyze_sentiment(documents, show_opinion_mining=True)

for idx, result in enumerate(results):
    print(f"Document {idx + 1}:")
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence scores: {result.confidence_scores}")
    
    if result.sentences:
        for sentence in result.sentences:
            print(f"  Sentence: '{sentence.text}'")
            print(f"  Sentiment: {sentence.sentiment}")
            print(f"  Confidence: {sentence.confidence_scores}")
```

---

#### Method: `recognize_entities`

**Description**: Recognize named entities in text documents.

**Signature**:
```python
def recognize_entities(
    documents: Union[List[str], List[TextDocumentInput], List[Dict[str, str]]],
    language: str = None,
    model_version: str = None,
    show_stats: bool = False,
    categories_filter: List[str] = None
) -> List[RecognizeEntitiesResult]
```

**Parameters**:
- `documents` (List): Text documents to analyze
- `language` (str): Language of the documents
- `model_version` (str): Version of the model to use
- `show_stats` (bool): Include document statistics
- `categories_filter` (List[str]): Entity categories to include

**Entity Categories**:
- `"Person"` - People's names
- `"Location"` - Geographic locations
- `"Organization"` - Companies, institutions
- `"DateTime"` - Dates and times
- `"Quantity"` - Numbers and measurements
- `"Event"` - Historical or cultural events
- `"Product"` - Objects, products
- `"Skill"` - Abilities or expertise

**Returns**: List of `RecognizeEntitiesResult` objects

**Example**:
```python
documents = [
    "Microsoft was founded by Bill Gates and Paul Allen in 1975 in Albuquerque, New Mexico."
]

results = client.recognize_entities(documents)

for result in results:
    for entity in result.entities:
        print(f"Entity: {entity.text}")
        print(f"Category: {entity.category}")
        print(f"Confidence: {entity.confidence_score:.2f}")
        print(f"Offset: {entity.offset}, Length: {entity.length}")
        if entity.subcategory:
            print(f"Subcategory: {entity.subcategory}")
        print("---")
```

---

#### Method: `extract_key_phrases`

**Description**: Extract key phrases from text documents.

**Signature**:
```python
def extract_key_phrases(
    documents: Union[List[str], List[TextDocumentInput], List[Dict[str, str]]],
    language: str = None,
    model_version: str = None,
    show_stats: bool = False
) -> List[ExtractKeyPhrasesResult]
```

**Parameters**:
- `documents` (List): Text documents to analyze
- `language` (str): Language of the documents
- `model_version` (str): Version of the model to use
- `show_stats` (bool): Include document statistics

**Returns**: List of `ExtractKeyPhrasesResult` objects

**Example**:
```python
documents = [
    "The concert was amazing. The band played all their hit songs and the crowd was energetic."
]

results = client.extract_key_phrases(documents)

for result in results:
    print(f"Key phrases: {', '.join(result.key_phrases)}")
```

---

#### Method: `begin_analyze_actions`

**Description**: Start a long-running operation to analyze text with multiple actions.

**Signature**:
```python
def begin_analyze_actions(
    documents: Union[List[str], List[TextDocumentInput], List[Dict[str, str]]],
    actions: List[Union[RecognizeEntitiesAction, ExtractKeyPhrasesAction, AnalyzeSentimentAction, RecognizePiiEntitiesAction, ExtractSummaryAction]],
    language: str = None,
    show_stats: bool = False,
    polling_interval: int = 30
) -> LROPoller[ItemPaged[AnalyzeActionsResult]]
```

**Action Types**:
- `RecognizeEntitiesAction` - Named entity recognition
- `ExtractKeyPhrasesAction` - Key phrase extraction
- `AnalyzeSentimentAction` - Sentiment analysis
- `RecognizePiiEntitiesAction` - PII entity recognition
- `ExtractSummaryAction` - Text summarization

**Example**:
```python
from azure.ai.textanalytics import (
    RecognizeEntitiesAction,
    ExtractKeyPhrasesAction,
    AnalyzeSentimentAction,
    ExtractSummaryAction
)

documents = [
    "Microsoft Corporation is an American multinational technology company. "
    "It develops, manufactures, licenses, supports, and sells computer software, "
    "consumer electronics, personal computers, and related services."
]

actions = [
    RecognizeEntitiesAction(),
    ExtractKeyPhrasesAction(),
    AnalyzeSentimentAction(),
    ExtractSummaryAction(max_sentence_count=2)
]

poller = client.begin_analyze_actions(documents, actions)
results = poller.result()

for result_collection in results:
    for result in result_collection:
        if hasattr(result, 'entities'):  # Entity recognition
            print("Entities:")
            for entity in result.entities:
                print(f"  {entity.text} ({entity.category})")
        
        elif hasattr(result, 'key_phrases'):  # Key phrases
            print(f"Key phrases: {', '.join(result.key_phrases)}")
        
        elif hasattr(result, 'sentiment'):  # Sentiment
            print(f"Sentiment: {result.sentiment}")
        
        elif hasattr(result, 'sentences'):  # Summary
            print("Summary:")
            for sentence in result.sentences:
                print(f"  {sentence.text}")
```

---

## Question Answering SDK

### QuestionAnsweringClient

#### Class: `QuestionAnsweringClient`

**Description**: Client for Azure Question Answering operations.

**Constructor**:
```python
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential

client = QuestionAnsweringClient(
    endpoint: str,
    credential: AzureKeyCredential
)
```

---

#### Method: `get_answers`

**Description**: Get answers from a knowledge base.

**Signature**:
```python
def get_answers(
    question: str,
    project_name: str,
    deployment_name: str = "production",
    top: int = 3,
    confidence_threshold: float = None,
    context: KnowledgeBaseAnswerContext = None,
    ranker_type: str = None,
    filters: QueryFilters = None,
    answer_span_request: AnswerSpanRequest = None,
    include_unstructured_sources: bool = True
) -> AnswersResult
```

**Parameters**:
- `question` (str): User question
- `project_name` (str): Name of the QnA project
- `deployment_name` (str): Deployment name (default: "production")
- `top` (int): Number of answers to return
- `confidence_threshold` (float): Minimum confidence score
- `context` (KnowledgeBaseAnswerContext): Context for follow-up questions
- `filters` (QueryFilters): Metadata filters
- `include_unstructured_sources` (bool): Include unstructured sources

**Returns**: `AnswersResult` containing answers

**Example**:
```python
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential

client = QuestionAnsweringClient(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    credential=AzureKeyCredential("your-key")
)

result = client.get_answers(
    question="How can I cancel my reservation?",
    project_name="travel-faq",
    deployment_name="production",
    top=3,
    confidence_threshold=0.5
)

for answer in result.answers:
    print(f"Answer: {answer.answer}")
    print(f"Confidence: {answer.confidence}")
    print(f"Source: {answer.source}")
    print("---")
```

---

## Machine Learning SDK

### MLClient

#### Class: `MLClient`

**Description**: Client for Azure Machine Learning operations.

**Constructor**:
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

client = MLClient(
    credential: DefaultAzureCredential,
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str
)
```

---

#### Method: `jobs.create_or_update`

**Description**: Create or update a machine learning job.

**Signature**:
```python
def create_or_update(
    job: Union[Job, dict],
    description: str = None,
    compute: str = None,
    **kwargs
) -> Job
```

**Example - AutoML Classification**:
```python
from azure.ai.ml import MLClient, automl
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace"
)

# Create AutoML job
automl_job = automl.classification(
    experiment_name="bike-rental-classification",
    compute="cpu-cluster",
    training_data=client.data.get("bike-rentals", version="1"),
    target_column_name="rentals",
    primary_metric="accuracy",
    enable_model_explainability=True,
    max_trials=10,
    timeout_minutes=60
)

# Submit job
job = client.jobs.create_or_update(automl_job)
print(f"Job created: {job.name}")
```

---

#### Method: `online_endpoints.begin_create_or_update`

**Description**: Create or update an online endpoint for model deployment.

**Example**:
```python
from azure.ai.ml.entities import OnlineEndpoint, ManagedOnlineDeployment, Model, Environment

# Create endpoint
endpoint = OnlineEndpoint(
    name="bike-rental-endpoint",
    description="Endpoint for bike rental predictions",
    auth_mode="key"
)

# Create endpoint
endpoint_result = client.online_endpoints.begin_create_or_update(endpoint).result()

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="bike-rental-endpoint",
    model=client.models.get("bike-rental-model", version="1"),
    instance_type="Standard_DS2_v2",
    instance_count=1
)

# Deploy model
deployment_result = client.online_deployments.begin_create_or_update(deployment).result()
```

---

## Common Types & Models

### TextDocumentInput

**Description**: Input document for text analytics operations.

```python
class TextDocumentInput:
    def __init__(
        self,
        text: str,
        id: str,
        language: str = None
    ):
        self.text = text
        self.id = id
        self.language = language
```

### ImageAnalysis

**Description**: Result of image analysis operation.

```python
class ImageAnalysis:
    description: ImageDescription
    tags: List[ImageTag]
    objects: List[DetectedObject]
    categories: List[Category]
    faces: List[FaceDescription]
    color: ColorInfo
    image_type: ImageType
    metadata: ImageMetadata
```

### DetectedFace

**Description**: Detected face information.

```python
class DetectedFace:
    face_id: str
    face_rectangle: FaceRectangle
    face_landmarks: FaceLandmarks
    face_attributes: FaceAttributes
```

### AnalyzeSentimentResult

**Description**: Sentiment analysis result.

```python
class AnalyzeSentimentResult:
    id: str
    sentiment: str  # "positive", "negative", "neutral"
    confidence_scores: SentimentConfidenceScores
    sentences: List[SentenceSentiment]
    warnings: List[TextAnalyticsWarning]
    statistics: TextDocumentStatistics
```

---

## Error Classes

### HttpResponseError

**Description**: HTTP response error from Azure services.

```python
class HttpResponseError(Exception):
    status_code: int
    message: str
    error_code: str
    inner_error: dict
```

**Common Status Codes**:
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `429`: Too Many Requests
- `500`: Internal Server Error

### TextAnalyticsError

**Description**: Text Analytics specific error.

```python
class TextAnalyticsError(Exception):
    code: str
    message: str
    target: str
```

**Common Error Codes**:
- `InvalidArgument`: Invalid input parameter
- `InvalidDocument`: Document format error
- `UnsupportedLanguageCode`: Language not supported
- `InvalidDocumentBatch`: Batch processing error

---

## Best Practices & Usage Patterns

### 1. Async Operations

```python
import asyncio
from azure.ai.textanalytics.aio import TextAnalyticsClient

async def analyze_documents_async():
    async with TextAnalyticsClient(endpoint, credential) as client:
        documents = ["Text 1", "Text 2", "Text 3"]
        results = await client.analyze_sentiment(documents)
        return results

# Run async operation
results = asyncio.run(analyze_documents_async())
```

### 2. Batch Processing

```python
def process_large_dataset(documents, batch_size=10):
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            batch_results = client.analyze_sentiment(batch)
            results.extend(batch_results)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
    return results
```

### 3. Error Handling with Retry

```python
import time
import random
from azure.core.exceptions import HttpResponseError

def analyze_with_retry(client, documents, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.analyze_sentiment(documents)
        except HttpResponseError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                # Rate limited - wait with exponential backoff
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise e
```

### 4. Confidence Score Filtering

```python
def filter_high_confidence_results(results, min_confidence=0.8):
    filtered_results = []
    for result in results:
        if result.confidence_scores.positive >= min_confidence:
            filtered_results.append(result)
    return filtered_results
```

---

This SDK reference provides comprehensive documentation for all major Azure AI services. Each method includes detailed parameter descriptions, return types, and practical examples to help developers integrate these services effectively into their applications.