# Azure AI Services - Quick Start Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [5-Minute Quick Start](#5-minute-quick-start)
3. [Vision Services Tutorial](#vision-services-tutorial)
4. [Text Analytics Tutorial](#text-analytics-tutorial)
5. [Question Answering Tutorial](#question-answering-tutorial)
6. [Machine Learning Tutorial](#machine-learning-tutorial)
7. [Common Use Cases](#common-use-cases)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Azure Account Setup
- Azure subscription (free tier available)
- Azure CLI installed (optional but recommended)

### 2. Environment Setup

**Python Environment:**
```bash
# Create virtual environment
python -m venv azure-ai-env
source azure-ai-env/bin/activate  # On Windows: azure-ai-env\Scripts\activate

# Install Azure AI packages
pip install azure-ai-textanalytics azure-ai-vision azure-cognitiveservices-vision-face
pip install azure-ai-language-questionanswering azure-ai-ml
pip install azure-identity python-dotenv
```

**Environment Variables:**
Create a `.env` file:
```bash
AZURE_COGNITIVE_SERVICES_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_COGNITIVE_SERVICES_KEY=your-subscription-key
AZURE_LANGUAGE_ENDPOINT=https://your-language-resource.cognitiveservices.azure.com
AZURE_LANGUAGE_KEY=your-language-key
```

---

## 5-Minute Quick Start

### Step 1: Create Azure Resources

**Using Azure CLI:**
```bash
# Login to Azure
az login

# Create resource group
az group create --name azure-ai-quickstart --location eastus

# Create Cognitive Services resource
az cognitiveservices account create \
  --name my-ai-services \
  --resource-group azure-ai-quickstart \
  --kind CognitiveServices \
  --sku S0 \
  --location eastus

# Get the endpoint and key
az cognitiveservices account show \
  --name my-ai-services \
  --resource-group azure-ai-quickstart \
  --query "properties.endpoint" --output tsv

az cognitiveservices account keys list \
  --name my-ai-services \
  --resource-group azure-ai-quickstart
```

### Step 2: Test Connection

**Python Script (test_connection.py):**
```python
import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

def test_text_analytics():
    # Initialize client
    endpoint = os.getenv("AZURE_COGNITIVE_SERVICES_ENDPOINT")
    key = os.getenv("AZURE_COGNITIVE_SERVICES_KEY")
    
    client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )
    
    # Test sentiment analysis
    documents = ["I love Azure AI services! They're amazing."]
    
    try:
        results = client.analyze_sentiment(documents)
        for result in results:
            print(f"Sentiment: {result.sentiment}")
            print(f"Confidence: {result.confidence_scores}")
        print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_text_analytics()
```

**Run the test:**
```bash
python test_connection.py
```

---

## Vision Services Tutorial

### Tutorial 1: Image Analysis Service

**Goal**: Analyze images to extract descriptions, objects, and tags.

**Step 1: Setup Vision Client**
```python
# vision_demo.py
import os
from dotenv import load_dotenv
from azure.ai.vision import VisionServiceOptions, VisionSource
from azure.ai.vision.imageanalysis import ImageAnalyzer, ImageAnalysisOptions
from azure.core.credentials import AzureKeyCredential

load_dotenv()

class ImageAnalysisDemo:
    def __init__(self):
        endpoint = os.getenv("AZURE_COGNITIVE_SERVICES_ENDPOINT")
        key = os.getenv("AZURE_COGNITIVE_SERVICES_KEY")
        
        # Create vision service options
        self.service_options = VisionServiceOptions(
            endpoint=endpoint,
            key=key
        )
    
    def analyze_image_from_url(self, image_url):
        """Analyze image from URL"""
        # Create vision source
        vision_source = VisionSource(url=image_url)
        
        # Set analysis options
        analysis_options = ImageAnalysisOptions()
        analysis_options.features = [
            "Caption",
            "DenseCaptions", 
            "Objects",
            "Tags",
            "Text"
        ]
        analysis_options.language = "en"
        
        # Create analyzer and analyze
        analyzer = ImageAnalyzer(self.service_options, vision_source, analysis_options)
        result = analyzer.analyze()
        
        return self.process_results(result)
    
    def process_results(self, result):
        """Process and display analysis results"""
        analysis = {}
        
        # Caption
        if result.caption:
            analysis['caption'] = {
                'text': result.caption.content,
                'confidence': result.caption.confidence
            }
        
        # Objects
        if result.objects:
            analysis['objects'] = []
            for obj in result.objects:
                analysis['objects'].append({
                    'name': obj.name,
                    'confidence': obj.confidence,
                    'bounding_box': {
                        'x': obj.bounding_box.x,
                        'y': obj.bounding_box.y,
                        'width': obj.bounding_box.w,
                        'height': obj.bounding_box.h
                    }
                })
        
        # Tags
        if result.tags:
            analysis['tags'] = [
                {'name': tag.name, 'confidence': tag.confidence}
                for tag in result.tags
            ]
        
        return analysis

# Usage example
if __name__ == "__main__":
    demo = ImageAnalysisDemo()
    
    # Test with sample image
    image_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"
    results = demo.analyze_image_from_url(image_url)
    
    print("🖼️ Image Analysis Results:")
    print(f"Caption: {results.get('caption', {}).get('text', 'N/A')}")
    print(f"Objects found: {len(results.get('objects', []))}")
    print(f"Tags found: {len(results.get('tags', []))}")
```

### Tutorial 2: Face Detection Service

**Goal**: Detect faces and analyze facial attributes.

```python
# face_detection_demo.py
import os
from dotenv import load_dotenv
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials

load_dotenv()

class FaceDetectionDemo:
    def __init__(self):
        endpoint = os.getenv("AZURE_COGNITIVE_SERVICES_ENDPOINT")
        key = os.getenv("AZURE_COGNITIVE_SERVICES_KEY")
        
        self.client = FaceClient(
            endpoint,
            CognitiveServicesCredentials(key)
        )
    
    def detect_faces(self, image_url):
        """Detect faces with attributes"""
        try:
            faces = self.client.face.detect_with_url(
                url=image_url,
                return_face_landmarks=True,
                return_face_attributes=[
                    FaceAttributeType.age,
                    FaceAttributeType.gender,
                    FaceAttributeType.emotion,
                    FaceAttributeType.glasses,
                    FaceAttributeType.facial_hair
                ]
            )
            
            return self.process_face_results(faces)
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def process_face_results(self, faces):
        """Process face detection results"""
        results = []
        
        for i, face in enumerate(faces):
            face_info = {
                'face_id': i + 1,
                'rectangle': {
                    'left': face.face_rectangle.left,
                    'top': face.face_rectangle.top,
                    'width': face.face_rectangle.width,
                    'height': face.face_rectangle.height
                }
            }
            
            if face.face_attributes:
                attrs = face.face_attributes
                face_info['attributes'] = {
                    'age': attrs.age,
                    'gender': attrs.gender.value if attrs.gender else None,
                    'glasses': attrs.glasses.value if attrs.glasses else None,
                    'primary_emotion': self.get_primary_emotion(attrs.emotion),
                    'facial_hair': {
                        'moustache': attrs.facial_hair.moustache if attrs.facial_hair else 0,
                        'beard': attrs.facial_hair.beard if attrs.facial_hair else 0,
                        'sideburns': attrs.facial_hair.sideburns if attrs.facial_hair else 0
                    } if attrs.facial_hair else None
                }
            
            if face.face_landmarks:
                landmarks = face.face_landmarks
                face_info['landmarks'] = {
                    'left_eye': {'x': landmarks.pupil_left.x, 'y': landmarks.pupil_left.y},
                    'right_eye': {'x': landmarks.pupil_right.x, 'y': landmarks.pupil_right.y},
                    'nose_tip': {'x': landmarks.nose_tip.x, 'y': landmarks.nose_tip.y}
                }
            
            results.append(face_info)
        
        return results
    
    def get_primary_emotion(self, emotion):
        """Get the primary emotion with highest confidence"""
        if not emotion:
            return None
        
        emotions = {
            'anger': emotion.anger,
            'contempt': emotion.contempt,
            'disgust': emotion.disgust,
            'fear': emotion.fear,
            'happiness': emotion.happiness,
            'neutral': emotion.neutral,
            'sadness': emotion.sadness,
            'surprise': emotion.surprise
        }
        
        return max(emotions.items(), key=lambda x: x[1])

# Usage example
if __name__ == "__main__":
    demo = FaceDetectionDemo()
    
    # Test with sample image
    image_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/detection1.jpg"
    results = demo.detect_faces(image_url)
    
    print("👥 Face Detection Results:")
    for face in results:
        print(f"Face {face['face_id']}:")
        if 'attributes' in face:
            attrs = face['attributes']
            print(f"  Age: {attrs.get('age', 'N/A')}")
            print(f"  Gender: {attrs.get('gender', 'N/A')}")
            print(f"  Primary emotion: {attrs.get('primary_emotion', ['N/A', 0])[0]} ({attrs.get('primary_emotion', ['N/A', 0])[1]:.2f})")
```

---

## Text Analytics Tutorial

### Tutorial 3: Complete Text Analysis Pipeline

**Goal**: Create a comprehensive text analysis pipeline.

```python
# text_analytics_demo.py
import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.ai.textanalytics import (
    RecognizeEntitiesAction,
    ExtractKeyPhrasesAction,
    AnalyzeSentimentAction
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

class TextAnalyticsDemo:
    def __init__(self):
        endpoint = os.getenv("AZURE_COGNITIVE_SERVICES_ENDPOINT")
        key = os.getenv("AZURE_COGNITIVE_SERVICES_KEY")
        
        self.client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
    
    def analyze_text_complete(self, texts):
        """Complete text analysis pipeline"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = {
            'sentiment': self.analyze_sentiment(texts),
            'entities': self.extract_entities(texts),
            'key_phrases': self.extract_key_phrases(texts),
            'language': self.detect_language(texts)
        }
        
        return results
    
    def analyze_sentiment(self, texts):
        """Analyze sentiment with opinion mining"""
        try:
            results = self.client.analyze_sentiment(
                texts, 
                show_opinion_mining=True
            )
            
            sentiment_results = []
            for result in results:
                sentiment_info = {
                    'overall_sentiment': result.sentiment,
                    'confidence_scores': {
                        'positive': result.confidence_scores.positive,
                        'neutral': result.confidence_scores.neutral,
                        'negative': result.confidence_scores.negative
                    },
                    'sentences': []
                }
                
                for sentence in result.sentences:
                    sentence_info = {
                        'text': sentence.text,
                        'sentiment': sentence.sentiment,
                        'confidence_scores': {
                            'positive': sentence.confidence_scores.positive,
                            'neutral': sentence.confidence_scores.neutral,
                            'negative': sentence.confidence_scores.negative
                        }
                    }
                    sentiment_info['sentences'].append(sentence_info)
                
                sentiment_results.append(sentiment_info)
            
            return sentiment_results
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return []
    
    def extract_entities(self, texts):
        """Extract named entities"""
        try:
            results = self.client.recognize_entities(texts)
            
            entity_results = []
            for result in results:
                entities = []
                for entity in result.entities:
                    entity_info = {
                        'text': entity.text,
                        'category': entity.category,
                        'subcategory': entity.subcategory,
                        'confidence': entity.confidence_score,
                        'offset': entity.offset,
                        'length': entity.length
                    }
                    entities.append(entity_info)
                entity_results.append(entities)
            
            return entity_results
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return []
    
    def extract_key_phrases(self, texts):
        """Extract key phrases"""
        try:
            results = self.client.extract_key_phrases(texts)
            return [result.key_phrases for result in results]
        except Exception as e:
            print(f"Error in key phrase extraction: {e}")
            return []
    
    def detect_language(self, texts):
        """Detect language"""
        try:
            results = self.client.detect_language(texts)
            language_results = []
            for result in results:
                primary_language = result.primary_language
                lang_info = {
                    'language': primary_language.name,
                    'iso6391_name': primary_language.iso6391_name,
                    'confidence': primary_language.confidence_score
                }
                language_results.append(lang_info)
            return language_results
        except Exception as e:
            print(f"Error in language detection: {e}")
            return []
    
    def batch_analyze_actions(self, texts):
        """Batch analysis with multiple actions"""
        actions = [
            RecognizeEntitiesAction(),
            ExtractKeyPhrasesAction(),
            AnalyzeSentimentAction(show_opinion_mining=True)
        ]
        
        try:
            poller = self.client.begin_analyze_actions(texts, actions)
            results = poller.result()
            
            combined_results = {}
            for result_collection in results:
                for i, result in enumerate(result_collection):
                    doc_id = f"document_{i}"
                    if doc_id not in combined_results:
                        combined_results[doc_id] = {}
                    
                    if hasattr(result, 'entities'):
                        combined_results[doc_id]['entities'] = [
                            {'text': e.text, 'category': e.category, 'confidence': e.confidence_score}
                            for e in result.entities
                        ]
                    elif hasattr(result, 'key_phrases'):
                        combined_results[doc_id]['key_phrases'] = result.key_phrases
                    elif hasattr(result, 'sentiment'):
                        combined_results[doc_id]['sentiment'] = {
                            'overall': result.sentiment,
                            'confidence': result.confidence_scores.__dict__
                        }
            
            return combined_results
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return {}

# Usage example
if __name__ == "__main__":
    demo = TextAnalyticsDemo()
    
    # Sample texts for analysis
    sample_texts = [
        "Microsoft Corporation is an American multinational technology company. "
        "I'm really excited about their new AI services!",
        
        "The weather today is terrible. It's been raining all day and I hate it. "
        "But at least I can work from home in Seattle.",
        
        "Azure AI services are revolutionizing how we process and understand data. "
        "The accuracy and speed of these machine learning models is impressive."
    ]
    
    print("📊 Complete Text Analysis Pipeline")
    print("=" * 50)
    
    # Individual analysis
    for i, text in enumerate(sample_texts):
        print(f"\n📝 Analyzing Text {i+1}:")
        print(f"Text: {text[:100]}...")
        
        results = demo.analyze_text_complete(text)
        
        # Display sentiment
        sentiment = results['sentiment'][0]
        print(f"💭 Sentiment: {sentiment['overall_sentiment']} "
              f"(confidence: {sentiment['confidence_scores'][sentiment['overall_sentiment']]:.2f})")
        
        # Display entities
        entities = results['entities'][0]
        if entities:
            print(f"🏷️  Entities: {', '.join([f\"{e['text']} ({e['category']})\" for e in entities[:3]])}")
        
        # Display key phrases
        key_phrases = results['key_phrases'][0]
        if key_phrases:
            print(f"🔑 Key Phrases: {', '.join(key_phrases[:5])}")
        
        # Display language
        language = results['language'][0]
        print(f"🌍 Language: {language['language']} (confidence: {language['confidence']:.2f})")
    
    # Batch analysis
    print(f"\n🔄 Batch Analysis Results:")
    batch_results = demo.batch_analyze_actions(sample_texts)
    for doc_id, analysis in batch_results.items():
        print(f"{doc_id}: {len(analysis.get('entities', []))} entities, "
              f"{len(analysis.get('key_phrases', []))} key phrases, "
              f"sentiment: {analysis.get('sentiment', {}).get('overall', 'N/A')}")
```

---

## Question Answering Tutorial

### Tutorial 4: Building a Knowledge Base

**Goal**: Create and query a custom knowledge base.

```python
# question_answering_demo.py
import os
from dotenv import load_dotenv
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

class QuestionAnsweringDemo:
    def __init__(self):
        endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        key = os.getenv("AZURE_LANGUAGE_KEY")
        
        self.client = QuestionAnsweringClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
    
    def ask_question(self, question, project_name, deployment_name="production"):
        """Ask a question to the knowledge base"""
        try:
            result = self.client.get_answers(
                question=question,
                project_name=project_name,
                deployment_name=deployment_name,
                top=3,
                confidence_threshold=0.5,
                include_unstructured_sources=True
            )
            
            return self.process_answers(result.answers)
        except Exception as e:
            print(f"Error getting answers: {e}")
            return []
    
    def process_answers(self, answers):
        """Process and format answers"""
        processed_answers = []
        
        for answer in answers:
            answer_info = {
                'answer': answer.answer,
                'confidence': answer.confidence,
                'source': answer.source,
                'metadata': answer.metadata,
                'questions': answer.questions
            }
            processed_answers.append(answer_info)
        
        return processed_answers
    
    def interactive_qa_session(self, project_name):
        """Interactive Q&A session"""
        print(f"🤖 Question Answering Session for '{project_name}'")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            answers = self.ask_question(question, project_name)
            
            if answers:
                print(f"\n💡 Top answer:")
                best_answer = answers[0]
                print(f"Answer: {best_answer['answer']}")
                print(f"Confidence: {best_answer['confidence']:.2f}")
                print(f"Source: {best_answer['source']}")
                
                if len(answers) > 1:
                    print(f"\n📋 Other answers:")
                    for i, answer in enumerate(answers[1:], 2):
                        print(f"{i}. {answer['answer'][:100]}... "
                              f"(confidence: {answer['confidence']:.2f})")
            else:
                print("❌ No answers found for your question.")

# Example FAQ data for testing
FAQ_EXAMPLES = {
    "travel": [
        "How can I cancel my reservation?",
        "What is the refund policy?",
        "How do I change my booking?",
        "What documents do I need for travel?",
        "Are pets allowed on flights?"
    ],
    "technical": [
        "How do I reset my password?",
        "What are the system requirements?",
        "How to troubleshoot connection issues?",
        "What payment methods are accepted?",
        "How to contact customer support?"
    ]
}

# Usage example
if __name__ == "__main__":
    demo = QuestionAnsweringDemo()
    
    # Test with sample questions (requires a deployed knowledge base)
    project_name = "travel-faq"  # Replace with your project name
    
    print("🔍 Testing Question Answering Service")
    print("=" * 50)
    
    # Test common questions
    test_questions = [
        "How can I cancel my reservation?",
        "What is the refund policy?",
        "Are pets allowed?",
        "What documents do I need?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        answers = demo.ask_question(question, project_name)
        
        if answers:
            best_answer = answers[0]
            print(f"✅ Answer: {best_answer['answer'][:150]}...")
            print(f"📊 Confidence: {best_answer['confidence']:.2f}")
        else:
            print("❌ No answer found")
    
    # Uncomment to start interactive session
    # demo.interactive_qa_session(project_name)
```

---

## Machine Learning Tutorial

### Tutorial 5: AutoML Training and Deployment

**Goal**: Train and deploy a machine learning model using AutoML.

```python
# automl_demo.py
import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient, automl
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

load_dotenv()

class AutoMLDemo:
    def __init__(self):
        self.client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_ML_WORKSPACE")
        )
    
    def create_automl_job(self, data_name, target_column, task_type="classification"):
        """Create an AutoML job"""
        if task_type == "classification":
            job = automl.classification(
                experiment_name="automl-classification-demo",
                compute="cpu-cluster",
                training_data=self.client.data.get(data_name, version="1"),
                target_column_name=target_column,
                primary_metric="accuracy",
                enable_model_explainability=True,
                max_trials=10,
                timeout_minutes=60
            )
        elif task_type == "regression":
            job = automl.regression(
                experiment_name="automl-regression-demo",
                compute="cpu-cluster",
                training_data=self.client.data.get(data_name, version="1"),
                target_column_name=target_column,
                primary_metric="normalized_root_mean_squared_error",
                enable_model_explainability=True,
                max_trials=10,
                timeout_minutes=60
            )
        
        return job
    
    def submit_job(self, job):
        """Submit the AutoML job"""
        try:
            submitted_job = self.client.jobs.create_or_update(job)
            print(f"✅ Job submitted successfully!")
            print(f"Job name: {submitted_job.name}")
            print(f"Job status: {submitted_job.status}")
            print(f"Studio URL: {submitted_job.studio_url}")
            return submitted_job
        except Exception as e:
            print(f"❌ Error submitting job: {e}")
            return None
    
    def monitor_job(self, job_name):
        """Monitor job progress"""
        try:
            job = self.client.jobs.get(job_name)
            print(f"Job Status: {job.status}")
            
            if job.status == "Completed":
                print("🎉 Job completed successfully!")
                return True
            elif job.status in ["Failed", "Canceled"]:
                print(f"❌ Job {job.status.lower()}")
                return False
            else:
                print("⏳ Job still running...")
                return None
        except Exception as e:
            print(f"Error monitoring job: {e}")
            return False
    
    def get_best_model(self, job_name):
        """Get the best model from completed AutoML job"""
        try:
            job = self.client.jobs.get(job_name)
            if job.status == "Completed":
                # Get the best model (this is a simplified example)
                print("🏆 Best model information:")
                print(f"Algorithm: {job.properties.get('algorithm', 'N/A')}")
                print(f"Primary metric value: {job.properties.get('best_metric', 'N/A')}")
                return job
            else:
                print("❌ Job not completed yet")
                return None
        except Exception as e:
            print(f"Error getting best model: {e}")
            return None

# Example usage for bike rental prediction
if __name__ == "__main__":
    demo = AutoMLDemo()
    
    print("🤖 AutoML Demo - Bike Rental Prediction")
    print("=" * 50)
    
    # Create and submit regression job
    try:
        job = demo.create_automl_job(
            data_name="bike-rentals",
            target_column="rentals",
            task_type="regression"
        )
        
        print("📊 Creating AutoML regression job...")
        submitted_job = demo.submit_job(job)
        
        if submitted_job:
            job_name = submitted_job.name
            print(f"\n⏳ Monitor your job at: {submitted_job.studio_url}")
            print(f"Job name: {job_name}")
            
            # Note: In a real scenario, you would periodically check job status
            # For demo purposes, we'll just show how to monitor
            print("\n📈 To monitor job progress:")
            print(f"demo.monitor_job('{job_name}')")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have:")
        print("  - Valid Azure ML workspace")
        print("  - Proper authentication")
        print("  - Required data assets")
```

---

## Common Use Cases

### Use Case 1: Content Moderation System

```python
# content_moderation.py
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

class ContentModerationSystem:
    def __init__(self, endpoint, key):
        self.client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
    
    def moderate_content(self, text):
        """Comprehensive content moderation"""
        results = {}
        
        # Sentiment analysis
        sentiment_results = self.client.analyze_sentiment([text])
        sentiment = sentiment_results[0]
        results['sentiment'] = {
            'overall': sentiment.sentiment,
            'confidence': sentiment.confidence_scores.__dict__
        }
        
        # PII detection
        pii_results = self.client.recognize_pii_entities([text])
        pii = pii_results[0]
        results['pii_detected'] = len(pii.entities) > 0
        results['pii_entities'] = [
            {'text': entity.text, 'category': entity.category}
            for entity in pii.entities
        ]
        
        # Custom moderation logic
        results['should_flag'] = self.should_flag_content(results)
        
        return results
    
    def should_flag_content(self, analysis):
        """Custom logic to determine if content should be flagged"""
        # Flag if very negative sentiment
        if (analysis['sentiment']['overall'] == 'negative' and 
            analysis['sentiment']['confidence']['negative'] > 0.8):
            return True
        
        # Flag if PII detected
        if analysis['pii_detected']:
            return True
        
        return False

# Usage
moderator = ContentModerationSystem(endpoint, key)
result = moderator.moderate_content("I hate this product! My email is john@example.com")
```

### Use Case 2: Document Intelligence Pipeline

```python
# document_intelligence.py
import asyncio
from azure.ai.textanalytics.aio import TextAnalyticsClient

class DocumentIntelligencePipeline:
    def __init__(self, endpoint, key):
        self.endpoint = endpoint
        self.key = key
    
    async def process_documents(self, documents):
        """Process multiple documents in parallel"""
        async with TextAnalyticsClient(self.endpoint, AzureKeyCredential(self.key)) as client:
            # Process all documents in parallel
            tasks = [
                self.process_single_document(client, doc, i) 
                for i, doc in enumerate(documents)
            ]
            
            results = await asyncio.gather(*tasks)
            return results
    
    async def process_single_document(self, client, document, doc_id):
        """Process a single document"""
        # Run multiple analyses in parallel
        sentiment_task = client.analyze_sentiment([document])
        entities_task = client.recognize_entities([document])
        key_phrases_task = client.extract_key_phrases([document])
        
        sentiment_result, entities_result, key_phrases_result = await asyncio.gather(
            sentiment_task, entities_task, key_phrases_task
        )
        
        return {
            'document_id': doc_id,
            'sentiment': sentiment_result[0].sentiment,
            'entities': [e.text for e in entities_result[0].entities],
            'key_phrases': key_phrases_result[0].key_phrases,
            'summary': self.create_summary(sentiment_result[0], entities_result[0], key_phrases_result[0])
        }
    
    def create_summary(self, sentiment, entities, key_phrases):
        """Create a document summary"""
        return {
            'tone': sentiment.sentiment,
            'main_topics': key_phrases.key_phrases[:5],
            'key_entities': [e.text for e in entities.entities[:5]],
            'confidence': max(sentiment.confidence_scores.__dict__.values())
        }

# Usage
async def main():
    pipeline = DocumentIntelligencePipeline(endpoint, key)
    documents = ["Document 1 text...", "Document 2 text...", "Document 3 text..."]
    results = await pipeline.process_documents(documents)
    
    for result in results:
        print(f"Document {result['document_id']}: {result['summary']}")

# asyncio.run(main())
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Authentication Errors

**Problem**: `401 Unauthorized` or `403 Forbidden` errors

**Solutions**:
```python
# Check endpoint format
endpoint = "https://your-resource.cognitiveservices.azure.com"  # Correct
endpoint = "your-resource.cognitiveservices.azure.com"          # Incorrect (missing https://)

# Verify key format
key = "your-32-character-subscription-key"  # Should be 32 characters

# Test connection
def test_connection():
    try:
        client = TextAnalyticsClient(endpoint, AzureKeyCredential(key))
        result = client.analyze_sentiment(["test"])
        print("✅ Connection successful")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
```

#### 2. Rate Limiting

**Problem**: `429 Too Many Requests` errors

**Solution**:
```python
import time
import random
from azure.core.exceptions import HttpResponseError

def retry_with_backoff(operation, max_retries=3):
    for attempt in range(max_retries):
        try:
            return operation()
        except HttpResponseError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            raise e

# Usage
result = retry_with_backoff(
    lambda: client.analyze_sentiment(["text to analyze"])
)
```

#### 3. Large Document Processing

**Problem**: Document too large or batch size exceeded

**Solution**:
```python
def process_large_text(text, max_chars=5000):
    """Split large text into chunks"""
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i + max_chars])
    return chunks

def process_large_batch(documents, batch_size=10):
    """Process large batches of documents"""
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_results = client.analyze_sentiment(batch)
        results.extend(batch_results)
    return results
```

#### 4. Regional Availability Issues

**Problem**: Service not available in selected region

**Solution**:
```python
# Check service availability by region
SUPPORTED_REGIONS = {
    'text_analytics': ['eastus', 'westus2', 'westeurope', 'southeastasia'],
    'vision': ['eastus', 'westus2', 'westeurope', 'southeastasia'],
    'face': ['eastus', 'westus', 'westeurope', 'southeastasia']
}

def check_region_support(service, region):
    return region.lower() in SUPPORTED_REGIONS.get(service, [])

# Example
if not check_region_support('text_analytics', 'northeurope'):
    print("⚠️  Text Analytics not available in North Europe")
    print("✅ Try: East US, West US 2, West Europe, or Southeast Asia")
```

---

This comprehensive quick start guide provides practical tutorials and examples for getting started with Azure AI services efficiently. Each tutorial builds upon previous concepts and includes real-world usage patterns and troubleshooting guidance.