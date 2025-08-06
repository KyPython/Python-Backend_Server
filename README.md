# **🧠 SmartScan AI: A Production-Ready AI Microservice**

This repository contains the Python-based AI microservice for the **[SmartScan AI](https://python-backend-server-8nsy.onrender.com)** project. It is a critical component that demonstrates the ability to deploy and optimize a machine learning model as a scalable web service, serving requests from a Node.js API gateway.

## **🔑 Key Skills Demonstrated**

This project showcases expertise in the full lifecycle of an AI-powered application, including:

  * **ML Model Deployment:** Deploying a complex image-to-text model from Hugging Face's `transformers` library into a production environment.
  * **Performance Engineering:** Implementing strategies for memory optimization, model selection, and efficient API serving.
  * **Microservices Architecture:** Building a focused, single-purpose service that integrates seamlessly into a larger, decoupled system.
  * **Production Deployment:** Configuring and using a robust WSGI server (`Gunicorn`) to manage concurrency, timeouts, and resource usage in a production setting.

## **🛠️ Tech Stack & Rationale**

The technology stack was specifically chosen to build an efficient and lightweight AI inference service.

  * **Flask** - A minimalist Python web framework, ideal for creating a focused, single-purpose API microservice.
  * **Transformers** - The industry-standard Hugging Face library for seamless integration with state-of-the-art models.
  * **BLIP** - The `Salesforce/blip-image-captioning-base` model was selected for its balanced performance and resource requirements.
  * **PyTorch** - The underlying deep learning framework that powers the AI model.
  * **Pillow** - A robust library for handling image preprocessing, a crucial step before feeding data to the model.
  * **Gunicorn** - A professional-grade Python WSGI server used to manage production workers and concurrency.

## **🚀 API Endpoints**

### `POST /local-caption`

**Purpose:** The primary API endpoint for generating a descriptive caption for a given image URL. It loads the image, processes it, and returns the AI-generated text.

**Request Body:**

```json
{
  "imageUrl": "https://example.com/image.jpg"
}
```

**Successful Response:**

```json
{
  "output": "a person sitting on a bench in a park"
}
```

## **📦 Dependencies**

The `requirements.txt` file ensures that all dependencies are version-controlled, a key professional practice for reproducible builds.

```txt
flask
torch
transformers
pillow
requests
gunicorn
```

## **📈 Deployment & Optimization Strategy**

This service was architected with a keen focus on balancing performance with resource constraints, a common challenge in AI deployment.

### **Model Selection**

The `BLIP-base` model was deliberately chosen over larger alternatives to manage the memory footprint and provide a responsive user experience.

## **🚀 Deployment (Render)**

The service is deployed on Render as a web service. The `gunicorn` command is configured as the entry point, demonstrating an understanding of how to run a production-ready Python application in a containerized environment. The architecture's separation of services allows for independent scaling and management of the AI component.
