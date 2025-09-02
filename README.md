# RescueVision: The AI Backend

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Ultralytics-0052FF?style=for-the-badge" alt="Ultralytics"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/LangChain-FFFFFF?style=for-the-badge&logo=langchain" alt="LangChain"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
</p>

This repository contains the complete backend infrastructure for the **RescueVision AI Command Center**, a sophisticated, multi-modal AI system designed to accelerate search and rescue operations by analyzing aerial drone footage.

<p align="center">
  <img src="./assets/RescueVision_demo.gif" alt="RescueVision AI Command Center Demo" width="900"/>
</p>

### **➡️ [View the Frontend Here](https://github.com/MdEhsanulHaqueKanan/rescuevision-ai-command-center)**

The backend consists of two independent, production-grade microservices:
1.  **A Computer Vision (CV) Service:** A custom-trained deep learning model that processes entire video files to detect and locate survivors.
2.  **A Natural Language Processing (NLP) Service:** A Retrieval-Augmented Generation (RAG) pipeline that acts as a mission-critical AI assistant, answering operator questions from a specialized knowledge base.

This project demonstrates a complete, end-to-end MLOps workflow, from data acquisition and model training to building and testing robust, container-ready APIs.

---

## 🚀 Core Capabilities & Technical Achievements

### 1. High-Performance Survivor Detection (Computer Vision)

The core of the CV service is a state-of-the-art **YOLOv8n object detection model**, custom-trained and fine-tuned for the specific, challenging task of identifying human figures from an aerial perspective.

#### **Key Statistics & Results:**

*   **Dataset:** Trained on a robust, open-source aerial dataset containing **over 2,200 images** and **more than 8,000 labeled instances** of people in diverse environments (snow, fields, water).
*   **Training:** Model was trained for **25 epochs** on a Kaggle NVIDIA T4 GPU.
*   **Performance:** Achieved an exceptional validation score, demonstrating high accuracy and reliability:
    *   **Precision:** **85.3%** - When the model identifies a survivor, it is correct over 85% of the time.
    *   **Recall:** **76.2%** - The model successfully finds over 76% of all survivors in the validation set.
    *   **mAP50:** **0.833** - A very high score indicating excellent overall accuracy.
    *   **mAP50-95:** **0.447** - A strong score indicating the bounding boxes are not just present, but precise.
*   **API Endpoint (`/api/process_video`):** A professional endpoint that accepts a full video file, processes it frame-by-frame using OpenCV, and returns a complete JSON-based event log of all high-confidence detections with their precise timestamps and coordinates.

### 2. Mission AI Assistant (RAG & NLP)

The NLP service provides operators with instant access to critical information through a conversational interface, powered by a state-of-the-art RAG pipeline.

#### **Key Features & Architecture:**

*   **Knowledge Base:** Built from a corpus of official **FEMA Search and Rescue manuals**, ensuring all responses are grounded in authoritative, real-world operational procedures.
*   **Embedding Model:** Utilizes the powerful `all-MiniLM-L6-v2` sentence-transformer model to generate high-quality vector embeddings, capturing the semantic meaning of the source documents.
*   **Vector Database:** Employs **ChromaDB** to create a persistent, efficient, and searchable vector store containing **27 vectorized text chunks** from the knowledge base.
*   **Retrieval Logic:** The system takes a natural language query, creates an embedding, and performs a similarity search to retrieve the top 3 most relevant passages from the FEMA manuals.
*   **API Endpoint (`/api/query`):** A robust Flask endpoint that accepts a JSON-based question and returns a structured JSON object containing the retrieved, grounded answer chunks.

---

## 🛠️ Technology Stack

*   **AI & Deep Learning:** Python, PyTorch, Ultralytics YOLOv8, OpenCV
*   **NLP & RAG:** LangChain, Sentence-Transformers, ChromaDB
*   **Backend Framework:** Flask, Werkzeug
*   **Core Libraries:** Pandas, NumPy, Pillow
*   **Environment:** Conda, Pip

---

## 🏁 Getting Started

### Prerequisites

*   Python 3.10+
*   Conda installed

### Local Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd rescue-vision-project
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    conda create --name ResCueVisionEnvironment python=3.10 -y
    conda activate ResCueVisionEnvironment
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place the trained model:**
    *   Download the custom-trained `best.pt` model weights.
    *   Place the file in a top-level `/models` directory.

### Running the Services

The backend consists of two independent services. Run each in a separate terminal from the project root.

1.  **Start the RAG API:**
    ```bash
    python src/api/rag_api.py
    ```
    *(Service will be live at `http://localhost:5001`)*

2.  **Start the CV API:**
    ```bash
    python src/api/cv_api.py
    ```
    *(Service will be live at `http://localhost:5002`)*