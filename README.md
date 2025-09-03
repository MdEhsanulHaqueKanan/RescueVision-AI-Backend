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

### Final Application Screenshot
![RescueVision Screenshot](./assets/RescueVision.png)

##### **➡️ [View the Frontend Here](https://github.com/MdEhsanulHaqueKanan/RescueVision-Frontend)**

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

---

## 🔬 Future Enhancements & Research Roadmap

The RescueVision platform is a robust MVP, but its architecture is designed for significant future expansion. The following roadmap outlines key areas for further research and development to evolve this tool into a fully-fledged, real-time operational asset.

### 1. Transition to Real-Time Tracking with Frontend Intelligence

*   **Objective:** Evolve from an "Event Reviewer" to a live, real-time tracking system suitable for active missions.
*   **Methodology:**
    *   **Implement a Frontend Tracking Algorithm:** Integrate a lightweight, efficient tracking algorithm (e.g., a Centroid Tracker or a simple Kalman Filter) directly into the React frontend.
    *   **Decouple Detection from Tracking:** The powerful YOLOv8 backend will continue to run at a throttled pace (e.g., 2-3 times per second) to provide high-confidence *detections*. The frontend tracker will use these detections as its "ground truth" and then interpolate the bounding box positions smoothly across the frames in between, running at 60fps.
*   **Expected Outcome:** A flawless, real-time user experience that provides the illusion of a 60fps AI tracker while efficiently managing backend resources, solving the latency problem we observed during initial prototyping.

### 2. Multi-Object and Vehicle Detection

*   **Objective:** Expand the model's capabilities to detect other critical "signs of life" in a disaster zone.
*   **Methodology:**
    *   **Data Sourcing:** Augment the current dataset with labeled images of vehicles (cars, boats), backpacks, and potentially large animals.
    *   **Multi-Class Model Training:** Re-train the YOLOv8 model on this new, multi-class dataset.
    *   **Frontend Updates:** Update the UI to display different colored bounding boxes and icons for each detected class (e.g., blue for 'person', green for 'vehicle').

### 3. Audio Analysis for "Cries for Help"

*   **Objective:** Add another modality to the system by analyzing the drone's audio feed for human vocal distress signals.
*   **Methodology:**
    *   **Research & Model Selection:** Investigate and fine-tune a pre-trained audio classification model (e.g., YAMNet) on a dataset of human sounds, specifically targeting keywords like "help."
    *   **New Microservice:** Develop a third, independent microservice for audio processing that accepts an audio stream and returns timestamped events for potential vocal distress.
    *   **UI Integration:** Add an "Audio Alert" indicator to the Detections Log.

### 4. Deployment to Edge Devices

*   **Objective:** Move the AI inference from a cloud/local server to a low-power, high-efficiency computer on the drone itself for true autonomy.
*   **Methodology:**
    *   **Model Optimization:** Convert the trained PyTorch model to a more efficient format like ONNX or TensorRT.
    *   **Quantization:** Further optimize the model by converting its weights from 32-bit floating points to 8-bit integers, drastically reducing its size and computational requirements.
    *   **Edge Deployment:** Deploy the optimized model to an edge device like an **NVIDIA Jetson Nano** mounted on the drone.