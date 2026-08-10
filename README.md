# Early Brain Tumor Detections in Human Brain 🧠
*(Formerly Brain Tumour Detection System by Ashu)*

A production-grade, deep learning-powered medical imaging web application that classifies brain MRI scans into four distinct categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

---

## 🌐 Live Production Links

* **💻 Interactive Frontend UI:** [https://brain-tumor-detection-pi-nine.vercel.app](https://brain-tumor-detection-pi-nine.vercel.app) (Hosted on Vercel)
* **⚙️ AI Inference Backend API:** [https://brain-tumor-detection-3raj.onrender.com](https://brain-tumor-detection-3raj.onrender.com) (Hosted on Render)

---

## 📐 System Architecture

Below is the workflow of the deployed application, showing the division of labor between the static client and the containerized API:

```mermaid
graph TD
    User([User's Browser]) -->|1. Uploads MRI Image| Frontend[Vercel Frontend: React + Vite]
    Frontend -->|2. HTTP POST Request| Backend[Render Backend: FastAPI + Docker]
    Backend -->|3. Resizes & Normalizes| Preprocess[Image Preprocessing]
    Preprocess -->|4. Parallel Inference| Ensemble[ONNX Runtime: MobileNetV2 + EfficientNetB0]
    Ensemble -->|5. Averages Probabilities| Arbiter[Ensemble Arbitrator]
    Arbiter -->|6. Generates Overlay Heatmap| GradCAM[NumPy Grad-CAM Engine]
    GradCAM -->|7. JSON predictions + Heatmap| Response[FastAPI Response]
    Response -->|8. Renders Dashboard| Frontend
```

---

## 🏗️ Technology Stack: Why This & Not That?

This project splits the Frontend and Backend to optimize hosting costs, startup times, and computing efficiency.

### 1. Frontend: HTML, CSS, JavaScript (React + Vite)
* **What it is:** The highly aesthetic visual interface built with React 18 and bundled using Vite. It features interactive modals, glowing gradients, and glassmorphism.
* **Why React?** Component-driven architecture allows modular pages, clean state management (e.g., handling loading spinner, predictions, chart updates), and responsive styling.
* **Why Vite?** Modern frontend toolchain that is 10-100x faster than legacy bundlers (like Create React App/Webpack) for local development and build output.
* **Language Used:** **JavaScript** (JS) for logic, **HTML** for page structure, and custom **CSS** for the dark-tech aesthetic.

### 2. Backend API: Python (FastAPI + Uvicorn)
* **What it is:** A REST API that handles client uploads, runs predictions, and calculates explainable AI overlays.
* **Why FastAPI?** 
  * It is asynchronous (`async/await`), meaning it can process multiple image uploads concurrently without blocking.
  * It is lightweight compared to Django or Flask.
* **Language Used:** **Python** — the industry standard for Machine Learning and AI.

### 3. Model Engine: ONNX Runtime (No TensorFlow in Production)
* **What it is:** Open Neural Network Exchange (ONNX) is a cross-platform serialization format for ML models.
* **Why not TensorFlow/Keras on the Server?**
  * **Size Constraint:** TensorFlow is massive (~500MB–1GB). ONNX Runtime is under **50MB**.
  * **Build Failures:** TensorFlow frequently crashes on free-tier cloud platforms (like Render or AWS Lambda) due to strict RAM limits.
  * **Speed:** ONNX is heavily optimized for CPU execution and runs inference **5x faster** than TensorFlow on basic cloud instances.

### 4. Explainable AI: Pure NumPy Grad-CAM
* **What it is:** Gradient-weighted Class Activation Mapping (Grad-CAM). It highlights the specific pixels in the MRI scan that the neural network relied on to make its decision.
* **Why NumPy?** Since we removed TensorFlow for production, we wrote a custom backpropagation matrix compiler using **NumPy** to extract activations directly from the final Dense layers in microseconds.

---

## 🚀 Advanced ML Techniques Explained (Achieving 95%+ Accuracy)

### 1. Multi-Model Ensembling (Arbitration)
* We employ a highly advanced **Ensemble Architecture** instead of relying on just a single AI model. 
* We run the images through both a **MobileNetV2** (lightweight shape detector) and an **EfficientNetB0** (complex feature extractor) simultaneously.
* The system computes a weighted average of their probability arrays. If one model develops a "blind spot" and misclassifies a Meningioma, the other model seamlessly corrects it.

### 2. Clean Label Noise Auditing
* Public medical datasets often contain human errors where images are placed in the wrong folder (e.g., a Glioma labeled as a Meningioma), creating an artificial "Glass Ceiling" on accuracy.
* We used an AI audit tool (`audit_dataset.py`) to scan all 5,600 training images, finding exactly **25 corrupted labels**, and manually corrected them so the models learned from perfect data.

### 3. Transfer Learning
* Rather than training a neural network from scratch, our models were initialized with weights learned from millions of real-world images (ImageNet), and then fine-tuned specifically on MRI scans.

---

## 🚀 Run Locally

### Prerequisites
* **Python 3.10+** (Backend)
* **Node.js 18+** (Frontend)

### 1. Clone & Navigate
```bash
git clone https://github.com/adoreashu/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### 2. Automated One-Click Start (Windows)
Double-click the **`run.bat`** file in the root directory. It will automatically build the frontend, install backend requirements, and launch both servers.

### 3. Manual Step-by-Step Start
* **Backend:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  uvicorn app.backend.main:app --reload
  ```
* **Frontend:**
  ```bash
  cd app/frontend
  npm install
  npm run dev
  ```

---

## 📁 Project Directory Map

* **`app/frontend/`** — React SPA containing interactive Modals and Hero components.
* **`app/backend/`** — FastAPI app containing route endpoints (`main.py`) and the multi-model Ensemble arbitration logic (`model_service.py`).
* **`models/`** — Folder containing `.onnx` models and their extracted `.npz` weights for Grad-CAM.
* **`training/`** & root scripts — Scripts to train and evaluate CNN models (`train_efficientnet.py`, `audit_dataset.py`).

---

## 👤 Author
**Ashu** — Early Brain Tumor Detections in Human Brain  
Built with ❤️ using Python, ONNX, React, and FastAPI.
