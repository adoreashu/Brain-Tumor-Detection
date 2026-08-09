# Brain Tumour Detection System by Ashu

A deep learning-powered web application that classifies brain MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor** — with 86%+ accuracy using a ResNet50 transfer learning model.

---

## 🌐 Live Demo
> Deploy using the steps below to get your own live link!

---

## ✨ Features
- 🧠 **AI-Powered Detection** — ResNet50 transfer learning model trained on 7,200+ MRI scans
- 📊 **Confidence Scores** — Per-class probability breakdown with beautiful bar charts
- 🔥 **Grad-CAM Heatmaps** — Visualize exactly where the AI is looking in the MRI
- ⚡ **Real-Time Analysis** — FastAPI backend processes images in under 2 seconds
- 🎨 **Premium UI** — Dark glassmorphism design built with React + Vite

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, CSS Glassmorphism |
| Backend | Python, FastAPI, Uvicorn |
| AI Model | TensorFlow, Keras, ResNet50 |
| Visualization | Grad-CAM, Chart.js |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Clone the repository
```bash
git clone https://github.com/adoreashu/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### 2. Set up Python backend
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Train the model (first time only)
```bash
python training/train.py --model resnet50 --epochs 5
```

### 4. Start the backend
```bash
uvicorn app.backend.main:app --reload
```

### 5. Start the frontend (new terminal)
```bash
cd app/frontend
npm install
npm run dev
```

### 6. Open the app
Visit **http://localhost:3000**

---

## 🌍 Deploy Globally

### Backend → [Render.com](https://render.com)
1. Connect your GitHub repo to Render
2. It auto-detects the `Dockerfile` and builds the API
3. Copy the live URL (e.g. `https://brain-tumor-api.onrender.com`)

### Frontend → [Vercel.com](https://vercel.com)
1. Import this repo into Vercel
2. Set Root Directory to `app/frontend`
3. Add Environment Variable: `VITE_API_URL` = *(your Render backend URL)*
4. Deploy!

---

## 📁 Project Structure
```
Brain-Tumor-Detection/
├── app/
│   ├── backend/          # FastAPI server + Model service
│   └── frontend/         # React + Vite web app
├── training/             # Model training scripts
├── preprocessing/        # Data loading & augmentation
├── evaluation/           # Evaluation & Grad-CAM scripts
├── utils/                # Constants & helpers
├── models/               # Trained model weights (gitignored)
├── Dockerfile            # Backend container config
├── requirements.txt      # Python dependencies
└── run.bat               # One-click local launcher (Windows)
```

---

## 👤 Author
**Ashu** — Brain Tumour Detection System  
Built with ❤️ using Python, TensorFlow, React, and FastAPI
