"""
Brain Tumor Detection — FastAPI Backend Server

Serves the trained brain tumor classification model via REST API.
Accepts MRI image uploads and returns predictions with Grad-CAM heatmaps.
"""

import io
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.backend.model_service import ModelService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MAX_FILE_SIZE_MB = 10
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# ---------------------------------------------------------------------------
# Application lifespan (model loading)
# ---------------------------------------------------------------------------
model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model on startup, release on shutdown."""
    global model_service
    logger.info("🚀 Starting Brain Tumor Detection API ...")
    model_service = ModelService(model_dir=MODEL_DIR)
    model_service.load_model()
    logger.info("✅ Model loaded and ready for inference.")
    yield
    logger.info("🛑 Shutting down API server.")
    model_service = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Brain Tumor Detection API",
    description="Classify brain MRI scans into Glioma, Meningioma, Pituitary, or No Tumor.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow local dev + any Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Vercel, Render, localhost)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: validate uploaded file
# ---------------------------------------------------------------------------
def _validate_upload(file: UploadFile) -> None:
    """Check file extension and size constraints."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_service is not None and (model_service.model1 is not None or model_service.model2 is not None),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an MRI image upload and return tumor classification results.

    Returns
    -------
    JSON with prediction, confidence, per-class probabilities,
    and a base64-encoded Grad-CAM heatmap overlay.
    """
    # --- Validate ----------------------------------------------------------
    _validate_upload(file)

    if model_service is None or (model_service.model1 is None and model_service.model2 is None):
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    # --- Read image --------------------------------------------------------
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
            )
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to read uploaded image: %s", exc)
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.") from exc

    # --- Inference ---------------------------------------------------------
    try:
        result = model_service.predict(image)
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.") from exc

    logger.info(
        "Prediction: %s (%.2f%%) for file '%s'",
        result["prediction"],
        result["confidence"] * 100,
        file.filename,
    )

    return JSONResponse(content=result)


@app.get("/api/model-info")
async def model_info():
    """Return information about the currently loaded model."""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "model_name": "Ensemble (MobileNetV2 + EfficientNetB0)",
        "input_shape": [1, 224, 224, 3],
        "classes": model_service.class_labels,
        "num_classes": len(model_service.class_labels),
    }


# ---------------------------------------------------------------------------
# Main entry-point (for development)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
