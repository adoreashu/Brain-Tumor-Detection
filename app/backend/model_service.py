"""
Brain Tumor Detection — Model Inference Service (ONNX Runtime Version)

Handles model loading, image preprocessing, prediction, and Grad-CAM generation using ONNX Runtime.
Eliminates TensorFlow dependencies for extremely fast and robust production deployments.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)

# We prefer ONNX format for deployment
MODEL_PREFERENCE = [
    "resnet50_best.onnx",
]


class ModelService:
    """
    Encapsulates all ML inference logic using ONNX Runtime:
    - Model loading with fallback discovery
    - Image preprocessing
    - Prediction with probabilities
    - Custom numpy-based Grad-CAM heatmap generation
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model = None  # Holds the ort.InferenceSession
        self.model_name: Optional[str] = None
        self.input_shape: Optional[tuple] = None
        self.class_labels = CLASS_LABELS
        self.weights = None  # Loaded dense layer weights for Grad-CAM numpy backprop

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """
        Load the best available trained ONNX model from the models directory.
        """
        model_path = self._find_best_model()

        if model_path is None:
            logger.warning(
                "⚠️  No trained ONNX model found in '%s'. "
                "The API will start but predictions will fail until a model is trained.",
                self.model_dir,
            )
            return

        logger.info("Loading ONNX model from: %s", model_path)
        try:
            # Set up session options for optimal CPU performance
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.model = ort.InferenceSession(str(model_path), sess_options)
            self.model_name = model_path.name
            
            inputs = self.model.get_inputs()
            self.input_shape = tuple(inputs[0].shape[1:])
            
            logger.info(
                "ONNX Model '%s' loaded — input shape: %s, output classes: %s",
                self.model_name,
                self.input_shape,
                len(self.class_labels),
            )
        except Exception as load_err:
            logger.error("Failed to load ONNX model: %s", load_err)
            self.model = None
            return

        # Load weights for NumPy Grad-CAM if available
        weights_path = self.model_dir / "resnet50_weights.npz"
        if weights_path.is_file():
            try:
                self.weights = np.load(str(weights_path))
                logger.info("Loaded NumPy Grad-CAM weights from %s", weights_path)
            except Exception as weights_err:
                logger.warning("Could not load NumPy Grad-CAM weights: %s", weights_err)
        else:
            logger.warning("NumPy Grad-CAM weights not found at %s. Grad-CAM will be disabled.", weights_path)

    def _find_best_model(self) -> Optional[Path]:
        """Return the path of the best available model, or None."""
        if not self.model_dir.exists():
            return None

        # Try preference order first
        for name in MODEL_PREFERENCE:
            candidate = self.model_dir / name
            if candidate.is_file():
                return candidate

        # Fallback: any .onnx file
        found = list(self.model_dir.glob("*.onnx"))
        if found:
            return sorted(found)[0]

        return None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Resize and normalize a PIL image for model input.
        """
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)  # add batch dimension

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single image using ONNX session.
        """
        if self.model is None:
            raise RuntimeError("No model is loaded. Train a model first.")

        img_array = self.preprocess_image(image)
        
        # Get input and output names
        input_name = self.model.get_inputs()[0].name
        output_names = [o.name for o in self.model.get_outputs()]
        
        # Run inference
        outputs = self.model.run(output_names, {input_name: img_array})
        
        # Mapping: output 0 is predictions, output 1 is conv outputs (for Grad-CAM)
        probs_output = outputs[0]
        probs = probs_output[0]

        predicted_idx = int(np.argmax(probs))
        predicted_label = self.class_labels[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Build probabilities dict
        probabilities = {
            label: round(float(prob), 4)
            for label, prob in zip(self.class_labels, probs)
        }

        # Generate Grad-CAM heatmap if conv_outputs exist in model outputs
        gradcam_b64 = None
        if len(outputs) > 1:
            conv_outputs = outputs[1]
            gradcam_b64 = self._generate_gradcam_numpy(img_array, conv_outputs, predicted_idx)

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "gradcam_image": gradcam_b64,
        }

    # ------------------------------------------------------------------
    # NumPy-based Grad-CAM
    # ------------------------------------------------------------------
    def _generate_gradcam_numpy(self, img_array: np.ndarray, conv_outputs: np.ndarray, class_idx: int) -> Optional[str]:
        """
        Generate a Grad-CAM heatmap overlay using NumPy backpropagation and return as base64-encoded PNG.
        """
        if self.weights is None:
            return None

        try:
            # Extract weights for backprop
            w1, b1 = self.weights["w1"], self.weights["b1"]
            w2, b2 = self.weights["w2"], self.weights["b2"]

            # conv_outputs shape: (1, 7, 7, 2048)
            # GAP is global average pool of conv_outputs
            gap = np.mean(conv_outputs, axis=(1, 2))  # shape: (1, 2048)

            # First Dense layer forward pass (Relu)
            z1 = gap @ w1 + b1  # shape: (1, 512)
            h = np.maximum(z1, 0)  # Relu activation, shape: (1, 512)

            # Second Dense layer forward pass (Logits / Softmax)
            # logits = h @ w2 + b2  # shape: (1, 4)

            # Compute derivative of predicted logit score with respect to GAP
            # dy/dlogits is a one-hot vector with 1 at class_idx, 0 elsewhere.
            # So dy/dh = w2[:, class_idx] (shape: (512,))
            dy_dh = w2[:, class_idx]

            # dy/dz1 = dy/dh * d(Relu(z1))/dz1 = dy/dh * (z1[0] > 0)
            dy_dz1 = dy_dh * (z1[0] > 0)  # shape: (512,)

            # dy/dgap = dy_dz1 @ w1.T
            dy_dgap = w1 @ dy_dz1  # shape: (2048,)

            # The weights for the last conv layer feature channels is exactly dy_dgap
            weights = dy_dgap

            # Compute weighted combination of conv channels
            # conv_outputs[0] shape: (7, 7, 2048)
            heatmap = np.dot(conv_outputs[0], weights)  # shape: (7, 7)

            # Apply Relu to heatmap and normalize
            heatmap = np.maximum(heatmap, 0)
            max_val = np.max(heatmap)
            if max_val > 0:
                heatmap /= max_val

            # Resize heatmap to target image dimensions
            heatmap = cv2.resize(heatmap, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

            # Colorize with JET colormap
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Overlay on original image
            original = np.uint8(img_array[0] * 255)
            overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

            # Encode to base64
            overlay_image = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_image.save(buffer, format="PNG")
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return b64_str

        except Exception as exc:
            logger.error("Grad-CAM generation failed: %s", exc)
            return None
