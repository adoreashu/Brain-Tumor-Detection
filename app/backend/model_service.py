"""
Brain Tumor Detection — Model Inference Service (Ensemble ONNX Runtime Version)

Handles model loading, image preprocessing, prediction, and Grad-CAM generation using ONNX Runtime.
Loads multiple models (if available) and ensembles their probabilities for maximum accuracy.
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

class ModelService:
    """
    Encapsulates all ML inference logic using ONNX Runtime:
    - Model loading (Ensemble support: MobileNetV2 + EfficientNetB0)
    - Image preprocessing
    - Prediction with ensembled probabilities
    - Custom numpy-based Grad-CAM heatmap generation
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        
        self.model1 = None  # resnet50_best.onnx (MobileNetV2)
        self.model2 = None  # efficientnet_best.onnx (EfficientNetB0)
        
        self.class_labels = CLASS_LABELS
        self.weights = None  # Loaded dense layer weights for Grad-CAM numpy backprop (using model1)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """
        Load the trained ONNX models from the models directory.
        """
        # Set up session options for optimal CPU performance
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load Model 1 (MobileNetV2 / Resnet50)
        m1_path = self.model_dir / "resnet50_best.onnx"
        if m1_path.is_file():
            try:
                self.model1 = ort.InferenceSession(str(m1_path), sess_options)
                logger.info("ONNX Model 1 loaded: %s", m1_path.name)
            except Exception as e:
                logger.error("Failed to load ONNX Model 1: %s", e)

        # Load Model 2 (EfficientNetB0)
        m2_path = self.model_dir / "efficientnet_best.onnx"
        if m2_path.is_file():
            try:
                self.model2 = ort.InferenceSession(str(m2_path), sess_options)
                logger.info("ONNX Model 2 loaded: %s", m2_path.name)
            except Exception as e:
                logger.error("Failed to load ONNX Model 2: %s", e)

        if self.model1 is None and self.model2 is None:
            logger.warning("⚠️ No trained ONNX models found in '%s'. Predictions will fail.", self.model_dir)

        # Load weights for NumPy Grad-CAM if available (using model 1 weights)
        weights_path = self.model_dir / "resnet50_weights.npz"
        if weights_path.is_file():
            try:
                self.weights = np.load(str(weights_path))
                logger.info("Loaded NumPy Grad-CAM weights from %s", weights_path)
            except Exception as weights_err:
                logger.warning("Could not load NumPy Grad-CAM weights: %s", weights_err)

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
        Run inference on a single image using Ensemble ONNX sessions.
        """
        if self.model1 is None and self.model2 is None:
            raise RuntimeError("No model is loaded. Train a model first.")

        img_array = self.preprocess_image(image)
        
        all_probs = []
        outputs_m1 = None

        # Run Model 1
        if self.model1 is not None:
            input_name = self.model1.get_inputs()[0].name
            output_names = [o.name for o in self.model1.get_outputs()]
            outputs_m1 = self.model1.run(output_names, {input_name: img_array})
            all_probs.append(outputs_m1[0][0])

        # Run Model 2
        if self.model2 is not None:
            input_name = self.model2.get_inputs()[0].name
            output_names = [o.name for o in self.model2.get_outputs()]
            outputs_m2 = self.model2.run(output_names, {input_name: img_array})
            all_probs.append(outputs_m2[0][0])

        # Ensemble Average
        avg_probs = np.mean(all_probs, axis=0)

        predicted_idx = int(np.argmax(avg_probs))
        predicted_label = self.class_labels[predicted_idx]
        confidence = float(avg_probs[predicted_idx])

        # Build probabilities dict
        probabilities = {
            label: round(float(prob), 4)
            for label, prob in zip(self.class_labels, avg_probs)
        }

        # Generate Grad-CAM heatmap using Model 1's features if available
        gradcam_b64 = None
        if outputs_m1 is not None and len(outputs_m1) > 1:
            conv_outputs = outputs_m1[1]
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

            # conv_outputs shape: (1, 7, 7, 2048) or similar
            # GAP is global average pool of conv_outputs
            gap = np.mean(conv_outputs, axis=(1, 2))

            # First Dense layer forward pass (Relu)
            z1 = gap @ w1 + b1
            h = np.maximum(z1, 0)

            # Compute derivative of predicted logit score with respect to GAP
            dy_dh = w2[:, class_idx]

            # dy/dz1 = dy/dh * d(Relu(z1))/dz1 = dy/dh * (z1[0] > 0)
            dy_dz1 = dy_dh * (z1[0] > 0)

            # dy/dgap = dy_dz1 @ w1.T
            dy_dgap = w1 @ dy_dz1

            # The weights for the last conv layer feature channels is exactly dy_dgap
            weights = dy_dgap

            # Compute weighted combination of conv channels
            heatmap = np.dot(conv_outputs[0], weights)

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
