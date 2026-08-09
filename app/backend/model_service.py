"""
Brain Tumor Detection — Model Inference Service

Handles model loading, image preprocessing, prediction, and Grad-CAM generation.
Designed as a singleton service loaded once at application startup.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)

# Preference order for model loading (best first)
MODEL_PREFERENCE = [
    "resnet50_best.h5",
    "resnet50_transfer.h5",
    "vgg16_best.h5",
    "vgg16_transfer.h5",
    "custom_best.h5",
    "custom_cnn.h5",
    "resnet50_transfer.keras",
    "vgg16_transfer.keras",
    "custom_cnn.keras",
]


class ModelService:
    """
    Encapsulates all ML inference logic:
    - Model loading with fallback discovery
    - Image preprocessing
    - Prediction with probabilities
    - Grad-CAM heatmap generation
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model = None
        self.model_name: Optional[str] = None
        self.input_shape: Optional[tuple] = None
        self.class_labels = CLASS_LABELS

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """
        Load the best available trained model from the models directory.
        Tries models in preference order; falls back to any .h5/.keras file.
        """
        try:
            import tensorflow as tf  # lazy import to keep startup fast if not needed
        except ImportError as exc:
            logger.error("TensorFlow is not installed. Cannot load model.")
            raise RuntimeError("TensorFlow is required for model inference.") from exc

        model_path = self._find_best_model()

        if model_path is None:
            logger.warning(
                "⚠️  No trained model found in '%s'. "
                "The API will start but predictions will fail until a model is trained.",
                self.model_dir,
            )
            return

        logger.info("Loading model from: %s", model_path)
        self.model = tf.keras.models.load_model(str(model_path))
        self.model_name = model_path.stem
        self.input_shape = tuple(self.model.input_shape[1:])
        logger.info(
            "Model '%s' loaded — input shape: %s, output classes: %d",
            self.model_name,
            self.input_shape,
            self.model.output_shape[-1],
        )

    def _find_best_model(self) -> Optional[Path]:
        """Return the path of the best available model, or None."""
        if not self.model_dir.exists():
            return None

        # Try preference order first
        for name in MODEL_PREFERENCE:
            candidate = self.model_dir / name
            if candidate.is_file():
                return candidate

        # Fallback: any .h5 or .keras file
        for pattern in ("*.h5", "*.keras"):
            found = list(self.model_dir.glob(pattern))
            if found:
                return sorted(found)[0]

        return None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Resize and normalize a PIL image for model input.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB image of any size.

        Returns
        -------
        np.ndarray
            Shape (1, 224, 224, 3) with pixel values in [0, 1].
        """
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)  # add batch dimension

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single image.

        Parameters
        ----------
        image : PIL.Image.Image
            The MRI scan to classify.

        Returns
        -------
        dict with keys: prediction, confidence, probabilities, gradcam_image
        """
        if self.model is None:
            raise RuntimeError("No model is loaded. Train a model first.")

        img_array = self.preprocess_image(image)
        predictions = self.model.predict(img_array, verbose=0)
        probs = predictions[0]

        predicted_idx = int(np.argmax(probs))
        predicted_label = self.class_labels[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Build probabilities dict
        probabilities = {
            label: round(float(prob), 4)
            for label, prob in zip(self.class_labels, probs)
        }

        # Generate Grad-CAM heatmap
        gradcam_b64 = self._generate_gradcam(img_array, predicted_idx)

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "gradcam_image": gradcam_b64,
        }

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------
    def _generate_gradcam(self, img_array: np.ndarray, class_idx: int) -> Optional[str]:
        """
        Generate a Grad-CAM heatmap overlay and return as base64-encoded PNG.

        Parameters
        ----------
        img_array : np.ndarray
            Preprocessed image array of shape (1, 224, 224, 3).
        class_idx : int
            Index of the predicted class.

        Returns
        -------
        str or None
            Base64-encoded PNG image, or None if generation fails.
        """
        try:
            import tensorflow as tf
            import cv2

            # Find the last convolutional layer
            last_conv_layer = self._find_last_conv_layer()
            if last_conv_layer is None:
                logger.warning("Could not find a convolutional layer for Grad-CAM.")
                return None

            # Build a model that outputs both the conv layer output and the predictions
            grad_model = tf.keras.models.Model(
                inputs=self.model.input,
                outputs=[last_conv_layer.output, self.model.output],
            )

            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                logger.warning("Grad-CAM: gradients are None.")
                return None

            # Pool gradients over spatial dimensions
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight the feature maps
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            heatmap = heatmap.numpy()

            # Resize heatmap to image dimensions
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

    def _find_last_conv_layer(self):
        """Find the last Conv2D layer in the model."""
        import tensorflow as tf

        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
            # Handle layers inside nested models (e.g., VGG16, ResNet50 base)
            if hasattr(layer, "layers"):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        return sub_layer
        return None
