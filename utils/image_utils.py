"""
Image processing utilities for Brain Tumor Detection.
"""
import os
import base64
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from typing import Tuple, Optional

def load_and_preprocess_image(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Loads an image and preprocesses it to the target size."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    img = image.load_img(path, target_size=target_size)
    img_array = image.img_to_array(img)
    return img_array

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalizes image pixel values to [0, 1]."""
    return img / 255.0

def apply_gradcam_heatmap(model: tf.keras.Model, img: np.ndarray, class_idx: int, layer_name: str) -> np.ndarray:
    """Generates a Grad-CAM heatmap for a specific image and class."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def image_to_base64(img_array: np.ndarray) -> str:
    """Converts a numpy array image to base64 string."""
    img = image.array_to_img(img_array)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
