"""
Grad-CAM visualization utilities.
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image
from utils.constants import CLASS_LABELS, MODEL_DIR, EVAL_DIR, IMAGE_SIZE
from utils.image_utils import load_and_preprocess_image

class GradCAM:
    """Class for generating Grad-CAM heatmaps."""
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.last_conv_layer = self._find_last_conv_layer()
        
    def _find_last_conv_layer(self) -> str:
        """Automatically detects the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if isinstance(layer, tf.keras.Model): # Handle nested models (like ResNet/VGG base)
                for inner_layer in reversed(layer.layers):
                    if isinstance(inner_layer, tf.keras.layers.Conv2D):
                        return layer.name # We need the wrapper layer name if it's functional
        raise ValueError("Could not find a convolutional layer in the model.")

    def generate_heatmap(self, img_array: np.ndarray, class_idx: int = None) -> np.ndarray:
        """Generates the heatmap for the given image."""
        # Find the actual tensor output layer
        layer_name = self.last_conv_layer
        
        # If the layer is a nested model, we might need a custom model for gradient tape
        # For simplicity, assuming standard sequential or functional API structure
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
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

def overlay_heatmap(img_path: str, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blends the heatmap with the original image."""
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)
    
    return np.array(superimposed_img)

def visualize_gradcam(model_name: str, img_path: str, save_name: str = "gradcam_result.png") -> None:
    """Creates a side-by-side plot of original, heatmap, and overlay."""
    model_path = os.path.join(MODEL_DIR, f'{model_name}_best.h5')
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found.")
        return
        
    model = tf.keras.models.load_model(model_path)
    gradcam = GradCAM(model)
    
    img_array = load_and_preprocess_image(img_path, IMAGE_SIZE)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 # Normalize for prediction
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    pred_class = CLASS_LABELS[class_idx]
    
    heatmap = gradcam.generate_heatmap(img_array, class_idx)
    overlay = overlay_heatmap(img_path, heatmap)
    
    original_img = image.load_img(img_path, target_size=IMAGE_SIZE)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original Image\nPredicted: {pred_class}')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Superimposed')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(EVAL_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Grad-CAM visualization saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument('--model', type=str, default='custom', help='Model to use')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        visualize_gradcam(args.model, args.image)
    else:
        print("Image not found!")
