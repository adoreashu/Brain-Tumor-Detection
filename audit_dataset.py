"""
Dataset Auditing Tool for Clean Label Noise
This script scans all training and validation images, runs them through the 
trained model, and flags any images where the model strongly disagrees with 
the folder label. These are likely misclassified images (label noise).
"""
import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)
DATASET_DIR = Path("dataset")
MODEL_PATH = Path("models/resnet50_best.onnx")

def preprocess_image(image_path: Path) -> np.ndarray:
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def main():
    if not MODEL_PATH.exists():
        logging.error(f"Model not found at {MODEL_PATH}. Train a model first.")
        return

    logging.info(f"Loading model {MODEL_PATH}...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    model = ort.InferenceSession(str(MODEL_PATH), sess_options)
    input_name = model.get_inputs()[0].name
    output_names = [model.get_outputs()[0].name]

    flagged_images = []
    
    # We will scan the Training dataset
    train_dir = DATASET_DIR / "Training"
    if not train_dir.exists():
        logging.error(f"Training directory not found at {train_dir}")
        return

    logging.info("Scanning dataset for label noise (this may take a few minutes)...")
    
    total_images = 0
    for class_label in CLASS_LABELS:
        class_dir = train_dir / class_label
        if not class_dir.exists():
            continue
            
        images = list(class_dir.glob("*.jpg"))
        logging.info(f"Scanning {len(images)} images in {class_label} folder...")
        
        for idx, img_path in enumerate(images):
            total_images += 1
            if total_images % 500 == 0:
                logging.info(f"Processed {total_images} images so far...")
                
            img_array = preprocess_image(img_path)
            if img_array is None:
                continue
                
            # Run inference
            outputs = model.run(output_names, {input_name: img_array})
            probs = outputs[0][0]
            
            predicted_idx = int(np.argmax(probs))
            predicted_label = CLASS_LABELS[predicted_idx]
            confidence = float(probs[predicted_idx])
            
            # Label Noise Logic:
            # If the model is extremely confident (>98%) but disagrees with the folder label
            if predicted_label != class_label and confidence > 0.98:
                flagged_images.append({
                    "file_path": str(img_path),
                    "actual_folder_label": class_label,
                    "model_prediction": predicted_label,
                    "model_confidence": round(confidence, 4)
                })

    # Save findings
    output_file = "potential_label_errors.csv"
    if flagged_images:
        keys = flagged_images[0].keys()
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flagged_images)
        logging.warning(f"Found {len(flagged_images)} potentially mislabeled images!")
        logging.warning(f"Report saved to {output_file}. Please review these images manually.")
    else:
        logging.info("No obvious label noise found! The dataset looks clean.")
        # Create an empty file to indicate completion
        with open(output_file, 'w', newline='') as f:
            f.write("No obvious label noise found.\n")

if __name__ == "__main__":
    main()
