"""
Exploratory Data Analysis script.
"""
import os
import random
import logging
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple
from utils.constants import DATASET_DIR, CLASS_LABELS, EVAL_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_class_counts(split: str) -> List[int]:
    """Gets the number of images per class for a given split."""
    counts = []
    split_dir = os.path.join(DATASET_DIR, split)
    for class_name in CLASS_LABELS:
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            counts.append(len(os.listdir(class_dir)))
        else:
            counts.append(0)
    return counts

def plot_class_distribution() -> None:
    """Plots and saves a bar chart of class distribution."""
    train_counts = get_class_counts('Training')
    test_counts = get_class_counts('Testing')
    
    x = range(len(CLASS_LABELS))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], train_counts, width, label='Training')
    ax.bar([i + width/2 for i in x], test_counts, width, label='Testing')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.legend()
    
    save_path = os.path.join(EVAL_DIR, 'class_distribution.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved class distribution plot to {save_path}")

def plot_sample_images() -> None:
    """Plots and saves a 4x4 grid of sample images per class."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    train_dir = os.path.join(DATASET_DIR, 'Training')
    
    for i, class_name in enumerate(CLASS_LABELS):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        images = os.listdir(class_dir)
        samples = random.sample(images, min(4, len(images)))
        
        for j, img_name in enumerate(samples):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(class_name, loc='left')
                
    plt.tight_layout()
    save_path = os.path.join(EVAL_DIR, 'sample_images.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved sample images plot to {save_path}")

def plot_image_dimensions() -> None:
    """Plots and saves histograms of image dimensions."""
    widths = []
    heights = []
    train_dir = os.path.join(DATASET_DIR, 'Training')
    
    for class_name in CLASS_LABELS:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_name in os.listdir(class_dir)[:200]: # Sample 200 per class for speed
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                logging.warning(f"Failed to read {img_path}: {e}")
                
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(widths, bins=50, color='blue', alpha=0.7)
    ax1.set_title('Image Width Distribution')
    ax1.set_xlabel('Width (pixels)')
    
    ax2.hist(heights, bins=50, color='red', alpha=0.7)
    ax2.set_title('Image Height Distribution')
    ax2.set_xlabel('Height (pixels)')
    
    save_path = os.path.join(EVAL_DIR, 'image_dimensions.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved image dimensions plot to {save_path}")

if __name__ == '__main__':
    plot_class_distribution()
    plot_sample_images()
    plot_image_dimensions()
