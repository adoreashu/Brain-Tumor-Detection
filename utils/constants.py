"""
Constants for Brain Tumor Detection project.
"""
import os
from typing import List, Tuple

# Class labels
CLASS_LABELS: List[str] = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image settings
IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32

# Paths
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR: str = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR: str = os.path.join(BASE_DIR, 'models')
EVAL_DIR: str = os.path.join(BASE_DIR, 'evaluation')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
