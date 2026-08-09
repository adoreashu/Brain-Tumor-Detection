"""
Data loading and preprocessing utilities.
"""
import os
import logging
from typing import Tuple, Any
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.constants import DATASET_DIR, IMAGE_SIZE, BATCH_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_data_generators() -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """Creates ImageDataGenerators for training, validation, and testing."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, train_datagen, test_datagen

def get_train_val_test_data() -> Tuple[Any, Any, Any]:
    """Returns training, validation, and testing data generators."""
    train_dir = os.path.join(DATASET_DIR, 'Training')
    test_dir = os.path.join(DATASET_DIR, 'Testing')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Dataset directories not found in {DATASET_DIR}")
        
    train_datagen, val_datagen, test_datagen = create_data_generators()
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

if __name__ == '__main__':
    train_gen, val_gen, test_gen = get_train_val_test_data()
    logging.info(f"Training samples: {train_gen.samples}")
    logging.info(f"Validation samples: {val_gen.samples}")
    logging.info(f"Testing samples: {test_gen.samples}")
