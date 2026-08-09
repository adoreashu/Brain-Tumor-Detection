"""
Custom CNN model architecture.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from utils.constants import IMAGE_SIZE, CLASS_LABELS

def build_custom_cnn() -> tf.keras.Model:
    """Builds and returns a custom CNN model."""
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling and Dense Layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_LABELS), activation='softmax')
    ])
    
    return model

if __name__ == '__main__':
    model = build_custom_cnn()
    model.summary()
