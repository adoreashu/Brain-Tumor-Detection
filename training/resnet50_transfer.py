"""
ResNet50 Transfer Learning model architecture.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from utils.constants import IMAGE_SIZE, CLASS_LABELS

from tensorflow.keras.applications.resnet50 import preprocess_input

def build_resnet50_model() -> tf.keras.Model:
    """Builds and returns a ResNet50-based transfer learning model."""
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    
    # Preprocess inputs from [0, 1] to caffe-style format expected by ResNet50
    x = inputs * 255.0
    x = preprocess_input(x)
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    base_model.trainable = False
    
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    predictions = layers.Dense(len(CLASS_LABELS), activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=predictions)
    return model

def unfreeze_resnet50(model: tf.keras.Model, num_layers: int = 10) -> tf.keras.Model:
    """Unfreezes the top 'num_layers' of the ResNet50 base model for fine-tuning."""
    # Find the ResNet50 layer
    base_model = None
    for layer in model.layers:
        if layer.name == 'resnet50':
            base_model = layer
            break
            
    if base_model:
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
            
    return model

if __name__ == '__main__':
    model = build_resnet50_model()
    model.summary()
