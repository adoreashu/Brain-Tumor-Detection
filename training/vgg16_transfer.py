"""
VGG16 Transfer Learning model architecture.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from utils.constants import IMAGE_SIZE, CLASS_LABELS

def build_vgg16_model() -> tf.keras.Model:
    """Builds and returns a VGG16-based transfer learning model."""
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    predictions = layers.Dense(len(CLASS_LABELS), activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

def unfreeze_vgg16(model: tf.keras.Model, num_layers: int = 4) -> tf.keras.Model:
    """Unfreezes the top 'num_layers' of the VGG16 base model for fine-tuning."""
    # Find the VGG16 layer
    base_model = None
    for layer in model.layers:
        if layer.name == 'vgg16':
            base_model = layer
            break
            
    if base_model:
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
            
    return model

if __name__ == '__main__':
    model = build_vgg16_model()
    model.summary()
