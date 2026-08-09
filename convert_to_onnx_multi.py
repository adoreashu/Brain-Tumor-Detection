"""
Rebuild the ResNet50 model, extract weights, and export to multi-output ONNX.
Outputs:
1. Predictions (classification)
2. Last convolutional layer output (for Grad-CAM)
And saves dense weights to resnet50_weights.npz.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import tf2onnx
import onnx
import numpy as np

print("Building model architecture...")
inputs = layers.Input(shape=(224, 224, 3), name="input")
base = ResNet50(weights=None, include_top=False, input_tensor=inputs)
x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base.output)
x = layers.Dense(512, activation='relu', name="dense")(x)
x = layers.Dropout(0.3, name="dropout")(x)
out = layers.Dense(4, activation='softmax', name="dense_1")(x)
model = models.Model(inputs=inputs, outputs=out)

print("Loading weights...")
model.load_weights("models/resnet50_best.h5", by_name=True, skip_mismatch=True)

# Extract dense layer weights
print("Extracting weights for numpy backprop...")
w1, b1 = model.get_layer("dense").get_weights()
w2, b2 = model.get_layer("dense_1").get_weights()
np.savez("models/resnet50_weights.npz", w1=w1, b1=b1, w2=w2, b2=b2)
print("Saved models/resnet50_weights.npz!")

# Create a multi-output model that also outputs the conv output
print("Creating multi-output model for ONNX export...")
multi_model = models.Model(inputs=inputs, outputs=[out, base.output])

# Convert to ONNX
print("Converting to ONNX...")
input_signature = [tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(multi_model, input_signature=input_signature, opset=13)

# Save ONNX model
onnx.save(onnx_model, "models/resnet50_best.onnx")
print("SUCCESS! Saved models/resnet50_best.onnx with multi-output!")
