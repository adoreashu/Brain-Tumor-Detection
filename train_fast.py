"""
Ultra-fast training script using pre-extracted MobileNetV2 features.
Extracts features in ~1-2 minutes, then trains the Dense classifier in 5 seconds.
Combines them into a single final model, evaluates it, and exports it to ONNX.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
import tf2onnx
import onnx
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

print("1. Loading data generators...")
from preprocessing.data_loader import get_train_val_test_data
train_gen, val_gen, test_gen = get_train_val_test_data()

# We need the class indices mapping
class_indices = train_gen.class_indices
print(f"Class indices mapping: {class_indices}")

# Build feature extractor base model
print("\n2. Building MobileNetV2 feature extractor...")
inputs = layers.Input(shape=(224, 224, 3))
x = inputs * 255.0
x = preprocess_input(x)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
base_model.trainable = False

# We want the output of the last convolutional layer (for Grad-CAM)
# and its global average pool (for MLP training input)
feature_extractor = models.Model(inputs=inputs, outputs=[base_model.output, base_model.output])

# Helper function to extract features from a generator
def extract_features(generator, name):
    print(f"Extracting features for {name} set ({generator.samples} samples)...")
    generator.reset()
    all_features = []
    all_labels = []
    
    # Iterate through the generator
    steps = len(generator)
    for step in range(steps):
        x_batch, y_batch = generator[step]
        # Run through MobileNetV2 to get features
        _, conv_out = feature_extractor.predict(x_batch, verbose=0)
        # Apply GlobalAveragePooling2D
        gap_features = np.mean(conv_out, axis=(1, 2))
        
        all_features.append(gap_features)
        all_labels.append(y_batch)
        if step % 20 == 0:
            print(f"  Progress: {step}/{steps} batches processed")
            
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

# Extract features
train_features, train_labels = extract_features(train_gen, "train")
val_features, val_labels = extract_features(val_gen, "val")
test_features, test_labels = extract_features(test_gen, "test")

print(f"\nFeature extraction complete!")
print(f"Train features shape: {train_features.shape}, labels shape: {train_labels.shape}")
print(f"Val features shape: {val_features.shape}, labels shape: {val_labels.shape}")

# Define MLP classifier
print("\n3. Training Dense MLP classifier on extracted features...")
mlp = models.Sequential([
    layers.Input(shape=(1280,)),
    layers.Dense(256, activation='relu', name="dense"),
    layers.Dropout(0.3, name="dropout"),
    layers.Dense(4, activation='softmax', name="dense_1")
])

mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the MLP classifier
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
]

mlp.fit(
    train_features, train_labels,
    validation_data=(val_features, val_labels),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Extract trained weights for NumPy Grad-CAM
w1, b1 = mlp.get_layer("dense").get_weights()
w2, b2 = mlp.get_layer("dense_1").get_weights()
np.savez("models/resnet50_weights.npz", w1=w1, b1=b1, w2=w2, b2=b2)
print("Saved models/resnet50_weights.npz!")

# Assemble the final model (Base Model + Trained MLP)
print("\n4. Assembling final end-to-end model...")
inputs_final = layers.Input(shape=(224, 224, 3), name="input")
x_final = inputs_final * 255.0
x_final = preprocess_input(x_final)
base_final = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x_final)
base_final.trainable = False

x_gap = layers.GlobalAveragePooling2D()(base_final.output)
predictions = mlp(x_gap)

# The model will output predictions AND the last conv output (base_final.output) for Grad-CAM
final_model = models.Model(inputs=inputs_final, outputs=[predictions, base_final.output])

# Evaluate final model accuracy on test features
test_preds = mlp.predict(test_features, verbose=0)
y_pred = np.argmax(test_preds, axis=1)
y_true = np.argmax(test_labels, axis=1)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=list(class_indices.keys())))

# Export final combined model to ONNX
print("\n5. Exporting combined model to ONNX...")
input_signature = [tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(final_model, input_signature=input_signature, opset=13)
onnx.save(onnx_model, "models/resnet50_best.onnx")
print("SUCCESS! Saved models/resnet50_best.onnx!")
