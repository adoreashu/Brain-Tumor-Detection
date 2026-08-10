"""
Deep Fine-Tuning Script for High Accuracy Brain Tumor Detection.
This script trains EfficientNetB0 by unfreezing its deeper layers to learn 
actual MRI textures, drastically improving accuracy (but taking longer to train).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import tf2onnx
import onnx
from sklearn.metrics import classification_report, confusion_matrix

def train_fine_tuned_model():
    print("=========================================================")
    print("🧠 Starting Deep Fine-Tuning of EfficientNetB0")
    print("=========================================================")
    print("Note: This will take longer than fast training but will yield much higher accuracy.\n")

    # 1. Load Data with strong Augmentation
    print("1. Loading and augmenting dataset...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.2
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    dataset_dir = "dataset"
    train_dir = os.path.join(dataset_dir, 'Training')
    test_dir = os.path.join(dataset_dir, 'Testing')

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32,
        class_mode='categorical', subset='validation', shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=32,
        class_mode='categorical', shuffle=False
    )
    class_indices = train_gen.class_indices

    # 2. Build the Model
    print("\n2. Building the Model Architecture...")
    inputs = layers.Input(shape=(224, 224, 3), name="input")
    x = inputs * 255.0 
    
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
    
    # Unfreeze the top 30 layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x_conv = base_model.output
    x_gap = layers.GlobalAveragePooling2D(name="gap")(x_conv)
    x_dense = layers.Dense(256, activation='relu', name="dense")(x_gap)
    x_drop = layers.Dropout(0.4, name="dropout")(x_dense)
    predictions = layers.Dense(4, activation='softmax', name="dense_1")(x_drop)

    # We use a standard single-output model for training to avoid Keras multi-output loss errors
    train_model = models.Model(inputs=inputs, outputs=predictions)

    # 3. Compile with a very small learning rate for fine-tuning
    print("\n3. Compiling model with low learning rate for fine-tuning...")
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train the Model
    print("\n4. Starting Training (This may take a while!)...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        )
    ]

    train_model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        epochs=30,
        callbacks=callbacks
    )

    # 5. Extract weights for Grad-CAM
    print("\n5. Extracting weights for Grad-CAM...")
    os.makedirs("models", exist_ok=True)
    w1, b1 = train_model.get_layer("dense").get_weights()
    w2, b2 = train_model.get_layer("dense_1").get_weights()
    np.savez("models/efficientnet_weights.npz", w1=w1, b1=b1, w2=w2, b2=b2)
    print("Saved models/efficientnet_weights.npz")

    # Re-wrap the trained graph to include the convolutional map for Grad-CAM inference
    export_model = models.Model(inputs=inputs, outputs=[predictions, x_conv])

    # 6. Evaluate on Test Set
    print("\n6. Evaluating on Testing Dataset...")
    test_gen.reset()
    all_preds = []
    all_labels = []
    for _ in range(len(test_gen)):
        x_batch, y_batch = next(test_gen)
        # Using the export_model which returns [preds, features]
        preds, _ = export_model.predict(x_batch, verbose=0)
        all_preds.extend(preds)
        all_labels.extend(y_batch)
        if len(all_preds) >= test_gen.samples:
            break
            
    y_pred = np.argmax(all_preds, axis=1)
    y_true = np.argmax(all_labels, axis=1)

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=list(class_indices.keys())))

    # 7. Export to ONNX
    print("\n7. Exporting model to ONNX for Production...")
    input_signature = [tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(export_model, input_signature=input_signature, opset=13)
    onnx.save(onnx_model, "models/efficientnet_best.onnx")
    print("SUCCESS! Saved models/efficientnet_best.onnx!")
    print("You can now commit and push the new models/ folder to GitHub to deploy it.")

if __name__ == "__main__":
    train_fine_tuned_model()
