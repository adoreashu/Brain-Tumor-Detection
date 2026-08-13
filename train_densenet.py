import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
import tf2onnx
import onnx
from sklearn.metrics import classification_report, confusion_matrix

def train_fine_tuned_densenet():
    print("=========================================================")
    print("Starting Extreme Fine-Tuning of DenseNet121")
    print("   (Aiming for 99.999% Accuracy with Merged Datasets)")
    print("=========================================================\n")

    # 1. Load Data with strong Augmentation
    print("1. Loading and augmenting the massive merged dataset...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.15 # We use 15% of the massive train set for validation during training
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    dataset_dir = "dataset_combined"
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

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
    print("\n2. Building DenseNet121 Architecture...")
    inputs = layers.Input(shape=(224, 224, 3), name="input")
    x = inputs * 255.0 
    
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=x)
    
    # Unfreeze the top 40 layers for extreme fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    x_conv = base_model.output
    x_gap = layers.GlobalAveragePooling2D(name="gap")(x_conv)
    x_dense = layers.Dense(256, activation='relu', name="dense")(x_gap)
    x_drop = layers.Dropout(0.4, name="dropout")(x_dense)
    predictions = layers.Dense(4, activation='softmax', name="dense_1")(x_drop)

    train_model = models.Model(inputs=inputs, outputs=predictions)

    # 3. Compile
    print("\n3. Compiling model with low learning rate...")
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train the Model
    print("\n4. Starting Training on the Mega-Dataset...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=7, restore_best_weights=True
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
    np.savez("models/densenet_weights.npz", w1=w1, b1=b1, w2=w2, b2=b2)
    print("Saved models/densenet_weights.npz")

    export_model = models.Model(inputs=inputs, outputs=[predictions, x_conv])

    # 6. Evaluate on Test Set
    print("\n6. Evaluating on Mega Testing Dataset...")
    test_gen.reset()
    all_preds = []
    all_labels = []
    for _ in range(len(test_gen)):
        x_batch, y_batch = next(test_gen)
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
    print("\n7. Exporting DenseNet121 model to ONNX for Production...")
    input_signature = [tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(export_model, input_signature=input_signature, opset=13)
    onnx.save(onnx_model, "models/densenet_best.onnx")
    print("SUCCESS! Saved models/densenet_best.onnx!")

if __name__ == "__main__":
    train_fine_tuned_densenet()
