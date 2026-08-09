"""
Master training script for Brain Tumor Detection models.
"""
import os
import argparse
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from preprocessing.data_loader import get_train_val_test_data
from training.custom_cnn import build_custom_cnn
from training.vgg16_transfer import build_vgg16_model
from training.resnet50_transfer import build_resnet50_model
from utils.constants import MODEL_DIR, EVAL_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_history(history: tf.keras.callbacks.History, model_name: str) -> None:
    """Plots and saves training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'{model_name} Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    save_path = os.path.join(EVAL_DIR, f'{model_name}_training_history.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved training history plot to {save_path}")

def train_model(model_type: str, epochs: int, lr: float) -> None:
    """Trains a specified model."""
    logging.info(f"Starting training for {model_type}...")
    
    train_gen, val_gen, _ = get_train_val_test_data()
    
    if model_type == 'custom':
        model = build_custom_cnn()
    elif model_type == 'vgg16':
        model = build_vgg16_model()
    elif model_type == 'resnet50':
        model = build_resnet50_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model_path = os.path.join(MODEL_DIR, f'{model_type}_best.h5')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True),
        TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs', model_type))
    ]
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    plot_history(history, model_type)
    logging.info(f"Training completed for {model_type}. Best model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Brain Tumor Detection Models")
    parser.add_argument('--model', type=str, choices=['custom', 'vgg16', 'resnet50', 'all'], default='custom',
                        help='Model to train (custom, vgg16, resnet50, or all)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    if args.model == 'all':
        for m in ['custom', 'vgg16', 'resnet50']:
            train_model(m, args.epochs, args.lr)
    else:
        train_model(args.model, args.epochs, args.lr)
