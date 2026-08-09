"""
Model evaluation utilities.
"""
import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model
from preprocessing.data_loader import get_train_val_test_data
from utils.constants import MODEL_DIR, EVAL_DIR, CLASS_LABELS
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Plots and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(EVAL_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved confusion matrix to {save_path}")

def plot_roc_curves(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, model_name: str) -> None:
    """Plots and saves one-vs-all ROC curves with AUC."""
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(CLASS_LABELS):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curves')
    plt.legend(loc="lower right")
    save_path = os.path.join(EVAL_DIR, f'{model_name}_roc_curves.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved ROC curves to {save_path}")

def evaluate_model(model_name: str) -> dict:
    """Evaluates a specific model on the test set."""
    model_path = os.path.join(MODEL_DIR, f'{model_name}_best.h5')
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return {}
        
    model = load_model(model_path)
    _, _, test_gen = get_train_val_test_data()
    
    logging.info(f"Evaluating {model_name} on test data...")
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    y_true_onehot = np.eye(len(CLASS_LABELS))[y_true]
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    report_path = os.path.join(EVAL_DIR, f'{model_name}_classification_report.csv')
    df_report.to_csv(report_path)
    logging.info(f"Classification report saved to {report_path}")
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))
    
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc_curves(y_true_onehot, y_pred_probs, model_name)
    
    return {'accuracy': report['accuracy'], 'macro_f1': report['macro avg']['f1-score']}

def compare_models(results: dict) -> None:
    """Creates a side-by-side comparison table for all evaluated models."""
    if not results:
        return
        
    df = pd.DataFrame(results).transpose()
    comp_path = os.path.join(EVAL_DIR, 'model_comparison.csv')
    df.to_csv(comp_path)
    logging.info(f"\nModel Comparison saved to {comp_path}")
    print("\nModel Comparison:")
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Detection Models")
    parser.add_argument('--model', type=str, choices=['custom', 'vgg16', 'resnet50', 'all'], default='all',
                        help='Model to evaluate')
    args = parser.parse_args()
    
    results = {}
    if args.model == 'all':
        for m in ['custom', 'vgg16', 'resnet50']:
            res = evaluate_model(m)
            if res:
                results[m] = res
        compare_models(results)
    else:
        evaluate_model(args.model)
