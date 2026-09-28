"""
Fine-tune DistilBERT transformer for sentiment analysis.
Targets 80%+ accuracy on real dataset.

Run:
    python train_transformer.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

import config
from src.data.loader import load_real_dataset
from src.preprocessing.cleaner import preprocess_batch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    """Load and preprocess dataset."""
    if os.path.exists(config.RAW_DATASET_PATH):
        print(f"Loading dataset from {config.RAW_DATASET_PATH}")
        df = load_real_dataset()
    else:
        raise FileNotFoundError(f"Dataset not found at {config.RAW_DATASET_PATH}")
    
    print(f"Dataset size: {len(df)}")
    print(df[config.LABEL_COLUMN].value_counts())
    
    return df

def prepare_dataset(df):
    """Prepare dataset for transformer."""
    # Map labels to integers
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df['label'] = df[config.LABEL_COLUMN].map(label_map)
    
    # Preprocess text
    print("Preprocessing text...")
    df['text'] = preprocess_batch(df[config.TEXT_COLUMN])
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, 
        df['label'].values,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=df['label'].values
    )
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def tokenize_function(examples, tokenizer):
    """Tokenize texts."""
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

def main():
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    
    # Initialize tokenizer and model
    print("\nLoading DistilBERT tokenizer and model...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    ).to(device)
    
    # Prepare training dataset
    train_dataset = Dataset.from_dict({
        'text': X_train,
        'label': y_train
    })
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Prepare test dataset
    test_dataset = Dataset.from_dict({
        'text': X_test,
        'label': y_test
    })
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
    output_dir='./results_transformer',
    num_train_epochs=1,  # Reduced from 3
    per_device_train_batch_size=32,  # Increased from 16
    per_device_eval_batch_size=32,
    warmup_steps=50,
    weight_decay=0.01,
    eval_strategy="no",  # Skip eval during training
    save_strategy="epoch",
    learning_rate=2e-5,
    seed=config.RANDOM_STATE,
)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train
    print("\nTraining DistilBERT...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on test set...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*60)
    print("DISTILBERT FINE-TUNED RESULTS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*60)
    
    # Save model and tokenizer
    print("\nSaving model and tokenizer...")
    model.save_pretrained('models/distilbert_fine_tuned')
    tokenizer.save_pretrained('models/distilbert_fine_tuned')
    
    # Save results
    results = {
        'model': 'distilbert_fine_tuned',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }
    joblib.dump(results, 'models/results_distilbert.pkl')
    
    print("\n✓ Model saved to models/distilbert_fine_tuned/")
    print("✓ Results saved to models/results_distilbert.pkl")

if __name__ == "__main__":
    main()