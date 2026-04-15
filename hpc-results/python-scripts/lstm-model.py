# Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import urllib.request
import zipfile
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv1D,
                                      MaxPooling1D, LSTM, Reshape,
                                      BatchNormalization, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Fixed seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Output directories
FIGURES_DIR = './figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f'TensorFlow version : {tf.__version__}')
print(f'NumPy version      : {np.__version__}')
print('All packages loaded successfully!')


# Load data
DATA_URL     = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
DATA_DIR     = './data'
ZIP_PATH     = os.path.join(DATA_DIR, 'UCI_HAR_Dataset.zip')
EXTRACT_PATH = os.path.join(DATA_DIR, 'UCI HAR Dataset')
BASE = './data/UCI HAR Dataset'
SIGNAL_NAMES = [
    'Body Acc X', 'Body Acc Y', 'Body Acc Z',
    'Body Gyro X', 'Body Gyro Y', 'Body Gyro Z',
    'Total Acc X', 'Total Acc Y', 'Total Acc Z'
]

CLASS_NAMES = [
    'Walking', 'Walking Upstairs', 'Walking Downstairs',
    'Sitting', 'Standing', 'Laying'
]

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(BASE):
    print('Downloading UCI HAR Dataset...')
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print('Download complete. Extracting...')
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(DATA_DIR)
    print('Extraction complete!')
else:
    print('Dataset already exists locally.')
    
    
# Load signals, labels and subjects
def load_signals(subset, base_path):
    """
    Loading all 9 raw inertial sensor signal files for a given subset (train/test).
    Returns array of shape (n_samples, 128 timesteps, 9 channels).
    """
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    signals = []
    for sig in signal_types:
        path = os.path.join(base_path, subset, 'Inertial Signals',
                            f'{sig}_{subset}.txt')
        signals.append(
            pd.read_csv(path, sep=r'\s+', header=None, engine='python').values
        )
    return np.transpose(np.array(signals), (1, 2, 0))

def load_labels(subset, base_path):
    """Loading integer activity labels (1–6) for a given subset."""
    path = os.path.join(base_path, subset, f'y_{subset}.txt')
    return pd.read_csv(path, header=None).values.ravel()

def load_subjects(subset, base_path):
    """Loading subject IDs for cross-user generalization analysis."""
    path = os.path.join(base_path, subset, f'subject_{subset}.txt')
    return pd.read_csv(path, header=None).values.ravel()

X_train_raw    = load_signals('train', BASE)
X_test_raw     = load_signals('test',  BASE)
y_train_raw    = load_labels('train',  BASE)
y_test_raw     = load_labels('test',   BASE)
subjects_train = load_subjects('train', BASE)
subjects_test  = load_subjects('test',  BASE)

print(f'X_train shape          : {X_train_raw.shape}')  # (7352, 128, 9)
print(f'X_test  shape          : {X_test_raw.shape}')   # (2947, 128, 9)
print(f'Unique activity labels : {np.unique(y_train_raw)}')
print(f'Train subjects         : {np.unique(subjects_train)}')
print(f'Test subjects          : {np.unique(subjects_test)}')

pd.DataFrame(X_train_raw[0], columns=SIGNAL_NAMES).head()


# Preprocessing
# Missing Value Check
train_flat = X_train_raw.reshape(X_train_raw.shape[0], -1)
test_flat  = X_test_raw.reshape(X_test_raw.shape[0], -1)

missing_train = np.isnan(train_flat).sum()
missing_test  = np.isnan(test_flat).sum()
print(f'Missing values in training set : {missing_train}')
print(f'Missing values in test set     : {missing_test}')
print('=> No imputation needed.\n' if missing_train == 0 and missing_test == 0
      else '=> Missing values detected — imputation required.\n')

#Scaling Check 
print(f'Training set value range : [{X_train_raw.min():.4f}, {X_train_raw.max():.4f}]')
print(f'Test set value range     : [{X_test_raw.min():.4f},  {X_test_raw.max():.4f}]')
print('=> Z-score normalization will be applied per channel.\n')

# Label Encoding Check 
print(f'Unique raw labels (train) : {np.unique(y_train_raw)}  (1-based integers)')
print(f'Unique raw labels (test)  : {np.unique(y_test_raw)}')
print('=> Labels will be encoded to 0-based integers and one-hot encoded for Keras.\n')

# Z-score Normalization (using training set statistics only)
mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
std  = X_train_raw.std(axis=(0, 1),  keepdims=True) + 1e-8

X_train_norm = (X_train_raw - mean) / std
X_test_norm  = (X_test_raw  - mean) / std

# Label Encoding: 1–6 → 0–5 → one-hot 
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_raw)
y_test_enc  = le.transform(y_test_raw)
NUM_CLASSES = len(le.classes_)

y_train_oh = to_categorical(y_train_enc, NUM_CLASSES)
y_test_oh  = to_categorical(y_test_enc,  NUM_CLASSES)

# Validation Split: stratified 85-15 
X_train, X_val, y_train, y_val = train_test_split(
    X_train_norm, y_train_oh,
    test_size=0.15, random_state=42, stratify=y_train_enc
)

print(f'Number of classes : {NUM_CLASSES}')
print(f'Train set         : {X_train.shape[0]} samples')
print(f'Validation set    : {X_val.shape[0]} samples')
print(f'Test set          : {X_test_norm.shape[0]} samples')
print(f'Input shape       : {X_train.shape[1:]}  (128 timesteps x 9 channels)')


# Define model parameters
TIMESTEPS  = X_train.shape[1]   # 128
FEATURES   = X_train.shape[2]   # 9
EPOCHS     = 50
BATCH_SIZE = 32

def get_callbacks():
    """Early stopping + learning rate reduction callbacks shared across all models."""
    return [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, verbose=0)
    ]


# Define LSTM model
def build_lstm(timesteps, features, num_classes):
    model = Sequential([
        Input(shape=(timesteps, features)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ], name='LSTM')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

lstm_model = build_lstm(TIMESTEPS, FEATURES, NUM_CLASSES)
lstm_model.summary()


# Train LSTM Model
print('=' * 55)
print('Training LSTM...')
print('=' * 55)
lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=get_callbacks(),
    verbose=1
)

# Evaluate on the test set
def evaluate_model(model, X_test, y_test_enc, model_name):
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    acc  = accuracy_score(y_test_enc,  y_pred)
    prec = precision_score(y_test_enc, y_pred, average='weighted')
    rec  = recall_score(y_test_enc,    y_pred, average='weighted')
    f1   = f1_score(y_test_enc,        y_pred, average='weighted')

    print(f'\n{"-" * 55}')
    print(f'  Model     : {model_name}')
    print(f'  Accuracy  : {acc:.4f}')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1-Score  : {f1:.4f}')
    print(f'{"-" * 55}')
    print(classification_report(y_test_enc, y_pred, target_names=CLASS_NAMES))

    return y_pred, {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

y_pred, metrics = evaluate_model(lstm_model, X_test_norm, y_test_enc, 'LSTM')