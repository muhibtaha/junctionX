import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import os

# --- SETTINGS ---
NPZ_FILE = "image_chips_labels_50m_balanced.npz"
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32

print("🔄 MODEL TRAINING WITH BALANCED DATA...")

# --- 1. Load Data ---
print(f"📦 Loading data: {NPZ_FILE}")
try:
    data = np.load(NPZ_FILE)
    X = data['X']
    y = data['y']
    print(f"✅ Data shape: {X.shape}")
    print(f"📊 Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
except Exception as e:
    print(f"❌ ERROR: Loading NPZ file: {e}")
    exit()

# --- 2. Normalize Data ---
print("\n🎛️  Applying normalization...")
X = X.astype('float32')

if np.max(X) > 1.0:
    if np.max(X) <= 10000:
        X = X / 10000.0
        print("✅ Normalization by dividing by 10000")
    else:
        X = X / np.max(X)
        print("✅ Normalization by dividing by max value")
else:
    print("✅ Already normalized")

print(f"📊 Normalized Min: {np.min(X):.3f}, Max: {np.max(X):.3f}")

# --- 3. Split Data into Train and Test Sets ---
print(f"\n🔀 Splitting data: {int((1-TEST_SPLIT_RATIO)*100)}% train, {int(TEST_SPLIT_RATIO*100)}% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SPLIT_RATIO,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"📊 Training set: {X_train.shape}")
print(f"📊 Test set: {X_test.shape}")

# --- 4. Define CNN Model ---
print("\n🧠 Creating model...")
input_shape = X_train.shape[1:]

model = keras.Sequential([
    keras.Input(shape=input_shape),
    
    # 1. Convolutional Layer
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    # 2. Convolutional Layer
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    # Global Pooling
    keras.layers.GlobalAveragePooling2D(),
    
    # Fully Connected Layers
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    
    # Output Layer
    keras.layers.Dense(1, activation='sigmoid')
])

# --- 5. Calculate Class Weights ---
print("\n⚖️ Calculating class weights...")
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"📊 Class weights: {class_weight_dict}")

# --- 6. Compile Model ---
print("\n🔧 Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

model.summary()

# --- 7. Define Callbacks ---
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        monitor='val_loss'
    )
]

# --- 8. Train Model ---
print("\n🎯 MODEL TRAINING STARTING...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# --- 9. Evaluate Model Performance ---
print("\n📊 EVALUATING MODEL...")
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test, verbose=0)

print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Test Precision: {test_precision:.4f}")
print(f"✅ Test Recall: {test_recall:.4f}")
print(f"✅ Test AUC: {test_auc:.4f}")

# --- 10. Detailed Metrics ---
print("\n📈 CLASSIFICATION REPORT:")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype("int32")

print(classification_report(y_test, y_pred, target_names=["Absent (0)", "Present (1)"]))

print("\n🔢 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- 11. Calculate PDF Score ---
print("\n📝 PDF SCORE CALCULATION:")
TN, FP, FN, TP = cm.ravel()

pdf_score = 200 * (TP * 0.5 - FN * 0.3 - FP * 0.2) / (TP + FN + FP)
print(f"📊 PDF Score: {pdf_score:.2f}")

# --- 12. Visualize Training History ---
try:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Precision & Recall')
    
    plt.tight_layout()
    plt.savefig("training_history_balanced.png", dpi=150, bbox_inches='tight')
    print("📊 Training history saved as 'training_history_balanced.png'.")
except Exception as e:
    print(f"⚠️ Could not save plot: {e}")

# --- 13. Save Model ---
model.save("acacia_detector_balanced.h5")
print("💾 Model saved as 'acacia_detector_balanced.h5'.")

print("\n🎉 ALL PROCESSES COMPLETED!")
