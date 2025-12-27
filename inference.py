import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# --- SETTINGS ---
MODEL_FILE = "acacia_detector_balanced.h5"
INPUT_IMAGE_FILE = "deneme3.tiff"
OUTPUT_MASK_FILE = "acacia_mask_final.tiff"

def find_optimal_threshold(model, X_val, y_val):
    """Find optimal threshold"""
    print("🎯 Calculating optimal threshold...")
    y_pred_proba = model.predict(X_val, verbose=0)
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"✅ Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold

print("🧠 Loading trained model...")
try:
    model = keras.models.load_model(MODEL_FILE)
    input_shape = model.input_shape
    print(f"✅ Model loaded successfully. Input shape: {input_shape[1:]}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Prepare validation set for optimal threshold
print("📊 Preparing validation set...")
try:
    data = np.load("image_chips_labels_50m_balanced.npz")
    X_val = data['X'][:1000].astype('float32')  # First 1000 samples
    y_val = data['y'][:1000]
    
    if np.max(X_val) > 1.0:
        X_val = X_val / 10000.0
    
    optimal_threshold = find_optimal_threshold(model, X_val, y_val)
except:
    print("⚠️ Validation set not found, using default threshold: 0.5")
    optimal_threshold = 0.5

print(f"\n🖼️ Opening image to process: {INPUT_IMAGE_FILE}")
try:
    with rasterio.open(INPUT_IMAGE_FILE) as src:
        img = src.read()
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
    
    print(f"📷 Image Dimensions (Band,Y,X): {img.shape}")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    exit()

# Process image
img_processed = np.transpose(img, (1, 2, 0))  # (Y, X, Band)
img_normalized = img_processed.astype('float32')

# Normalization
if np.max(img_normalized) > 1.0:
    if np.max(img_normalized) <= 10000:
        img_normalized = img_normalized / 10000.0
    else:
        img_normalized = img_normalized / np.max(img_normalized)

print(f"🎛️  Normalization - Min: {np.min(img_normalized):.3f}, Max: {np.max(img_normalized):.3f}")

# Model parameters
CHIP_SIZE = input_shape[1]
stride = CHIP_SIZE  # No overlap
height, width, bands = img_normalized.shape

print(f"🔍 Splitting image {height}x{width} -> {CHIP_SIZE}x{CHIP_SIZE} chips...")
print(f"📐 Chip size: {CHIP_SIZE}x{CHIP_SIZE}, Stride: {stride}")

# Calculations
num_chips_y = (height - CHIP_SIZE) // stride + 1
num_chips_x = (width - CHIP_SIZE) // stride + 1
total_chips = num_chips_y * num_chips_x

print(f"🔢 Total number of chips: {num_chips_y} x {num_chips_x} = {total_chips}")

# Mask and prediction arrays
mask = np.zeros((height, width), dtype=np.uint8)
probability_map = np.zeros((height, width), dtype=np.float32)
count_map = np.zeros((height, width), dtype=np.uint8)  # Track how many times prediction was made

print("🎯 MAKING PREDICTIONS...")

chip_count = 0
predictions_list = []

for i in tqdm(range(0, height - CHIP_SIZE + 1, stride), desc="Rows"):
    for j in range(0, width - CHIP_SIZE + 1, stride):
        try:
            # Get chip
            chip = img_normalized[i:i+CHIP_SIZE, j:j+CHIP_SIZE, :]
            
            # Size check
            if chip.shape[0] != CHIP_SIZE or chip.shape[1] != CHIP_SIZE:
                continue
            
            # Add batch dimension and predict
            chip_batch = np.expand_dims(chip, axis=0)
            prediction = model.predict(chip_batch, verbose=0)[0][0]
            
            predictions_list.append(prediction)
            
            # Update probability map
            probability_map[i:i+CHIP_SIZE, j:j+CHIP_SIZE] += prediction
            count_map[i:i+CHIP_SIZE, j:j+CHIP_SIZE] += 1
            
            # Apply threshold for binary mask
            if prediction > optimal_threshold:
                mask[i:i+CHIP_SIZE, j:j+CHIP_SIZE] = 255
                
            chip_count += 1
            
        except Exception as e:
            continue

# Calculate average probability
probability_map = np.divide(probability_map, count_map, where=count_map>0)

print(f"\n📊 PREDICTION STATISTICS:")
predictions_array = np.array(predictions_list)
print(f"   Min Probability: {predictions_array.min():.4f}")
print(f"   Max Probability: {predictions_array.max():.4f}")
print(f"   Mean Probability: {predictions_array.mean():.4f}")
print(f"   Standard Deviation: {predictions_array.std():.4f}")
print(f"   Greater than {optimal_threshold}: {np.sum(predictions_array > optimal_threshold)} / {len(predictions_array)}")


#mask = 255 - mask
# Save mask
print(f"\n💾 Saving mask: {OUTPUT_MASK_FILE}")
meta.update({
    "count": 1,
    "dtype": "uint8",
    "compress": 'lzw'
})

try:
    with rasterio.open(OUTPUT_MASK_FILE, 'w', **meta) as dst:
        dst.write(mask, 1)
    print("✅ Mask saved successfully")
except Exception as e:
    print(f"❌ Error saving mask: {e}")

# Visualization
print("📊 Creating visualization...")
try:
    plt.figure(figsize=(20, 5))
    
    # 1. Original Image
    plt.subplot(1, 4, 1)
    # Show RGB channels (first 3 bands)
    display_img = img_normalized[..., :3]
    # Contrast enhancement
    p2, p98 = np.percentile(display_img, (2, 98))
    display_img_enhanced = np.clip((display_img - p2) / (p98 - p2), 0, 1)
    plt.imshow(display_img_enhanced)
    plt.title('Original Image (RGB)')
    plt.axis('off')
    
    # 2. Probability Map
    plt.subplot(1, 4, 2)
    plt.imshow(probability_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Acacia Probability')
    plt.title('Probability Map')
    plt.axis('off')
    
    # 3. Binary Mask
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Binary Mask (Threshold: {optimal_threshold:.2f})')
    plt.axis('off')
    
    # 4. Overlay
    plt.subplot(1, 4, 4)
    plt.imshow(display_img_enhanced)
    plt.imshow(mask, cmap='Reds', alpha=0.3)
    plt.title('Overlay (Red = Acacia)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization_final1.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'prediction_visualization_final1.png'")
    
except Exception as e:
    print(f"⚠️ Could not create visualization: {e}")

print(f"\n🎉 PREDICTION COMPLETED!")
print(f"📁 Mask file: {OUTPUT_MASK_FILE}")
print(f"📁 Visualization file: prediction_visualization_final2.png")
print(f"🎯 Threshold used: {optimal_threshold:.2f}")
