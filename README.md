# 🌿 AcaciaNet: Invasive Species Detection

> **Built in 24 hours at JunctionX Lisbon Hackathon 🇵🇹**

AcaciaNet is a Deep Learning pipeline designed to detect invasive Acacia trees in Portugal using high-resolution satellite imagery. This project automates the identification process, helping environmental agencies monitor and manage invasive vegetation spread efficiently.

## 🚀 Overview

Portugal faces a significant challenge with invasive Acacia species disrupting local ecosystems. Manual mapping is slow and expensive. **AcaciaNet** leverages Computer Vision (CNNs) and Geospatial Data Engineering to solve this by:

1. **Processing** massive satellite GeoTIFFs into manageable datasets.
2. **Training** a specialized CNN model with class-imbalance handling.
3. **Generating** precise segmentation masks for affected areas.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `preprocess.py` | **Data Ingestion:** Reads satellite imagery and `.gpkg` ground truth vectors. Tiles the map into 50m chips and balances the dataset (handling the scarcity of Acacia samples). |
| `train.py` | **Model Training:** Defines a Custom CNN architecture using TensorFlow/Keras. Includes class weighting, data normalization, and training history visualization. |
| `detect.py` | **Inference:** Applies the trained model (`.h5`) to new satellite imagery using a sliding window approach to generate a binary classification mask (GeoTIFF). |

## 🛠️ Tech Stack

* **Core:** Python 3.9+
* **Deep Learning:** TensorFlow, Keras
* **Geospatial:** Rasterio, Geopandas, Shapely
* **Data Processing:** NumPy, Scikit-learn, TQDM

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/muhibtaha/junctionX.git
cd junctionX
```

2. Install dependencies:
```bash
pip install tensorflow rasterio geopandas numpy scikit-learn matplotlib tqdm
```

## 🏃 Usage

### 1. Data Preparation
Place your satellite image (`.tiff`) and ground truth vector (`.gpkg`) in the root directory.
```bash
python preprocess.py
# Output: image_chips_labels_50m_balanced.npz
```

### 2. Training
Train the Convolutional Neural Network.
```bash
python train.py
# Output: acacia_detector_balanced.h5 and training accuracy plots
```

### 3. Detection / Inference
Generate a map of invasive species on a full satellite image.
```bash
python detect.py
# Output: acacia_mask_final.tiff and visualization PNGs
```

## 🏆 Hackathon Context

This solution was prototyped and deployed in under 24 hours during the JunctionX Lisbon hackathon. It addresses the "Nature & Sustainability" track by providing a scalable tool for environmental protection.

## 📜 License

MIT License
