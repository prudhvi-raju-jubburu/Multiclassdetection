# Multi-Class Urban Object Detection Using YOLOv8

This repository contains code and notebooks for training and evaluating a multi-class urban object detection model using **YOLOv8** on the BDD100K dataset.

## 📌 Project Overview
Urban object detection is critical for autonomous driving, smart city surveillance, and traffic monitoring. This project detects key urban objects:
- Person
- Rider
- Car
- Bus
- Truck
- Bike
- Motorbike
- Traffic Light
- Traffic Sign
- Train

## 📁 Repository Structure
```text
├── Multi_Class_Urban_Object_Detection.ipynb  # Complete Jupyter / Google Colab Notebook
├── train.py                                  # Training and validation script
├── dataset.py                                # Script to download BDD100K dataset via kagglehub
├── create_small_dataset.py                   # Helper script to sample subsets for training/val
├── data.yaml                                 # YOLO dataset configuration
├── requirements.txt                          # Project dependencies
└── README.md                                 # Project documentation
```

## 🚀 Quick Start

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/prudhvi-raju-jubburu/Multiclassdetection.git
cd Multiclassdetection
pip install -r requirements.txt
```

### 2. Download Dataset & Prepare Data
```bash
python dataset.py
python create_small_dataset.py
```

### 3. Model Training & Evaluation
```bash
python train.py
```

## 📊 Results & Visualization
The model leverages YOLOv8 architecture fine-tuned for high efficiency on urban traffic scenes.
