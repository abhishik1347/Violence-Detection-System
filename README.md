# Violence Detection Using Hybrid Deep Learning (CNN + Transformer)

This project presents a hybrid deep learning model designed to detect violent behavior in images using a combination of **Convolutional Neural Networks (CNNs)** and **Transformer blocks**. It leverages the feature extraction capabilities of pre-trained **VGG16** and the global context learning of **multi-head attention**, offering a robust approach to visual violence detection.


## 🚀 Introduction

Violence detection in surveillance footage has become an essential component in public safety and monitoring systems. Traditional techniques often fall short due to manual analysis limitations. To overcome this, the project combines the local feature extraction power of CNNs with the contextual learning capability of Transformers.

---

## 🧠 Project Architecture

The hybrid model architecture includes:

- **Input Layer**: Image input resized to 224x224x3
- **CNN Block**: Pre-trained VGG16 (excluding top layers)
- **Pooling Layer**: Global Average Pooling
- **Dense Layer**: 256 neurons with ReLU
- **Reshape Layer**: Adjusts input for transformer
- **Transformer Block**: Includes multi-head self-attention, feedforward layers, and normalization
- **Output Layer**: Sigmoid activated dense layer for binary classification

---

## 📂 Dataset

- The dataset contains images labeled as `violent` or `non-violent`.
- Includes **data augmentation** and **SMOTE** for class balance.
- Images resized to **224x224**.
- Divided into **train, test, and validation** sets.

---

## 🔬 Methodology

1. **Data Preprocessing**:
   - Resizing
   - Normalization
   - Augmentation
   - SMOTE oversampling

2. **Model Design**:
   - VGG16 for deep feature extraction
   - Transformer for contextual encoding
   - Combined model for violence classification

3. **Training**:
   - Loss Function: Binary Crossentropy
   - Optimizer: Adam
   - Epochs: 10
   - Batch Size: 32

---

## 📊 Model Performance

- **Accuracy**: 78%
- **Precision**: 82%
- **Recall**: 74%
- **F1 Score**: 78%

The combination of CNN and Transformer enhances both spatial and contextual understanding, resulting in a balanced and generalizable model.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- imblearn (SMOTE)

---

## ▶️ How to Run

**Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/violence-detection-cnn-transformer.git
   cd violence-detection-cnn-transformer
   ```
## 🛠️ Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Training Script
``bash
python train_model.py
```

## 📈 Run Evaluation
``bash
python evaluate.py

```
## ✅ Results

The hybrid model demonstrated strong potential in violence classification tasks, particularly in surveillance images. The Transformer block helped the model focus on spatially important areas, improving performance.

---

## 📌 Conclusion

This project showcases the effectiveness of combining CNNs and Transformers for violence detection tasks. It outperforms traditional CNN-only models by integrating attention mechanisms that enhance contextual understanding. Future improvements may include real-time video integration and advanced temporal modeling using Vision Transformers (ViTs) or 3D CNNs.
