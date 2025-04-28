
# Eye Disease Detection - ECSA-Net

This repository contains the implementation of **ECSA-Net (Efficient Channel-Spatial Attention Network)**, a lightweight deep learning model for **eye disease classification**.  
The model combines channel and spatial attention mechanisms with convolutional neural networks to enhance feature extraction and classification accuracy while maintaining computational efficiency.

## 🔥 Project Structure

```
├── data_preprocessing.py        # Data loading and preprocessing
├── attention_modules.py         # Channel and Spatial Attention modules
├── ecsa_block.py                 # ECSA block combining attention modules
├── ecsa_net_model.py             # ECSA-Net model architecture
├── train_model.py                # Training the model
├── evaluate_model.py             # Evaluating the model
└── README.md                     # Project documentation
```

## 📦 Dataset Sources

- **Bajwa Hospital Multi Eye Disease Dataset:**  
  [Mendeley Link](https://data.mendeley.com/datasets/rgwpd4m785/3)

- **Eye Disease Image Dataset:**  
  [Mendeley Link](https://data.mendeley.com/datasets/s9bfhswzjb/1)

## 🚀 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/knowledge4every1/Eye-Disease-Detection.git
    cd Eye-Disease-Detection
    ```

2. Install required packages:
    ```bash
    pip install tensorflow scikit-learn matplotlib
    ```

3. Prepare the datasets and organize them into `/train` and `/val` folders.

4. Train the model:
    ```bash
    python train_model.py
    ```

5. Evaluate the model:
    ```bash
    python evaluate_model.py
    ```

## ⚡ Model Highlights

- Efficient use of Channel-Spatial Attention.
- Lightweight architecture (only ~0.56 million parameters).
- Superior accuracy compared to heavier pretrained models (VGG16, ResNet50).
- Suitable for real-world and clinical deployments.

## 📈 Results Summary

| Dataset | Accuracy | F1-Score |
|:--------|:---------|:---------|
| Bajwa Multi Eye Disease | 60.00% | 52% |
| Eye Disease Image Dataset | 69.92% | 70% |

## ✨ Future Work

- Incorporating more diverse datasets.
- Applying data augmentation and self-supervised learning.
- Optimizing the model for mobile and edge devices.

## 📃 License

This project is licensed under the MIT License.

---

**Developed with ❤️ for the Eye Disease Detection community.**
