# 🔥 Wildfire Prediction from Satellite Imagery using CNNs

This project demonstrates a deep learning pipeline for **binary wildfire classification** from satellite images using:

- ✅ A **Custom Convolutional Neural Network (CNN)**
- 🚀 A **Transfer Learning model** based on **ResNet50**

We compare their performance using training metrics and evaluation visuals like confusion matrix, ROC curve, and prediction probability distributions.

---

## 📁 Project Structure

```plaintext
📦wildfire-prediction/
├── custom_cnn.py             # Defines custom CNN architecture
├── transfer_model.py         # Defines ResNet50-based transfer model
├── train.py                  # Handles training and callbacks
├── evaluate.py               # Evaluation logic and visualization
├── utils.py                  # Helper functions (data prep, plotting)
├── dataset/                  # Contains `train`, `val`, `test` image folders
├── README.md
└── images/                   # Visuals used in this README


---
```

### ⚙️ Transfer Learning with ResNet50

* Uses **ResNet50** pretrained on ImageNet (frozen base)
* Global Average Pooling + 2 Dense layers
* Final sigmoid layer for binary classification

```python
Model: "resnet50_transfer"
Total params: ~23.6M
Trainable params: ~650K
```

---

## 📊 Data Visualization

Below is a sample of images from both wildfire and nowildfire categories used for training:

![Training Samples](images/training_samples.png)

---

## 📈 Training History

We tracked **Accuracy**, **Loss**, **Precision**, and **Recall** for both models.

![Training Comparison](images/training_history_comparison.png)

---

## 📊 Evaluation

The models were evaluated on the test set using:

* Confusion Matrix
* ROC Curve (AUC)
* Prediction Probability Distribution

---

## 📊 Results

This section compares both models based on test set metrics and visualization outputs.

### 📌 Test Set Metrics

| Model                         | Accuracy  | Precision | Recall   | AUC (ROC) |
|------------------------------|-----------|-----------|----------|-----------|
| Custom CNN                   | 86.7%     | 84.2%     | 85.5%    | 0.89      |
| Transfer Learning (ResNet50) | **91.3%** | **89.8%** | **90.5%**| **0.94**  |

---

### 🛠️ Custom CNN Results

Below are the combined evaluation visualizations (Confusion Matrix, ROC Curve, and Prediction Distribution) for the custom CNN model:

![Custom CNN Evaluation](images/custom_cnn_results.png)

---

### ⚙️ ResNet50 Transfer Model Results

Below are the combined evaluation visualizations (Confusion Matrix, ROC Curve, and Prediction Distribution) for the custom  ResNet50 Transfer Model:

![Custom CNN Evaluation](images/pretrained_model_results.png)

> 🔍 As shown above, the ResNet50-based transfer model performs better across most evaluation metrics and shows clearer separation in predicted probabilities.

---


## 📌 Key Metrics (Test Set)

| Model                        | Accuracy  | Precision | Recall    | AUC (ROC) |
| ---------------------------- | --------- | --------- | --------- | --------- |
| Custom CNN                   | 86.7%     | 84.2%     | 85.5%     | 0.89      |
| Transfer Learning (ResNet50) | **91.3%** | **89.8%** | **90.5%** | **0.94**  |

> 🧠 **Observation**: Transfer learning outperforms the custom CNN in all key metrics with faster convergence and better generalization.

---

## 🧪 How to Run

1. Clone the repository:

```bash
git clone https://github.com/theUtkarshRa/wildfire-prediction.git
cd wildfire-prediction
```

2. Prepare the dataset structure:

```
dataset/
├── train/
│   ├── wildfire/
│   └── nowildfire/
├── val/
│   ├── wildfire/
│   └── nowildfire/
└── test/
    ├── wildfire/
    └── nowildfire/
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train and evaluate:

```bash
python train.py
python evaluate.py
```

---

## 🧾 References

* Jonnalagadda, A. V., et al. *Transfer Learning vs Custom CNN for Wildfire Detection*, arXiv 2024.
* Toan, T. N., et al. *Wildfire Detection with Deep Learning Models*, MDPI 2024.
* FIRMS – Fire Information for Resource Management System (NASA).

---

## 🙌 Acknowledgments

This project was part of a wildfire detection research initiative at \[Your Institution].

---

## 📄 License

MIT License

```
