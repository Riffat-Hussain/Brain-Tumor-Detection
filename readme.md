# 🧠 Brain Tumor Detection Using Convolutional Neural Networks (CNN)

## Sherozgar AI & Machine Learning Training Project

### 👤 Author

**Rifat Hussain**

AI & Machine Learning Trainee

---

## 📌 Project Overview

This project was developed as the final practical project for the **Sherozgar AI & Machine Learning Training Program** under the guidance of **Ma'am Gulshan Melvi**.

The objective of this project is to detect the presence of brain tumors from MRI images using a **Convolutional Neural Network (CNN)** built with **TensorFlow** and **Keras**.

The model classifies MRI brain scans into two categories:

* 🟢 No Brain Tumor
* 🔴 Brain Tumor

---

## 🚀 Features

* Brain MRI Image Classification
* CNN-based Deep Learning Model
* Binary Classification (Tumor / No Tumor)
* Image Preprocessing
* Model Training and Evaluation
* Prediction on New MRI Images
* Saved Trained Model
* Training Accuracy & Loss Graphs

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* OpenCV
* Visual Studio Code
* Git
* GitHub

---

## 📂 Dataset

This project uses the **Brain Tumor MRI Dataset**.

The dataset is organized into:

```text
train/
   ├── yes/
   └── no/

test/
   ├── yes/
   └── no/
```

> **Note:** The dataset is not included in this repository because of GitHub file size limits. Download the dataset separately and place it inside the `dataset/Brain_Tumor_Datasets/` folder.

---

## 📁 Project Structure

```text
Brain-Tumor-Detection-CNN/
│
├── dataset/
│
├── models/
│   └── brain_tumor_model.keras
│
├── results/
│   └── training_graph.png
│
├── report/
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/Brain-Tumor-Detection-CNN.git
```

Move into the project folder:

```bash
cd Brain-Tumor-Detection-CNN
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

---

## 🔍 Predict an MRI Image

Place an MRI image in the project folder and rename it as:

```text
sample.jpg
```

Then run:

```bash
python predict.py
```

---

## 📊 Results

| Metric          |      Value |
| --------------- | ---------: |
| Training Images |       7012 |
| Testing Images  |       1752 |
| Epochs          |          5 |
| Test Accuracy   | **95.09%** |
| Test Loss       | **0.1407** |

---

## 📈 Training Graph

The training accuracy and loss graph is saved in:

```text
results/training_graph.png
```

---

## 🎯 Learning Outcomes

This project helped me gain practical experience in:

* Convolutional Neural Networks (CNNs)
* Medical Image Classification
* TensorFlow & Keras
* Image Preprocessing
* Model Training
* Model Evaluation
* Git & GitHub
* Python Programming

---

## 🔮 Future Improvements

* Use larger datasets
* Apply Data Augmentation
* Implement Transfer Learning (ResNet50, EfficientNet)
* Develop a Web Application
* Deploy the model online
* Support multiple tumor classes

---

## 🙏 Acknowledgement

I would like to sincerely thank **Ma'am Gulshan Melvi** for her guidance and support throughout the **Sherozgar AI & Machine Learning Training Program**. The knowledge and practical experience gained during this training made it possible to successfully complete this project.

---

## 📄 License

This project was developed for educational and training purposes as part of the Sherozgar AI & Machine Learning Training Program.
