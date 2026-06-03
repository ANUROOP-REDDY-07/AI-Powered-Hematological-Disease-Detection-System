# 🩸 AI-Powered Hematological Disease Detection System

A Deep Learning-based system for automated detection of hematological diseases from microscopic blood smear images.

This project uses **EfficientNetB0**, **Transfer Learning**, and **Explainable AI (Grad-CAM)** to classify blood smear images into four categories:

* Leukemia
* Malaria
* Sickle Cell Anemia
* Normal

The objective is to assist in the early screening of blood-related diseases by providing fast, accurate, and interpretable predictions.

---

## 📌 Project Motivation

Microscopic examination of blood smear images is a common method for diagnosing hematological diseases. However, manual analysis can be time-consuming and depends heavily on expert knowledge.

This project explores how Deep Learning can automatically identify disease-specific patterns in blood cells and support medical diagnosis through computer vision techniques.

---


## 🏗️ Project Workflow

```text
Blood Smear Image
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
EfficientNetB0
        ↓
Feature Extraction
        ↓
Disease Classification
        ↓
Grad-CAM Visualization
        ↓
Final Prediction
```

---

## 🗂️ Dataset

This project combines multiple publicly available datasets from Kaggle.

### Malaria Dataset

* Source: Cell Images for Detecting Malaria
* Classes Used:

  * Parasitized
  * Uninfected

### Leukemia Dataset

* Source: Leukemia Classification Dataset
* Class Used:

  * Leukemia

### Sickle Cell Dataset

* Source: Sickle Cell Disease Dataset
* Class Used:

  * Sickle Cell Anemia

### Final Classes

| Class    | Description                   |
| -------- | ----------------------------- |
| Leukemia | Abnormal white blood cells    |
| Malaria  | Parasite-infected blood cells |
| Sickle   | Sickle-shaped red blood cells |
| Normal   | Healthy blood cells           |

**Total Dataset Size:** 12,000+ Images

---


## 🔬 Explainable AI (Grad-CAM)

To improve transparency, Grad-CAM was integrated into the system.

Grad-CAM highlights the image regions that contributed most to the final prediction.

Color Interpretation:

* 🔴 Red / Yellow → High Importance
* 🟢 Green → Moderate Importance
* 🔵 Blue → Low Importance

### Example

(Add Grad-CAM image here)

```markdown
![GradCAM](images/gradcam.png)
```

---

## 📁 Repository Structure

```text
AI-Powered-Hematological-Disease-Detection-System/
│
├── app.py
├── hematology_final_model_14.keras
├── requirements.txt
├── version2.ipynb
├── README.md
```

---

## ▶️ How to Run

### Clone the Repository

```bash
git clone https://github.com/ANUROOP-REDDY-07/AI-Powered-Hematological-Disease-Detection-System.git

cd AI-Powered-Hematological-Disease-Detection-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Application

```bash
streamlit run app.py
```

### Using the Application

1. Upload a blood smear image.
2. Wait for the model prediction.
3. View confidence scores.
4. Analyze Grad-CAM visualization.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* EfficientNetB0

### Data Science Libraries

* NumPy
* Scikit-Learn
* Matplotlib

### Deployment

* Streamlit

### Version Control

* Git
* GitHub

---



## 🔮 Future Improvements

* Ensemble Learning Models
* Cell Segmentation Before Classification
* Additional Blood Disease Categories
* Mobile Application Deployment
* Larger Clinical Datasets
* Real-Time Hospital Integration

---



### ⭐ If you found this project useful, consider giving it a star.
