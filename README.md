# Early Detection of Alzheimer’s Disease Using Deep Learning

An AI-based system for early detection and stage classification of Alzheimer’s disease using Brain MRI images and Convolutional Neural Networks (CNNs).

This project was developed as part of the Project Based Learning – II (PBL-II) course in the B.Tech Computer Science & Engineering program at Symbiosis Institute of Technology, Pune.


##  Project Overview

Alzheimer’s disease is a progressive neurodegenerative disorder that affects memory, cognition, and daily functioning. Early diagnosis is crucial for slowing disease progression and improving patient quality of life.

Traditional diagnostic approaches involving MRI or PET scans:
- Require expert interpretation
- Are time-consuming
- Can be costly
- May not be easily accessible in rural areas

This project proposes a deep learning–based image classification system that:
- Analyzes brain MRI scans
- Classifies them into four stages of Alzheimer’s disease
- Provides visual explainability using Grad-CAM
- Offers a simple web interface for predictions


##  Objectives

- Study Alzheimer’s disease progression using MRI scans
- Build a CNN model for multi-class classification
- Classify MRI images into:
  - Non-Demented
  - Very Mild Dementia
  - Mild Dementia
  - Moderate Dementia
- Apply image preprocessing techniques
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
- Provide explainable predictions using Grad-CAM
- Build a user interface for image upload and prediction

---

##  Key Features

- 🧠 MRI-based Alzheimer’s stage classification
- 🤖 CNN-based deep learning model
- 🖼 Image preprocessing (resizing & normalization)
- 🔥 Grad-CAM visual explanations
- 🌐 Flask-based local web application
- 📊 Four-class disease classification
- ⚕️ Designed as a decision-support tool (not a replacement for medical professionals)


## 🛠️ Technologies Used

### Programming & Frameworks
- Python 3.11
- TensorFlow (Keras API)
- Flask

### Image Processing & Visualization
- Pillow (PIL)
- Grad-CAM

### Development Tools
- Jupyter Notebook
- Visual Studio Code
- Git & GitHub


## 📂 Dataset

The dataset consists of labeled Brain MRI images categorized into four classes representing different stages of Alzheimer’s disease.

Each image represents a brain slice used for supervised learning.

> Note: Dataset is used strictly for academic and research purposes.


## ⚙️ Methodology

### 1️⃣ Data Collection & Understanding
- Studied MRI image distribution
- Analyzed class imbalance
- Observed disease characteristics

### 2️⃣ Image Preprocessing
- Grayscale conversion
- Image resizing to uniform dimensions
- Pixel normalization
- Channel duplication (where required)

### 3️⃣ Model Architecture
- Convolutional layers
- Max pooling layers
- Dropout layers
- Fully connected dense layers
- Softmax output layer

Optimizer: Adam  
Loss Function: Categorical Cross-Entropy  
Regularization: Dropout & Early Stopping  

### 4️⃣ Model Evaluation
- Accuracy
- Confusion Matrix
- Precision & Recall

### 5️⃣ Explainability
Grad-CAM heatmaps were generated to highlight important brain regions influencing the model’s predictions.

### 6️⃣ Deployment
A Flask-based local web application allows:
- MRI image upload
- Disease stage prediction
- Visualization of Grad-CAM results


## 📊 Results

- Achieved approximately 97% classification accuracy (local testing)
- Clear separation between dementia stages
- Grad-CAM visualizations aligned with neurologically relevant regions
- Demonstrated feasibility of AI-assisted early Alzheimer’s detection


## ⚠️ Limitations

- Performance may drop on low-quality MRI images
- Limited dataset diversity
- Not validated for real-world clinical use
- Cloud deployment challenges


## 🔮 Future Scope

- Train on larger and more diverse datasets
- Integrate multimodal medical data (MRI, PET, cognitive scores, genetics)
- Implement advanced architectures (ResNet, DenseNet, EfficientNet)
- Deploy on AWS / Azure / Google Cloud
- Enable mobile or edge-device healthcare solutions
- Implement secure authentication and encrypted medical data handling


## ▶️ How to Run the Project

1. Clone the repository
   ```
   git clone https://github.com/your-username/your-repository-name.git
   ```

2. Navigate to the project folder
   ```
   cd project-folder
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask app
   ```
   python gradcam_flask.py
   ```

5. Open browser and go to:
   ```
   http://127.0.0.1:5000/
   ```


## 📜 License

This project is developed strictly for academic and educational purposes.

It is not intended for clinical diagnosis or medical use.
