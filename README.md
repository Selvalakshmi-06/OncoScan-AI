**🧠 ONCOSCAN-AI**

**AI-Powered Brain MRI Tumor Detection with Explainable AI**

OncoScan-AI is an AI-based web application that classifies Brain MRI images into four categories using EfficientNetB0 and provides a Grad-CAM visualization to explain the model's prediction.

**🌐 LIVE DEMO**

https://oncoscan-ai-xgqbz7wffdpsrfkfglwbtn.streamlit.app/

**✨ FEATURES**

- Brain MRI tumor classification
- EfficientNetB0 deep learning model
- Transfer Learning with ImageNet
- Four-class tumor classification
- Confidence score for predictions
- Class probability visualization
- Grad-CAM explainability
- Interactive web interface
- Streamlit deployment

**🧠 SUPPORTED CLASSES**

- Glioma
- Meningioma
- No Tumor
- Pituitary

**🔄 PROJECT WORKFLOW**

Brain MRI Image  
↓  
Image Preprocessing  
↓  
Resize to 224 × 224  
↓  
EfficientNetB0  
↓  
Tumor Classification  
↓  
Prediction + Confidence  
↓  
Grad-CAM  
↓  
Visual Explanation

**🤖 MODEL**

The project uses EfficientNetB0 with Transfer Learning.

EfficientNetB0 is used as the main deep learning architecture because it provides a good balance between accuracy and computational efficiency.

The model was initialized with ImageNet-pretrained knowledge and trained for Brain MRI tumor classification.

**🔍 EXPLAINABLE AI**

OncoScan-AI uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the important regions of the MRI image that influenced the model's prediction.

This helps make the AI prediction easier to understand.

**📊 PERFORMANCE**

Test Accuracy: 84.25%

Test Loss: 0.4667

Test Dataset: 1,600 images

Class-wise Recall:

- Glioma: 65.25%
- Meningioma: 74.00%
- No Tumor: 99.00%
- Pituitary: 98.75%

**📁 DATASET**

Training images: 5,600

Testing images: 1,600

The dataset contains four classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary

**🛠️ TECHNOLOGIES**

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- Flask
- Streamlit
- HTML
- CSS
- JavaScript
- Git & GitHub

**📂 PROJECT STRUCTURE**

OncoScan-AI/

├── backend/  
│   └── app.py

├── frontend/  
│   ├── index.html  
│   ├── script.js  
│   └── style.css

├── models/  
│   ├── oncoscan_efficientnetb0.keras  
│   └── training_history.json

├── outputs/  
│   └── gradcam/

├── src/  
│   ├── evaluation/  
│   │   └── evaluate.py  
│   ├── explainability/  
│   │   └── gradcam.py  
│   ├── preprocessing/  
│   │   └── data_loader.py  
│   └── training/  
│       └── train.py

├── streamlit_app.py  
├── requirements.txt  
├── Dockerfile  
├── .gitignore  
└── README.md

**▶️ RUN LOCALLY**

Clone the repository:

git clone https://github.com/Selvalakshmi-06/OncoScan-AI.git

Go to the project folder:

cd OncoScan-AI

Create and activate a virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install the requirements:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run streamlit_app.py

The application will open in your browser.

**🚀 DEPLOYMENT**

The application is deployed using Streamlit Community Cloud.

Live application:

https://oncoscan-ai-xgqbz7wffdpsrfkfglwbtn.streamlit.app/

**📸 APPLICATION**

The application allows users to:

1. Upload a Brain MRI image.
2. Analyze the image.
3. View the predicted tumor class.
4. View prediction confidence.
5. View class probabilities.
6. View the Grad-CAM explanation.

**⚠️ DISCLAIMER**

OncoScan-AI is an educational and research project. It is not intended to replace professional medical diagnosis or clinical decision-making.

**👩‍💻 DEVELOPER**

Selvalakshmi  
B.Tech Artificial Intelligence and Data Science

**🔗 PROJECT LINKS**

GitHub:

https://github.com/Selvalakshmi-06/OncoScan-AI

Live Demo:

https://oncoscan-ai-xgqbz7wffdpsrfkfglwbtn.streamlit.app/

**Predict • Explain • Understand**
