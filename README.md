# 🔢 Digit Recognition using CNN (MNIST)

A deep learning project that uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) from the **MNIST dataset**.  
Built using **TensorFlow + Keras**, this program trains a CNN, visualizes accuracy/loss graphs, and predicts a random digit from test data.

---

## 🚀 Features
- Loads MNIST dataset of 70,000 images  
- Preprocesses (normalize + reshape) for CNN  
- User input for **epochs** and **batch size**  
- CNN architecture with Conv → Pool → Dense → Dropout  
- Training accuracy & loss graphs  
- Random test digit prediction with visualization  
- Prints model summary & final accuracy  

---

## 📂 Project Structure
```
digit_recognition_cnn/
│── digit_recognition_cnn.py
│── README.md
└── requirements.txt (optional)
```

---

## 🧠 Model Architecture
```
Conv2D (32 filters, 3×3) + ReLU
MaxPooling2D
Conv2D (64 filters, 3×3) + ReLU
MaxPooling2D
Flatten
Dense (128 units, ReLU)
Dropout (0.3)
Dense (10 units, Softmax)
```

Optimizer: **Adam**  
Loss: **Sparse Categorical Crossentropy**  
Metric: **Accuracy**

---

## 🛠️ How to Run

### 1️⃣ Install dependencies
```bash
pip install tensorflow matplotlib numpy
```

### 2️⃣ Run the script
```bash
python digit_recognition_cnn.py
```

### 3️⃣ Enter your training settings
Example:
```
Epochs: 5
Batch size: 64
```

---

## 📊 Training Visualizations
The script automatically displays:

- 📈 Accuracy vs Epochs  
- 📉 Loss vs Epochs  

*(Graphs appear after training completes)*

---

## 🔍 Sample Prediction Output
At the end of training, the script:

- Selects a random test image  
- Displays the image  
- Predicts the digit  

Example:
```
Actual Digit: 7
Predicted Digit: 7 ✔
```

---

## 📈 Model Performance
Typical results (5–10 epochs):

```
Test Accuracy: 97% – 99%
Test Loss: ~0.05
```

---

## 📦 Optional: requirements.txt
```
tensorflow
matplotlib
numpy
```

---

## 🧑‍💻 Author
**Devadharshan B**  
Cybersecurity | Python | Machine Learning | AI Enthusiast  

⭐ *If you like this project, please give it a star!*

