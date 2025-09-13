# 📚 Handwritten Digit Classifier

## 📖 Overview
This project predicts **handwritten digit classes (0-9)** using the **MNIST dataset** and a neural network built with **PyTorch**.  
It demonstrates a full machine learning pipeline from data loading to inference, including:

- 🧠 **Neural Network** with multiple hidden layers using **LeakyReLU** activation function and **Dropout**  
- ⚖️ **Cross-Entropy Loss** for multi-class classification
- 🚀 **Adam optimizer** for gradient updates 
- 🔀 **Mini-batch training** with `DataLoader`  
- 📊 **Train/Validation/Test split** for robust evaluation  
- 📈 **Live training & validation loss monitoring**  
- ✅ **Softmax activation** on the output for probability distribution across 10 classes

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **pandas** – data handling  
- **matplotlib** – loss visualization  
- **pickle** – saving/loading normalization params and trained model

---

## ⚙️ Requirements

- Python **3.13+**
- Recommended editor: **VS Code**

---

## 📦 Installation

- Clone the repository
```bash
git clone https://github.com/hurkanugur/Handwritten-Digit-Classifier.git
```

- Navigate to the `Handwritten-Digit-Classifier` directory
```bash
cd Car_Price_Predictor
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Navigate to the `Handwritten-Digit-Classifier/src` directory
```bash
cd src
```

---

## 🔧 Setup Python Environment in VS Code

1. `View → Command Palette → Python: Create Environment`  
2. Choose **Venv** and your **Python version**  
3. Select **requirements.txt** to install dependencies  
4. Click **OK**

---

## 📂 Project Structure

```bash
data/
└── MNIST                             # MNIST dataset (raw greyscale images and labels)
└── test_data                         # Sample images for inference

model/
└── mnist_digit_classifier.pth        # Trained model (after training)

src/
├── config.py                         # Paths, hyperparameters, split ratios
├── dataset.py                        # Data loading & preprocessing
├── main_train.py                     # Training & model saving
├── main_inference.py                 # Inference pipeline
├── model.py                          # Neural network definition
├── visualize.py                      # Training/validation plots

requirements.txt                      # Python dependencies
```

---

## 📂 Model Architecture

```bash
Input(28x28) → Flatten()
             → Linear(512)  → ReLU   → Dropout(0.2)
             → Linear(256)  → ReLU   → Dropout(0.1)
             → Linear(10)   → Softmax(Output)

```

---

## 📂 Train the Model
```bash
python main_train.py
```
or
```bash
python3 main_train.py
```

---

## 📂 Run Predictions on Real Data
```bash
python main_inference.py
```
or
```bash
python3 main_inference.py
```
