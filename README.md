# 🐶🐱🧑 Custom CNN Image Classifier with Gradio Interface

## Overview

This project showcases a complete end-to-end image classification web app built with **Gradio** and powered by a **custom-trained Convolutional Neural Network (CNN)** using **PyTorch**. The model is designed to classify images into one of three categories:

- **Dog**
- **Cat**
- **Person**

The project highlights deep learning model training, PyTorch model integration, and user-friendly interface deployment with Gradio.

## Demo

Visit the following link to view the demo:
https://huggingface.co/spaces/nivakaran/classification-gradio-KNCVU

## 🔍 Key Highlights

- ✅ **Custom CNN architecture** developed from scratch
- 🧠 Model trained on a clean and curated dataset of animals and people
- ⚡ Lightning-fast predictions using optimized PyTorch model
- 🎨 Annotated output images showing the predicted label
- 🌐 Gradio web interface for instant, local testing

---

## 🚀 Tech Stack

- **Python 3.7+**
- **PyTorch** for custom CNN model training
- **Gradio** for UI/UX and interaction
- **Pillow** for image manipulation

---

## 📁 Project Structure

```
.
├── app.py                     # Main Gradio app script
├── core/
│   └── predict.py             # ImageClassifier logic: model loading & inference
├── model/
│   └── cnn_128_model-100.pth  # Trained PyTorch CNN model
├── requirements.txt           # Required dependencies
└── README.md                  # Documentation
```

---

## 🖥️ Running the App

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/image-classification-gradio.git
cd image-classification-gradio
```

### Step 2: Install required packages

```bash
pip install -r requirements.txt
```

### Step 3: Launch the Gradio app

```bash
python app.py
```

The app will be available at [http://127.0.0.1:7860](http://127.0.0.1:7860).

---

## 🧠 Model Architecture

This model is a **custom CNN** built and trained from scratch without relying on pretrained backbones. It consists of convolutional layers, ReLU activations, pooling layers, and fully connected layers optimized for classifying 128x128 RGB images.

- Input size: `3 x 128 x 128`
- Architecture: `Conv2D -> ReLU -> MaxPool -> ... -> FC -> Softmax`
- Output classes: `Cat`, `Dog`, `Person`

---

## 🛠 Functionality

The Gradio interface allows users to:

- Upload an image
- Get the predicted class (`Cat`, `Dog`, or `Person`)
- View a labeled version of the uploaded image with the prediction

---

## 📦 Dependencies

```
gradio
torch
torchvision
opencv-python
pillow
```

Install them via:

```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 Author

**Your Name**
Machine Learning Enthusiast | Full Stack Developer
[GitHub](https://github.com/yourgithub) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 💼 Why This Project Matters

This project demonstrates end-to-end ML deployment skills:

- Custom neural network design
- Model training and evaluation in PyTorch
- Real-time inference using Gradio
- Clean and interactive UI for non-technical users

It represents a strong foundation for scalable, production-ready ML applications with real-world user interaction.

---

## 📄 License

MIT License
