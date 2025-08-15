Micro-Doppler Based Target Classification
This repository contains the code and results for Micro-Doppler based target classification using various deep learning models. The models are trained and evaluated on the DIAT-RADSATNET dataset, which consists of 4849 images of Micro-Doppler signatures.

Introduction
The goal of this project is to accurately classify different targets based on their Micro-Doppler signatures. Micro-Doppler signatures provide detailed information about the motion of a target, making them valuable for classification tasks. We explore several state-of-the-art deep learning architectures to determine the most effective model for this application.

Dataset
The models in this project are trained on the DIAT-RADSATNET dataset, which contains 4849 images of Micro-Doppler signatures from six different classes:

3_long_blade_rotor

3_short_blade_rotor

Bird

Bird+mini-helicopter

drone

rc_plane

Models
The following deep learning models have been implemented and evaluated for this classification task:

MobileNetV2: A lightweight and efficient convolutional neural network (CNN) architecture designed for mobile and embedded vision applications.

Vision Mamba with ZSL and Cosine Loss: A vision model incorporating the Mamba state-space model, designed for zero-shot learning (ZSL) with a cosine loss function.

ResNet50 with Skip Connections: A 50-layer residual network that uses skip connections to allow for deeper architectures without degradation in performance.

ConvKAN: A convolutional network that utilizes Kolmogorov-Arnold Networks (KANs) for improved performance and interpretability.

VGG16: A 16-layer CNN architecture known for its simplicity and effectiveness in image classification tasks.

VGG19 with Attention: A 19-layer version of the VGG network, enhanced with an attention mechanism to focus on the most relevant parts of the image.

Performance Comparison
The following table summarizes the validation accuracies and training epochs of the different models on the DIAT-RADSATNET dataset:

<img width="1578" height="654" alt="image" src="https://github.com/user-attachments/assets/1d8efad0-e2ee-45bf-a54a-e08a7676cc00" />

Results and Analysis
Based on the evaluation, the Vision Mamba with ZSL model achieved the highest validation accuracy of 99.79%. The ResNet50 with Skip Connections and VGG19 with Attention models also performed exceptionally well, both achieving 99.28% accuracy. The ConvKAN model followed closely with 99.18%. The MobileNetV2 model provided a respectable accuracy of 92.99%, while the VGG16 model had the lowest performance in this comparison. (Note:All the graphs of all codes are present in their respective output terminals.) 

The high accuracies of the top-performing models demonstrate the effectiveness of modern CNN architectures, attention mechanisms, and novel approaches like State-Space Models (Mamba) and KANs for Micro-Doppler signature classification.

Confusion Matrices
Here are the confusion matrices for some of the top-performing models, showing their per-class performance:

ResNet50 with Skip Connections

VGG19 with Attention

Getting Started
To get started with this project, clone the repository and install the necessary dependencies.

Prerequisites
Make sure you have the following libraries installed:

Python 3.x

TensorFlow

PyTorch

NumPy

Matplotlib

Seaborn

scikit-learn

LIME (optional, for explainability)

SHAP (optional, for explainability)

Installation
Clone the repository:

git clone https://github.com/your-username/micro-doppler-classification.git

Install the required packages:

pip install -r requirements.txt

Usage
To train and evaluate the models, you can run the Jupyter notebooks provided in this repository.

Launch Jupyter Notebook:

jupyter notebook

Open one of the model notebooks (e.g., vision-mamba-zsl-cosine-loss.ipynb).

Run the cells in the notebook to train the model and see the evaluation results.

Conclusion
This project successfully demonstrates the application of various deep learning models for Micro-Doppler based target classification. The Vision Mamba with ZSL model stands out with the highest accuracy, but several other models also provide excellent performance. Future work could involve exploring other advanced architectures, experimenting with different data augmentation techniques, or deploying the best-performing model in a real-time application.
