# Graduate Admission Prediction using Custom ANN

📌 Project Overview

This project implements a custom Artificial Neural Network (ANN) from scratch (using NumPy) to predict the probability of student admissions based on their academic profile (GRE, TOEFL, CGPA, SOP, LOR, Research, etc.).

Unlike typical ML/DL libraries, this ANN is manually coded to demonstrate forward propagation, backpropagation, weight updates, and training visualization.

🚀 Purpose

Learn the math and logic behind ANNs without relying on TensorFlow/PyTorch.

Apply neural networks to a real-world dataset (graduate admission prediction).

Visualize accuracy and loss curves to track training performance.

📊 Dataset

Dataset: Admission_Predict_ver1.1

Features: GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research

Target: Chance of Admit (0–1)

🔑 Features of the ANN

Feedforward Neural Network (1 hidden layer)

Sigmoid activation (hidden layer), Softmax activation (output layer)

Manual forward + backward propagation

Tracks accuracy & loss per epoch

Visualization of training progress

📈 Results

Model learns admission probability trends successfully.

Accuracy and loss curves plotted during training.

Shows effectiveness of a scratch-built ANN on real data.
