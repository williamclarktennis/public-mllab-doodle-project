# Doodle Prediction

A machine learning project that classifies hand-drawn doodles using Convolutional Neural Networks (CNN) and Principal Component Analysis (PCA) with fully connected networks.

**Authors:** Simon Ruhland and William Clark

## Dataset

This project uses the ['Quick, Draw!' dataset](https://quickdraw.withgoogle.com/data), which contains 50 million doodles submitted from around the world across 345 categories.

**Our subset:**
- 30 handpicked categories (e.g., airplane, alarm clock, axe, banana, bicycle, blueberry, elbow)
- Approximately 3GB of data
- ~150,000 grayscale images per category
- Image dimensions: 28×28 pixels

## Approach 1: Convolutional Neural Network

### Architecture

- **Input layer:** Shape (28, 28, 1)
- **Conv2D layer 1:** 16 filters, (3,3) kernel, ReLU activation
- **Conv2D layer 2:** 32 filters, (3,3) kernel, ReLU activation
- **MaxPooling2D layer**
- **Fully connected layer:** 128 nodes, ReLU activation
- **Output layer:** Softmax activation (number of nodes = number of categories)

### Loss Function

Categorical crossentropy:

```
min_f { -Σ_i log(σ(f)_i) }
```

Where σ(f)_i = exp(f_i) / Σ_j exp(f_j) produces a probability distribution.

### Training Results

| Categories | Training Images | Training Time |
|-----------|----------------|---------------|
| 5         | 45,000         | ~minutes      |
| 10        | 90,000         | ~minutes      |
| 20        | 180,000        | ~minutes      |
| 30        | 960,000        | 1.5 hours     |

## Approach 2: PCA + Fully Connected Neural Network

To reduce dimensionality and explore alternative approaches, we applied Principal Component Analysis.

### PCA Configuration

- Training categories: airplane, alarm clock, axe, banana, bicycle
- **200 principal components** selected (out of maximum 784)
- Explains ~90% of variance
- Visualized "eigendoodles" show the principal components as interpretable doodle features

### Neural Network Architecture (PCA-transformed data)

- **Input layer:** 200 nodes (principal components)
- **Hidden layer 1:** 128 nodes, ReLU activation
- **Dropout:** 20% (training only)
- **Hidden layer 2:** 128 nodes, ReLU activation
- **Dropout:** 20% (training only)
- **Output layer:** 5 nodes, Softmax activation

## Project Structure

```
.
├── neural_network.py          # CNN model definition
├── training.py                # Training scripts
├── UI.py                      # User interface for CNN predictions
├── UIpca.py                   # User interface for PCA model predictions
├── Presentation.ipynb         # Project presentation notebook
├── TrainingAndVisualizationsPCA.ipynb  # PCA analysis and training
├── training_histograms/       # Training history visualizations
├── category_names.txt         # List of category names
└── category_filenames.txt     # Dataset file references
```

## Usage

### Training

Run the training script to train models on different numbers of categories:

```bash
python training.py
```

### Interactive Prediction

Launch the UI to draw doodles and get real-time predictions:

```bash
python UI.py        # For CNN model
python UIpca.py     # For PCA model
```

## Results

The CNN achieved strong performance across all category counts, with training time scaling approximately linearly with dataset size. The PCA approach successfully reduced dimensionality while maintaining ~90% variance, demonstrating that lower-dimensional representations can still capture essential doodle features.

## Requirements

- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn (for PCA)
- tkinter (for UI)

## Acknowledgments

Dataset: [Google's Quick, Draw!](https://quickdraw.withgoogle.com/data)
