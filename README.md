# Brain Tumor Classification using Deep Learning

## Project Overview

This project implements and compares multiple deep learning architectures for classifying brain tumor MRI images into four categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary Tumor**. The project demonstrates both custom CNN architecture design and transfer learning approaches using state-of-the-art pre-trained models.

## Objective

Develop and evaluate multiple CNN models for accurate brain tumor classification from MRI scans, comparing performance across different architectural approaches to identify optimal solutions for medical image classification.

## Dataset

**Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle

**Dataset Statistics**:

- **Total Images**: ~7,000+ MRI scans
- **Training Set**: ~5,700 images
- **Testing Set**: ~1,300 images
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Format**: JPG/PNG images
- **Characteristics**: Varied image sizes, slight class imbalance

## Models Implemented

### 1. Custom CNN

- **Architecture**: Custom-designed Convolutional Neural Network
- **Input Size**: 128×128 pixels
- **Layers**: Multiple Conv2D, MaxPooling2D, Dense layers with Dropout
- **Approach**: Built from scratch without pre-trained weights
- **Highlights**:
  - Lightweight architecture
  - Faster training and inference
  - Lower computational requirements
  - Class weight balancing for handling imbalance

### 2. VGG16 Transfer Learning

- **Architecture**: VGG16 with custom classification head
- **Input Size**: 224×224 pixels
- **Approach**: Two-phase training
  - Phase 1: Frozen base model (feature extraction)
  - Phase 2: Fine-tuning top layers
- **Highlights**:
  - Deep feature extraction (16 layers)
  - Pre-trained on ImageNet
  - Proven performance on medical imaging tasks

### 3. InceptionV3 Transfer Learning

- **Architecture**: InceptionV3 with custom classification layers
- **Input Size**: 224×224 pixels
- **Approach**: Multi-scale feature extraction with inception modules
- **Highlights**:
  - Efficient multi-scale processing
  - Reduced parameters compared to VGG
  - Advanced architectural design with parallel pathways

### 4. DenseNet121 Transfer Learning

- **Architecture**: DenseNet121 with custom top layers
- **Input Size**: 224×224 pixels
- **Approach**: Dense connections between layers
- **Highlights**:
  - Feature reuse through dense connections
  - Parameter efficient architecture
  - Better gradient flow for deeper networks
  - Excellent for limited medical data

## Model Comparison (`model_comparison.ipynb`)

Comprehensive comparison notebook that:

- Loads all four trained models from Google Drive
- Evaluates on the same test dataset
- Generates performance metrics and visualizations
- Provides detailed per-class analysis
- Creates confusion matrices for all models
- Compares architectural characteristics

**Evaluation Metrics**:

- Test Accuracy
- Test Loss
- Precision, Recall, F1-Score (per class)
- Confusion Matrices
- Parameter Count Comparison

## Technical Stack

- **Framework**: TensorFlow 2.x / Keras
- **Language**: Python 3.12
- **Libraries**:
  - NumPy, Pandas (data manipulation)
  - Matplotlib, Seaborn (visualization)
  - Scikit-learn (metrics)
  - Kaggle API (dataset download)
- **Environment**: Google Colab (GPU-accelerated)

## Getting Started

### Prerequisites

- Google Colab account (recommended) or local Python environment
- Kaggle account and API credentials (`kaggle.json`)
- Google Drive (for model storage)

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <project-name>
   ```

2. **Upload to Google Colab**

   - Upload the notebooks to your Google Colab environment
   - Mount Google Drive for model storage

3. **Setup Kaggle API**

   - Download `kaggle.json` from your Kaggle account settings
   - Upload it to the Colab environment
   - The notebooks will automatically configure Kaggle credentials

4. **Run Notebooks**
   - Execute each model notebook sequentially
   - Models will be saved to Google Drive
   - Run comparison notebook after all models are trained

## Training Process

Each model follows a similar training pipeline:

1. **Data Loading**: Download dataset via Kaggle API
2. **Preprocessing**:
   - Image resizing (128×128 or 224×224)
   - Normalization/preprocessing specific to each architecture
   - Data augmentation (rotation, shifts, flips, zoom)
3. **Class Imbalance Handling**: Computed class weights
4. **Training**:
   - Train/validation split (typically 85/15)
   - Early stopping and learning rate reduction
   - Model checkpointing
5. **Evaluation**:
   - Test set evaluation
   - Confusion matrix generation
   - Per-class metrics analysis
6. **Model Saving**: Save trained models to Google Drive

## Key Findings

### Model Performance Characteristics

**Custom CNN**:

- Fastest training and inference
- Smallest model size
- Lower computational requirements
- May have lower accuracy than transfer learning models

**VGG16**:

- Strong feature extraction
- High accuracy on medical images
- Large number of parameters
- Higher memory requirements

**InceptionV3**:

- Multi-scale feature extraction
- More parameter efficient than VGG
- Good balance of accuracy and efficiency

**DenseNet121**:

- Excellent feature reuse
- Parameter efficient
- Strong performance with limited data
- Good gradient flow

### Use Case Recommendations

| Scenario                  | Recommended Model      | Reason                           |
| ------------------------- | ---------------------- | -------------------------------- |
| **Production Deployment** | VGG16 or DenseNet      | Highest accuracy and reliability |
| **Resource-Constrained**  | Custom CNN             | Lightweight, fast inference      |
| **Mobile/Edge Devices**   | Custom CNN or DenseNet | Smaller size, efficient          |
| **Research/Development**  | All models             | Comparative analysis             |

## Data Augmentation Techniques

Applied augmentation strategies:

- Horizontal flipping
- Rotation (±10-20 degrees)
- Width/Height shifts (5-10%)
- Zoom (5-10%)
- Brightness adjustments (model-specific)

## Handling Class Imbalance

- Computed balanced class weights
- Applied during model training
- Ensures fair representation of minority classes
- Improves model generalization

## Medical Imaging Considerations

- **Preprocessing**: Consistent image normalization
- **Validation**: Stratified splits to maintain class distribution
- **Evaluation**: Focus on recall to minimize false negatives
- **Interpretability**: Confusion matrices for clinical understanding

## Training Configuration

**Common Hyperparameters**:

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Batch Size**: 32
- **Epochs**: 15-20 (with early stopping)
- **Validation Split**: 15-20%

**Transfer Learning Specifics**:

- **Phase 1**: Frozen base, learning rate ~1e-3
- **Phase 2**: Unfrozen top layers, learning rate ~5e-5
- **Fine-tuning**: Last 15-24 layers unfrozen

## Results Visualization

All notebooks include:

- Training/Validation accuracy and loss curves
- Confusion matrices with heatmaps
- Per-class precision, recall, F1-scores
- ROC-AUC scores (where applicable)
- Comparative bar charts

## Contributing

This is an academic project for Deep Learning coursework.

## License

This project is part of academic coursework. Dataset is licensed under CC0-1.0 from Kaggle.

## Author

Deep Learning Assignment - Brain Tumor Classification
SLIIT - SE4050 Deep Learning Module

## Acknowledgments

- Dataset: [Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- TensorFlow/Keras documentation and community
- Pre-trained model weights from ImageNet
- Google Colab for providing free GPU resources

---

**Note**: This project is for educational purposes as part of the SE4050 Deep Learning module at SLIIT. All models and results are based on academic research and should not be used for actual medical diagnosis without proper validation and regulatory approval.
