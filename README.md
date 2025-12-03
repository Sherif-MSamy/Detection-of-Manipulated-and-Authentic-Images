# Detection of Manipulated and Authentic Images

This project introduces a robust neural network for detecting manipulated (fake) and authentic (real) images. Utilizing a custom-built Convolutional Neural Network (CNN) from scratch, the system effectively classifies images without relying on pre-trained models. The methodology encompasses comprehensive data preparation, a meticulously designed CNN architecture with regularization techniques, and rigorous training and evaluation. Achieving a test accuracy of approximately 75.76%, this work demonstrates a foundational approach to image forensics, with detailed insights into model performance through classification reports and confusion matrices.

## Overview
This project focuses on developing a neural network capable of distinguishing between real and digitally manipulated images. The goal is to build a robust model from scratch, without utilizing pre-trained architectures, to classify images as either "authentic" or "manipulated" (fake).

## Dataset
The project utilizes the "[Real and Fake Images Dataset for Image Forensics](https://www.kaggle.com/datasets/shivamardeshna/real-and-fake-images-dataset-for-image-forensics)" from Kaggle. This dataset is specifically curated for image forensics tasks, containing a diverse collection of both real and fake images.

### Dataset Distribution:
The dataset is split into training, validation, and testing sets as follows:

| Set        | Type        | Count   |
|------------|-------------|---------|
| Training   | Fake Images | 20,001  |
| Training   | Real Images | 20,001  |
| **Total Training** |             | **40,002**|
| Validation | Fake Images | 6,161   |
| Validation | Real Images | 6,199   |
| **Total Validation**|             | **12,360**|
| Testing    | Fake Images | 2,623   |
| Testing    | Real Images | 2,604   |
| **Total Testing**  |             | **5,227** |
| **Grand Total**    |             | **57,589**|

## Methodology

### 1. Data Preparation
- Images are resized to a uniform dimension of (256, 256) pixels.
- Pixel values are normalized by rescaling them to the range [0, 1] (dividing by 255).
- Data augmentation techniques are applied to the training set to enhance model generalization and prevent overfitting. These include:
    - Random horizontal flipping.
    - Random rotation with a factor of 0.1.
    - Random zooming with a factor of 0.1.

### 2. Model Architecture
A custom Convolutional Neural Network (CNN) is designed and implemented using TensorFlow/Keras. The architecture consists of several convolutional blocks followed by dense layers:

-   **Input Layer:** Takes images of shape (256, 256, 3).
-   **Data Augmentation Layers:** Apply random transformations to the input images.
-   **Rescaling Layer:** Normalizes pixel values.
-   **Convolutional Blocks (4 blocks):** Each block typically includes:
    -   `Conv2D` layers with ReLU activation for feature extraction.
    -   `BatchNormalization` to stabilize and accelerate training.
    -   `MaxPooling2D` for down-sampling and reducing spatial dimensions.
    -   `Dropout` layers (with rates of 0.2, 0.2, 0.3, and 0.4 respectively) for regularization to prevent overfitting.
-   **Flatten Layer:** Converts the 2D feature maps into a 1D vector.
-   **Dense Layers:**
    -   A dense layer with 512 units and ReLU activation.
    -   `BatchNormalization` and `Dropout` (0.5) are applied.
-   **Output Layer:** A final dense layer with 1 unit and a sigmoid activation function for binary classification (authentic vs. manipulated).

### 3. Training and Evaluation
-   **Loss Function:** Binary Crossentropy, suitable for binary classification tasks.
-   **Optimizer:** Adam optimizer.
-   **Metrics:** Accuracy.
-   **Epochs:** The model is trained for 20 epochs.
-   **Callbacks:**
    -   `EarlyStopping`: Monitors validation loss and stops training if it doesn't improve for 5 consecutive epochs, restoring the best weights.
    -   `ReduceLROnPlateau`: Reduces the learning rate by a factor of 0.2 if the validation loss plateaus for 3 epochs, with a minimum learning rate of 1e-6.
-   **Evaluation:** The model's performance is evaluated on the test set using:
    -   Overall Accuracy.
    -   Classification Report (Precision, Recall, F1-Score for each class).
    -   Confusion Matrix visualization.

## Project Structure
```
.
├── Detection-of-Manipulated-and-Authentic-Images.ipynb  # Main Jupyter Notebook with code
├── Detection-of-Manipulated-and-Authentic-Images.pdf    # PDF version of the notebook
├── Project Description.pdf                                # Project requirements document
└── Dataset/                                             # Contains the image dataset
    ├── test/                                            # Test set images
    │   ├── fake/
    │   └── real/
    ├── train/                                           # Training set images
    │   ├── fake/
    │   └── real/
    └── validation/                                      # Validation set images
        ├── fake/
        └── real/
```

## How to Run the Project
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Detection-of-Manipulated-and-Authentic-Images.git
    cd Detection-of-Manipulated-and-Authentic-Images
    ```
2.  **Ensure you have the dataset:** The project expects the `Dataset/` folder with `train`, `validation`, and `test` subdirectories, each containing `fake` and `real` image folders. If you don't have it, download it from the [Kaggle link](https://www.kaggle.com/datasets/shivamardeshna/real-and-fake-images-dataset-for-image-forensics) and place it in the project root.
3.  **Install dependencies:**
    ```bash
    pip install tensorflow matplotlib seaborn scikit-learn
    ```
4.  **Run the Jupyter Notebook:**
    Open the `Detection-of-Manipulated-and-Authentic-Images.ipynb` notebook in a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, VS Code with Python extension) and execute the cells sequentially.

## Results
The model achieves a test accuracy of approximately 75.76%. Detailed performance metrics, including precision, recall, F1-score, and a confusion matrix, are available within the Jupyter Notebook after training and evaluation.

**Classification Report (Example from Notebook):**
```
              precision    recall  f1-score   support

        real       0.71      0.87      0.78      2623
        fake       0.84      0.64      0.72      2604

    accuracy                           0.76      5227
   macro avg       0.77      0.76      0.75      5227
weighted avg       0.77      0.76      0.75      5227
```

**Confusion Matrix (Example from Notebook):**
A visual representation of the confusion matrix is generated and displayed within the notebook, illustrating the true positives, true negatives, false positives, and false negatives.

## Future Work
- Explore more advanced data augmentation techniques.
- Experiment with different custom CNN architectures to improve performance.
- Investigate the use of explainable AI techniques to understand model predictions.
- Test the model on other publicly available image forensics datasets.
