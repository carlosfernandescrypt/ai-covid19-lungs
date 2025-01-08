# Neural Network for COVID-19 Detection

This project implements a convolutional neural network (CNN) to detect COVID-19 from chest X-ray images. The model is trained using Keras and TensorFlow, and it classifies images into three categories: COVID-19, non-COVID-19, and normal.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/carlosfernandescrypt/ai-covid19-lungs.git
    cd ai-covid19-lungs
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

The dataset used for training and testing the model consists of chest X-ray images categorized into three classes:
- COVID-19
- Non-COVID-19
- Normal

The dataset should be organized into the following directory structure:
```
dataset/
├── treino/
│   ├── COVID/
│   ├── NON-COVID/
│   └── NORMAL/
└── teste/
    ├── COVID/
    ├── NON-COVID/
    └── NORMAL/
```

## Model Architecture

The model is a convolutional neural network (CNN) with the following layers:
- Convolutional layers with ReLU activation and batch normalization
- Max pooling layers
- Fully connected (dense) layers with dropout
- Output layer with softmax activation

## Training

To train the model, just run the notebook 

The training process includes:
- Data augmentation using `ImageDataGenerator`
- Early stopping and learning rate reduction on plateau
- Model checkpointing to save the best model

## Evaluation

The model is evaluated using:
- Confusion matrix
- ROC curve and AUC
- Accuracy, recall, and specificity metrics

## Results

The results of the model training and evaluation are saved in the `results` directory, including:
- Training logs
- Model weights
- Plots of accuracy, loss, recall, and specificity over epochs
- Confusion matrices and ROC curves

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

