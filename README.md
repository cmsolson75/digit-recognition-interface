# Digit Recognition Interface

## Overview
This project is designed as an basic educational tutorial to teach students how to deploy a neural network for digit recognition. The tutorial demonstrates the use of a convolutional neural network (CNN) for handwritten digit classification using Python, Tkinter for GUI interaction, and PyTorch for model implementation.

The application allows users to draw digits on a canvas and uses a trained LeNet model to predict the digit.

## Features
- Handwritten digit recognition using a trained LeNet model.
- Interactive GUI for drawing and predicting digits.
- Uses Tkinter for the user interface.
- Model deployment using PyTorch.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install torch torchvision numpy pillow
```
or
```bash
pip install -r requirements.txt
```

## File Structure
```
.
├── digit_recognizer.py  # Main application script with GUI and model integration
├── architecture.py      # Neural network definitions
├── model/               # Directory containing the trained model weights
│   ├── leNet_model.pth  # Pre-trained LeNet model weights
└── README.md            # Documentation
```

## Usage
### Running the Application
To launch the digit recognition GUI, run:

```bash
python digit_recognizer.py
```

### How It Works
1. Draw a digit (0-9) on the provided canvas.
2. Click the `PREDICT` button to classify the digit.
3. The predicted digit will be displayed.
4. Use the `CLEAR` button to reset the canvas.

## Model Details
### LeNet Architecture
The tutorial utilizes a Convolutional Neural Network (CNN) based on LeNet:

- Two convolutional layers followed by ReLU activation and max pooling.
- Fully connected layers with dropout for regularization.
- Outputs a probability distribution over 10 digit classes.

### Preprocessing
The drawn image undergoes preprocessing before classification:
- Converted to grayscale.
- Resized to 28x28 pixels.
- Normalized using mean 0.5 and standard deviation 0.5.


