# Digit Recognition Model README

### See live website at : https://digit-recognition-benab04.streamlit.app/

## Overview

This repository contains the implementation of a digit recognition model using a neural network with three layers. The model is designed to recognize handwritten digits and can be utilized for various applications such as digit classification.

## Model Architecture

The digit recognition model is built with three layers:

1. **Input Layer:** The input layer represents the flattened pixel values of the input images. Each pixel serves as a feature for the model.

2. **Hidden Layer:** The hidden layer is responsible for learning complex patterns and representations from the input data. It enhances the model's ability to recognize features in the images.

3. **Output Layer:** The output layer produces the final prediction. For digit recognition, this layer typically has 10 neurons corresponding to digits 0 through 9, and the softmax activation function is commonly used to convert raw scores into probability distributions.

## Requirements

- Python 3.11
- Dependencies listed in the `requirements.txt` file.

## Usage

1. **Installation:**
   ```bash
   pip install -r requirements.txt
