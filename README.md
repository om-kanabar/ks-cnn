# The Effect of Convolutional Kernel Size on Handwritten Character Recognition Accuracy Using the EMNIST Dataset
##### By: Om Kanabar
---
This project was created for Chicago Public Schools STEM Fair 2025-2026.

---
## Table of Contents
- [Overview](#overview)
- [Hypothesis](#hypothesis)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Experiment](#running-the-experiment)
- [Project Structure](#project-structure)
- [Methods](#methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)


## Overview

This experiment tests which convolution kernel size produces the highest accuracy when identifying different characters from the [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

- **Independant Variable:** Convolutional Kernel Size
  - *Experimental Group:*
    - 2x2 px
    - 4x4 px
    - 5x5 px
  - *Control Group:*
    - 3x3 px
- **Dependant Variable:** Model accuracy on the EMNIST ByClass test split

## Hypothesis

If the kernel size of a convolutional neural network is 4x4 px,then it will achieve the highest accuracy because smaller kernels capture very local features but may miss larger patterns, while larger kernels can blur finer details.

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- 4.5 GB of free disk space
- 8 GB of RAM *(16 GB recommended)*

### Installation

1. Clone the repository

```bash
git clone https://github.com/om-kanabar/ks-cnn.git
cd ks-cnn
```

2. **Create a virtual environment** *(recommended)*

- MacOS/Linux

  ```bash
    python -m venv venv
    source venv/bin/activate
  ```

- Windows
  ```bash
    python -m venv venv
    venv\Scripts\activate
  ```

3. Install Dependecies

```bash
  pip install -r requirements.txt
```

### Running the Experiment

1. Training the Models

```bash
cd Scripts
python3 train-all.py
```

**Note:** This trains 12 models, 3 for each Kernel Size this took ~30 hours for me.

2. Testing the Models

```bash
python3 testmodels.py
```

## Project Structure
 
```
/ks-cnn
├── Assets
│   └── NeuralNetworkVisualization.png
├── Data
│   └── emnist-byclass-mapping.txt
├── Learning
│   ├── pandas_numpy_practice.py
│   ├── plot_sample.py
│   ├── tiny_data.csv
│   └── visualoptimize.py
├── Miscellaneous
│   └── uncited-sources.txt
├── Models
├── Results
│   └── model_results.csv
├── Scripts
│   ├── convertfiles.py
│   ├── databalance.py
│   ├── datacheck.py
│   ├── datacount.py
│   ├── testmodels.py
│   ├── train-all.py
│   ├── train-base.py
│   └── train-model.py
├── LICENSE
├── README.md
├── project_structure.txt
└── requirements.txt
```

## Methods

1. **Dataset Preparation**: Use TFDS to import the EMNIST ByClass split and oversample each character class (digits: 20,000–25,000; majuscules: 15,000–20,000; minuscules: 18,000–22,000 samples).
2. **CNN Construction**: Normalize pixel values to [0, 1] and reshape images into 4D tensors. Build a CNN with 3 convolution layers (target kernel size) and 2 max-pooling layers, a 128-neuron dense layer, and a 62-neuron softmax output layer. Use ReLU for all other layers. Compile with Adam optimizer and sparse categorical cross-entropy loss.
3. **Training**: Split the dataset (test size: 0.115) and augment training data with Gaussian noise (factor: 0.05). Train for up to 20 epochs (batch size: 64) with early stopping monitoring val_loss. Repeat 3 times per kernel size (12 models total).
4. **Testing**: Evaluate each model on the EMNIST ByClass test split with Gaussian noise applied to simulate real-world conditions. Calculate accuracy as the percentage of correctly classified images.
5. **Analysis**: Compare mean accuracy across all kernel sizes to support or reject the hypothesis.
 
