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

- **Independent Variable:** Convolutional Kernel Size
  - *Experimental Group:*
    - 2x2 px
    - 4x4 px
    - 5x5 px
  - *Control Group:*
    - 3x3 px
- **Dependent Variable:** Model accuracy on the EMNIST ByClass test split

## Hypothesis

If the kernel size of a convolutional neural network is 4x4 px, then it will achieve the highest accuracy because smaller kernels capture very local features but may miss larger patterns, while larger kernels can blur finer details.

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

3. Install Dependencies

```bash
  pip install -r requirements.txt
```

### Running the Experiment

1. Training the Models

```bash
cd Scripts
python3 train-all.py
```

**Note:** This trains 12 models, 3 for each Kernel Size — this took ~30 hours for me.

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
 
## Results

### Raw Data Table

| Kernel Size (px) | Model Instance | Trial 1 Accuracy (%)\* | Trial 2 Accuracy (%)\* | Trial 3 Accuracy (%)\* |
| :---: | :---: | :---: | :---: | :---: |
| 2x2 | 1 | 80.162 | 80.174 | 80.382 |
| 2x2 | 2 | 80.525 | 80.539 | 80.435 |
| 2x2 | 3 | 80.473 | 80.541 | 80.490 |
| 3x3 | 1 | 81.655 | 81.623 | 81.671 |
| 3x3 | 2 | 81.460 | 81.477 | 81.484 |
| 3x3 | 3 | 81.515 | 81.501 | 81.536 |
| 4x4 | 1 | 81.896 | 81.863 | 81.803 |
| 4x4 | 2 | 80.500 | 80.508 | 80.519 |
| 4x4 | 3 | 82.199 | 82.138 | 82.133 |
| 5x5 | 1 | 81.900 | 81.881 | 81.948 |
| 5x5 | 2 | 81.600 | 81.643 | 81.684 |
| 5x5 | 3 | 82.494 | 82.465 | 82.480 |

*Table 1\. Raw Results*

---

*\*All results have been rounded to three decimal places.*

### Per-Model Mean Accuracy Data Table\* 

| Kernel Size | Model Instance | Per-Model Mean Accuracy (%)† |
| :---: | :---: | :---: |
| 2x2 | 1 | 80.240 |
| 2x2 | 2 | 80.499 |
| 2x2 | 3 | 80.501 |
| 3x3 | 1 | 81.649 |
| 3x3 | 2 | 81.474 |
| 3x3 | 3 | 81.517 |
| 4x4 | 1 | 81.854 |
| 4x4 | 2 | 80.509 |
| 4x4 | 3 | 82.156 |
| 5x5 | 1 | 81.910 |
| 5x5 | 2 | 81.642 |
| 5x5 | 3 | 82.480 |

*Table 2\. Per-Model Averaged Results*

---

*\*Means were calculated with the raw, non-rounded values*  
*†All results have been rounded to three decimal places*

### Mean Accuracy Data Table\*

| Kernel Size | Mean Accuracy (%)† |
| :---: | :---: |
| 2x2 | 80.413 |
| 3x3 | 81.547 |
| 4x4 | 81.506 |
| 5x5 | 82.010 |

*Table 3\. Mean Results*

---

*\*Means were calculated with the raw, non-rounded values*  
*†All results have been rounded to three decimal places*

### Per Model Mean Data Graph

**Fig. 1**  
![Fig. 1](/Assets/image9.png)
*Per-Model Mean Data Graph. Visualization created by the researcher with Tableau Public* 

### Mean Data Graph

**Fig. 2**  
![Fig. 2](/Assets/image12.png)
*Mean Data Graph. Visualization created by the researcher with Tableau Public*

## Conclusion

Artificial Intelligence and neural networks are increasingly drawing attention from students, teachers, law enforcement, companies, the postal service, and consumers alike. This experiment made multiple convolutional neural networks that recognized handwritten characters from the EMNIST dataset. More importantly, this experiment's results can be applied to the real world for multiple purposes, like business records, handwritten mail addresses, and historical records, among others.

In this experiment, the independent variable was the size of the convolutional kernel. The experimental group consisted of kernel sizes of 2x2 px, 4x4 px, and 5x5 px. The control was a kernel size of 3x3 px. The dependent variable was the model's accuracy in recognizing handwritten characters from the EMNIST Dataset.

There are some limitations to this experiment. Experimental error could have affected the accuracy and reproducibility of this experiment. One possible experimental error is that, before training a neural network, the computer generates random weights for all of the neurons so it can learn — slight variances in the start could have slightly favored a certain model.

Another possible source of error is that the EMNIST dataset is relatively small, as some characters may be harder to distinguish, leading to small errors in classification that are inherent to the dataset.

Randomness inside of code may also pose experimental error, which is why this experiment trained 3 separate CNNs, tested them 3 different times, and then averaged their results to help normalize results and ensure that repeated runs of the experiment's code will produce similar results.

The results of this experiment **rejected the hypothesis**: If the kernel size of a convolutional neural network is 4x4 px, then it will achieve the highest accuracy because smaller kernels capture very local features but may miss larger patterns, while larger kernels can blur finer details.

The best kernel size for handwritten recognition was 5x5 px, followed by 3x3, 4x4, and 2x2, with a mean accuracy of 82.010%, 81.547%, 81.506%, and 80.413%, respectively.

One likely explanation for 5x5 px having the highest accuracy is that even-sized kernels don't have a true center pixel, which can introduce misalignment across convolutional layers. Another likely explanation is that the 5x5 px kernel balanced local detail and contextual awareness for character recognition.

## License

Copyright © 2025–2026 Om Kanabar. This project is licensed for educational and non-commercial use only. See [`LICENSE`](./LICENSE) for full terms.