# Kernel Size in Character Recognition Accuracy
### By: Om Kanabar
--- 

## IV & DV
In this experiment, the independent variable is the size of the convolutional kernel. The experimental group for this experiment consists of a kernel size of 2x2 pixels (px), 4x4 px, and 5x5 px. The control for this experiment is a kernel size of 3x3 px.  The dependent variable for this experiment is the model’s accuracy in recognizing handwritten characters from the EMNIST Dataset.


## Hypothesis
If the kernel size of a convolutional neural network is 4x4 px, then it will achieve the highest accuracy because smaller kernels capture very local features but may miss larger patterns, while larger kernels can blur finer details.

## Methods
Prepare the EMNIST dataset using the training ByClass split by oversampling. For building the convolutional neural network (CNN), add a convolution layer with the desired kernel size, a max-pooling layer, 2 convolution layers with the desired kernel size, and a max-pooling layer. Using the oversampled dataset, convert each image into a 4-dimensional tensor and train the CNN on it. Use the testing ByClass split to evaluate the CNN's accuracy. Compare the data to support or reject the hypothesis. 

## Results
Full results are available in [Results/model_results.csv](https://github.com/om-kanabar/ks-cnn/blob/1dd1d9e30bc8a3ed7a12fcb880710a31331fae1e/Results/model_results.csv)

Kernel Size (px): Accuracy
2x2: 80.413%
3x3 (Control): 81.547%
4x4: 81.506%
5x5 px: 82.010%

## Conclusion
  Artificial Intelligence and neural networks are increasingly drawing attention from students, teachers, law enforcement, companies, the postal service, and consumers alike. This experiment made multiple convolutional neural networks that recognized handwritten characters from the EMNIST dataset. More importantly, this experiment’s results can be applied to the real world for multiple purposes, like business records, handwritten mail addresses, and historical records, among others.
  In this experiment, the independent variable was the size of the convolutional kernel. The experimental group for this experiment consisted of a kernel size of 2x2 px, 4x4 px, and 5x5 px. The control for this experiment was a kernel size of 3x3 px. The dependent variable for this experiment was the model’s accuracy in recognizing handwritten characters from the EMNIST Dataset.
  There are some limitations to this experiment, though. Experimental error could have affected the accuracy and reproducibility of this experiment. One possible experimental error is that, before training a neural network, including CNNs, the computer generates random weights for all of the neurons so it can learn. Slight variances in the start could’ve slightly favored a certain model. 
  Another possible source of error is that the EMNIST dataset is relatively small, as some characters may be harder to distinguish, leading to small errors in classification that are inherent to the dataset. 
  Randomness inside of code may also pose experimental error, which is why this experiment trained 3 separate CNN’s, tested them 3 different times, and then averaged their results for the conclusion to help normalize results per CNN, and ensure that repeated runs of the experiment's code will produce similar results.
  The results of this experiment rejected the hypothesis: If the kernel size of a convolutional neural network is 4x4 px, then it will achieve the highest accuracy because smaller kernels capture very local features but may miss larger patterns, while larger kernels can blur finer details. 
  The best kernel size for handwritten recognition was a kernel size of 5x5, followed by 3x3, 4x4, and 2x2, with a mean accuracy of 82.010%, 81.547%, 81.506%, and 80.413%, respectively. 
  One likely explanation for 5x5 px having the highest accuracy is that even-sized layers don’t have a true center pixel, which can introduce misalignment across convolutional layers. Another likely explanation is that the 5x5 px kernel had the highest accuracy because it balanced local detail and contextual awareness for character recognition. 
