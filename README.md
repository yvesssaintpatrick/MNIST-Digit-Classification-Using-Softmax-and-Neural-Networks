Overview

The goal of this project is to classify images of handwritten digits (1, 2, and 3) from the MNIST dataset using a simple neural network with ReLU activation and softmax output.
![Unknown-3](https://github.com/user-attachments/assets/9c65bcd1-a4b3-485b-977b-877489b4fefa)

Data Preprocessing

The dataset is loaded using Keras and filtered to include only the digits 1, 2, and 3. The images are then flattened and normalized.

Loading and Filtering Data
The MNIST dataset is loaded using Keras, and the data is filtered to include only the digits 1, 2, and 3.

Flattening and Normalizing Images
The images are reshaped from 28x28 to 784-dimensional vectors and normalized to have values between 0 and 1.

One-Hot Encoding Labels
The labels are converted to one-hot vectors. For example, the label 1 is converted to [1, 0, 0].

Training the Neural Network

The neural network is trained using a feedforward approach with ReLU activation, softmax output, and cross-entropy loss.

Initialization
The weights and biases are initialized using NumPy. Weights are initialized randomly, and biases are initialized to zero.

Softmax Function
The softmax function is used to convert logits to probabilities.

Cross-Entropy Loss
The cross-entropy loss function measures the difference between the predicted probabilities and the true labels.

ReLU Activation Function
ReLU activation is used in the hidden layers to introduce non-linearity.

Feedforward Function
The feedforward function calculates the logits by applying weights, biases, and ReLU activation.

Training Function
The training function updates the weights and biases using gradient descent. The network is trained for a specified number of epochs and learning rate.

Prediction

The prediction function uses the trained network to predict the digit class of new images.

Model Evaluation

The model's performance is evaluated using accuracy, which is calculated as the ratio of correctly predicted labels to the total number of labels.

Plotting Loss and Accuracy
Loss and accuracy are plotted as functions of epochs for different learning rates to visualize the training process.

Visualizing Results

The results of the model's predictions are visualized for a subset of test images.

Plotting Predictions
The predicted digits for the first 10 and last 10 images in the test set are plotted along with the corresponding images.

Conclusion

This project demonstrates the implementation of a neural network to classify digits from the MNIST dataset using softmax and ReLU activation. The model is trained and evaluated using various metrics, and the results are visualized.

![Unknown](https://github.com/user-attachments/assets/218d50d2-57fb-4c81-aee3-572a36f8ce86)![Unknown-2](https://github.com/user-attachments/assets/aa7711e3-e9da-424a-8c9a-3cf4da052cb4)

