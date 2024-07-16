
#The below libraries are used in the assignment
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


#==================================================================#
# Load the mnist dataset using keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Filter data for digits 1, 2, 3 based on the assignemnt requirement
train_filter = np.where((train_labels == 1) | (
    train_labels == 2) | (train_labels == 3))
test_filter = np.where((test_labels == 1) | (
    test_labels == 2) | (test_labels == 3))
train_images, train_labels = train_images[train_filter], train_labels[train_filter]
test_images, test_labels = test_images[test_filter], test_labels[test_filter]
# Flatten and normalize the images
train_images = train_images.reshape(train_images.shape[0], 28*28) / 255
test_images = test_images.reshape(test_images.shape[0], 28*28) / 255

print(train_labels)
#==================================================================#
# Convert labels to one-hot vectors for example class 1 one hot vector is [1,0,0]
def one_hot_encode(labels):
    # Write your code here
    one_hot = np.zeros((len(labels), 4))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot[:, 1:4]  # Slice from 1 to 3 as those are our classes


train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)
print(train_labels)
#==================================================================#


# Softmax function
def softmax(x):
    #Implement the softmax function based on the above formula
    # return the values
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    softmax_output = exp_x / sum_exp_x
    return softmax_output

def ReLu(x):
    return np.maximum(0, x)

# Feedforward function
def feedforward(tr_i):
    # Implement the feedforward function based on the above formula
    # Apply ReLU activation after the dot product with weights and biases
    return ReLu(np.dot(tr_i, weights) + bias) # this returns a 1x3 array of logits

# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    # Returns the cross entropy average loss
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon) # To avoid log(0), clip values to epsilon and 1-epsilon
    cross_entropy = -np.sum(y_true * np.log(y_pred))
    return cross_entropy
#==================================================================#

def train(epochs, losses, accuracies, learning_rate):
  global weights, bias  # Declare weights and bias as global variables
  # # Training the network
  for epoch in range(epochs):
          # reshape the input image to 784x1
          # train_images[i] = train_images[i].reshape(784, 1)
          # Forward pass
          logits = feedforward(train_images)
          loss = cross_entropy_loss(train_labels, softmax(logits))
          losses.append(loss)
          # print("Epoch:", epoch, "Loss:", loss)

          # Backward pass
          # dL/dZ = softmax(Z) - y_true
          # dL/dW = X^T * dL/dZ dot product of input and dL/dZ
          # dL/db = mean(dL/dZ, axis=0, keepdims=True)


          dL_dZ = softmax(logits) - train_labels
          d_weights = np.dot(train_images.T, dL_dZ)/train_images.shape[0]
          d_bias = np.mean(dL_dZ, axis=0, keepdims=True)

          # Update weights and biases
          weights -= learning_rate * d_weights
          bias -= learning_rate * d_bias

          # Calculate accuracy after each epoch
          y_pred = []
          for i in range(len(test_images)):
            test_predictions = predict(test_images[i])
            y_pred.append(test_predictions)
          accuracy = np.mean(y_pred == test_labels)
          accuracies.append(accuracy)
          # print(f"Epoch: {epoch + 1}, Loss: {loss}, Accuracy: {accuracy}")
  return np.mean(accuracies)


# Initialize weights and bias
weights = np.random.randn(784, 3)
#use numpy to inialilize the weights i.e., [784,3]
bias = np.zeros((1, 3))
#use numpy to initalize biases i.e., [1,3]
#==================================================================#


learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]  # Adjust as needed

# Initialize lists to store accuracies for each learning rate
accuracies_lr = []

losses = []
accuracies = []
epochs = 50

# This code here will train the model for each learning rate and store accuracies
for lr in learning_rates:
    accuracy = train(epochs, losses, accuracies, lr)
    accuracies_lr.append(accuracy)
    print(f"Learning Rate: {lr}, Accuracy: {accuracy}")

# Plot the accuracy as a function of learning rate
plt.plot(learning_rates, accuracies_lr, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy as a Function of Learning Rate')
plt.grid(True)
plt.show()


#==================================================================#
# This will be used to predict the digit class
def predict(z):
    logits = feedforward(z)
    softmax_output = softmax(logits)
    output = np.argmax(softmax_output, axis=1)
    return one_hot_encode(output + 1)


# Test the model on the entire test set
# y_pred = []
# for i in range(len(test_images)):
#     test_predictions = predict(test_images[i])
#     y_pred.append(test_predictions)
# print(y_pred[:10])
# print(len(y_pred))

# print(test_labels[:10])
# print(len(test_labels))
# # Evaluate the model
# #==================================================================#
# accuracy = np.mean(y_pred == test_labels)
# print("Test accuracy:", accuracy)

# Plot the loss
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# Plot the accuracy as a function of # of epochs

# Plot the predictions
# Plot the predictions for first 10 images and last 10 images
plt.figure(figsize=(10, 5))
for i in range(len(test_images[0:10])):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(" Prediction: " + str(np.argmax(y_pred[i]) + 1))
    plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
for i in range(len(test_images[11:20])):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i+11].reshape(28, 28), cmap='gray')
    plt.title(" Prediction: " + str(np.argmax(y_pred[i+11]) + 1))
    plt.axis('off')
plt.show()

# Plot the accuracy as a function of learning rate
# Plot the accuracy as a function of # of epochs
# Plot the loss as a function of # of epochs






# Plot the accuracy as a function of epochs
plt.plot(accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy as a Function of Epochs')
plt.grid(True)
plt.show()

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


#==================================================================#
# Load the mnist dataset using keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Filter data for digits 1, 2, 3 based on the assignment requirement
train_filter = np.where((train_labels == 1) | (
    train_labels == 2) | (train_labels == 3))
test_filter = np.where((test_labels == 1) | (
    test_labels == 2) | (test_labels == 3))
train_images, train_labels = train_images[train_filter], train_labels[train_filter]
test_images, test_labels = test_images[test_filter], test_labels[test_filter]
# Flatten and normalize the images
train_images = train_images.reshape(train_images.shape[0], 28*28) / 255
test_images = test_images.reshape(test_images.shape[0], 28*28) / 255


#==================================================================#
# Convert labels to one-hot vectors for example class 1 one hot vector is [1,0,0]
def one_hot_encode(labels):
    one_hot = np.zeros((len(labels), 4))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot[:, 1:4]  # Slice from 1 to 3 as those are our classes


train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)
print(train_labels)
#==================================================================#

# Initialize weights and bias
weights = np.random.randn(784, 3)
# use numpy to initialize the weights i.e., [784,3]
bias = np.zeros((1, 3))
# use numpy to initialize biases i.e., [1,3]
#==================================================================#


# Softmax function
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    softmax_output = exp_x / sum_exp_x
    return softmax_output


# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -np.sum(y_true * np.log(y_pred))
    return cross_entropy


# ReLU activation function
def ReLu(x):
    return np.maximum(0, x)


# Feedforward function
def feedforward(tr_i):
    return ReLu(np.dot(tr_i, weights) + bias)


# Accuracy calculation
def calculate_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))


# Prediction function
def predict(z):
    logits = feedforward(z)
    softmax_output = softmax(logits)
    return softmax_output


# Training function
def train(epochs, learning_rate):
    global weights, bias
    accuracies = []
    losses = []
    for epoch in range(epochs):
        logits = feedforward(train_images)
        loss = cross_entropy_loss(train_labels, softmax(logits))
        losses.append(loss)

        dL_dZ = softmax(logits) - train_labels
        d_weights = np.dot(train_images.T, dL_dZ) / train_images.shape[0]
        d_bias = np.mean(dL_dZ, axis=0, keepdims=True)

        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias

        # Evaluate accuracy
        y_pred = predict(test_images)
        accuracy = calculate_accuracy(test_labels, y_pred)
        accuracies.append(accuracy)

        print(f"Epoch: {epoch + 1}, Loss: {loss}, Accuracy: {accuracy}")

    return losses, accuracies


learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
epochs = 50

# Initialize lists to store results
all_losses = []
all_accuracies = []

for lr in learning_rates:
    weights = np.random.randn(784, 3)
    bias = np.zeros((1, 3))
    losses, accuracies = train(epochs, lr)
    all_losses.append(losses)
    all_accuracies.append(accuracies)

# Plotting the results
plt.figure(figsize=(10, 5))

# Plot loss for each learning rate
plt.subplot(1, 2, 1)
for i, lr in enumerate(learning_rates):
    plt.plot(all_losses[i], label=f'LR={lr}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss as a Function of Epochs')
plt.legend()

# Plot accuracy for each learning rate
plt.subplot(1, 2, 2)
for i, lr in enumerate(learning_rates):
    plt.plot(all_accuracies[i], label=f'LR={lr}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy as a Function of Epochs')
plt.legend()

plt.tight_layout()
plt.show()
