### NN relation to logistic regression
- Complexity and increase in representional power...
- 
### What does a single node do

### Single Hidden layer
- square bracket superscript index for layer...
- Means 

## **Overview of Neural Networks**

A **Neural Network** is a machine learning model inspired by the structure and function of the human brain. It consists of layers of interconnected **neurons** that process and learn from data. Neural networks are particularly powerful for tasks such as image recognition, natural language processing, and time-series forecasting.

---

### **1. Definitions and Important Terms**
#### **Artificial Neural Network (ANN)**
An ANN is a computational model composed of multiple layers of neurons that transform input data into meaningful outputs.

#### **Neuron (Node)**
A neuron is a fundamental unit of a neural network that applies a mathematical operation to input data and passes the result to the next layer.

#### **Layers in a Neural Network**
1. **Input Layer** – The first layer that receives raw data.
2. **Hidden Layers** – Intermediate layers where computations take place.
3. **Output Layer** – The final layer that provides the model's prediction.

#### **Weights and Biases**
- **Weights (W):** The strength of the connection between neurons.
- **Bias (b):** A constant value added to adjust the weighted sum.

#### **Activation Functions**
Activation functions introduce non-linearity to the network, allowing it to learn complex patterns. Common activation functions include:
- **Sigmoid**: \( \sigma(x) = \frac{1}{1+e^{-x}} \) (used in probability-based tasks)
- **ReLU (Rectified Linear Unit)**: \( f(x) = \max(0, x) \) (used in deep networks)
- **Tanh**: \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) (used for centered outputs)

#### **Forward Propagation**
This is the process where input data flows through the layers, and predictions are made.

#### **Loss Function**
A function that measures the difference between predicted and actual outputs. Common loss functions:
- **Mean Squared Error (MSE)**: Used in regression.
- **Cross-Entropy Loss**: Used in classification.

#### **Backpropagation**
A learning process where errors from the output layer are propagated backward to update weights using **gradient descent**.

#### **Gradient Descent**
An optimization algorithm that adjusts weights by computing the gradient of the loss function. Variants include:
- **Stochastic Gradient Descent (SGD)**
- **Adam (Adaptive Moment Estimation)**
- **RMSprop**

---

### **2. Key Points in Neural Network Training**
1. **Data Preprocessing**: Normalize or standardize input features.
2. **Weight Initialization**: Proper initialization prevents vanishing/exploding gradients.
3. **Choosing the Right Architecture**: Number of layers and neurons impact performance.
4. **Overfitting and Regularization**: Techniques like **dropout** and **L2 regularization** prevent overfitting.
5. **Hyperparameter Tuning**: Learning rate, batch size, and the number of epochs affect performance.
- Learning rate: Controls how much the model updates weights during training.
- Batch size: Number of samples processed before updating model weights.
- Epochs: One complete pass through the entire training dataset.


- Batch Size:
- Stochastic Gradient Descent
---

### **3. Python Code Example (Using TensorFlow/Keras)**
Here’s a simple example of a neural network using **TensorFlow** to classify the MNIST dataset (handwritten digits).

```python
import tensorflow as tf  # Import TensorFlow library
from tensorflow import keras  # Import Keras from TensorFlow
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Load dataset (MNIST handwritten digits dataset)
mnist = keras.datasets.mnist  
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load training and testing data

# Normalize data (scale pixel values to range [0,1] for better training performance)
x_train, x_test = x_train / 255.0, x_test / 255.0  

# Build the neural network model
model = keras.Sequential([  
    keras.layers.Flatten(input_shape=(28, 28)),  # Input Layer: Flattens 28x28 image to a 1D array of 784 elements
    keras.layers.Dense(128, activation='relu'),  # Hidden Layer: Fully connected with 128 neurons, ReLU activation
    keras.layers.Dropout(0.2),  # Regularization: Drops 20% of neurons randomly to prevent overfitting
    keras.layers.Dense(10, activation='softmax')  # Output Layer: 10 neurons (one for each digit), Softmax for classification
])

# Compile the model (define optimizer, loss function, and evaluation metric)
model.compile(optimizer='adam',  # Adam optimizer for adaptive learning rate
              loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])  # Track accuracy during training

# Train the model using the training dataset
model.fit(x_train, y_train, epochs=10)  # Train for 10 epochs

# Evaluate the model using the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)  # Compute test loss and accuracy
print(f"\nTest Accuracy: {test_acc:.4f}")  # Print test accuracy to 4 decimal places
```
#### **Training Phase Output**
```
Epoch 1/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2941 - accuracy: 0.9157
Epoch 2/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1393 - accuracy: 0.9584
Epoch 3/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1057 - accuracy: 0.9680
Epoch 4/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0869 - accuracy: 0.9734
Epoch 5/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0734 - accuracy: 0.9771
Epoch 6/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0654 - accuracy: 0.9796
Epoch 7/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0571 - accuracy: 0.9820
Epoch 8/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0526 - accuracy: 0.9833
Epoch 9/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0480 - accuracy: 0.9851
Epoch 10/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0432 - accuracy: 0.9866
```

#### **Evaluation Phase Output**
```
313/313 [==============================] - 1s 2ms/step - loss: 0.0742 - accuracy: 0.9772

Test Accuracy: 0.9772
```

### **Explanation of the Output**
- **Training Phase (`model.fit`)**:
  - Each epoch shows the number of training batches processed (`1875/1875`).
  - `loss` decreases over epochs, indicating the model is learning.
  - `accuracy` increases over time, showing improved performance.

- **Evaluation Phase (`model.evaluate`)**:
  - Displays test loss and accuracy after evaluating on the test set.
  - **Test Accuracy ~97.7%** means the model correctly classifies about **97.7% of handwritten digits**.

---

Sequential vs Functiona API


---

### **4. Advanced Topics**
- **Convolutional Neural Networks (CNNs)**: Used for image classification.
- **Recurrent Neural Networks (RNNs)**: Used for sequential data like text and time series.
- **Transformer Models**: Used in NLP tasks like ChatGPT.
- **Autoencoders**: Used for data compression and anomaly detection.
- **Generative Adversarial Networks (GANs)**: Used for generating new data.

Neural networks have revolutionized AI, enabling systems to perform tasks previously thought impossible. 🚀

---



the idea of Training Neural Networks...

The many interconections
let nodes at each layer user relations it learned in adjacent layers


The weights you need to train 
this neuron takes these two imputs, finds sum of these two to get this linear combintations 

If you addd more complexity

Information being past from one layer to the next, its sequential

As long as the number of neurons, is not to great... you can have many layers... you can get effiencent learning and not get overfit

You need a lot of data to be effecitive


Neural Network: Minimizing objective fucnton (squared/loss/binary cross-entropy)
- tune connections
- each node/unit learning features in the proccess
feeds feature


appropriate for segmenting the data


There is a singlle unit node doing a computation

single node is basically taking 4 features to do a weighted sum, adds a bias term



You have to create your own envirment for tensorflow

---

- Dense Layer They are densley to the node oft eh past and single layer

- units= number of nodes in layer
  activation:


Define a Sequential model


---

##  Sequential vs Functional
When designing neural networks in **TensorFlow/Keras**, you can use two primary APIs: **Sequential API** and **Functional API**. Both approaches help in defining and structuring neural networks, but they differ in flexibility and complexity.

---

### **1. Sequential API**
The **Sequential API** is the simplest way to create a neural network in Keras. It allows you to stack layers one after another in a linear fashion. This approach is best for models that have a **single input, single output, and a straightforward layer-by-layer structure**.

#### **Key Characteristics:**
- **Linear Stack**: Each layer connects only to the previous and next layer.
- **Easier to Use**: Ideal for beginners and standard feedforward networks.
- **Limited Flexibility**: Cannot create models with multiple inputs/outputs, shared layers, or non-sequential architectures (like residual connections).

#### **Example of Sequential API:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(28, 28)),  # Input layer (Flatten for images)
    Dense(128, activation='relu'),  # Hidden layer
    Dense(10, activation='softmax') # Output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

#### **When to Use Sequential API?**
- Simple feedforward networks (MLP, CNNs without complex branching).
- Prototyping and quick experiments.
- When layers are strictly stacked one after another.

---

### **2. Functional API**
The **Functional API** allows for more complex architectures, including models with multiple inputs/outputs, shared layers, residual connections, and more.

#### **Key Characteristics:**
- **More Flexible**: Can create non-sequential models like ResNets, Inception, or Siamese networks.
- **Supports Multiple Inputs/Outputs**: Can handle complex tasks that require multiple data sources or outputs.
- **Explicit Graph Representation**: Allows layers to be connected flexibly, making it ideal for advanced architectures.

#### **Example of Functional API:**
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten

# Define the input layer
inputs = Input(shape=(28, 28))
x = Flatten()(inputs)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

#### **When to Use Functional API?**
- When you need multiple inputs/outputs.
- If you want to implement **skip connections (Residual Networks)**.
- When layers need to be shared across different parts of the network.
- For designing **custom architectures** like GANs, Autoencoders, or Attention mechanisms.

---

### **Comparison Table**
| Feature | Sequential API | Functional API |
|---------|---------------|---------------|
| Model Complexity | Simple, linear stack | Complex, non-linear (multi-input/output, skip connections) |
| Flexibility | Limited | High |
| Multiple Inputs/Outputs | ❌ Not supported | ✅ Supported |
| Shared Layers | ❌ No | ✅ Yes |
| Use Case | Simple MLP, CNNs | Advanced architectures (ResNet, Autoencoders, GANs) |
| Code Readability | Easier, concise | Slightly more complex |

---

### **Which One Should You Choose?**
- If you're building a **basic neural network**, go with **Sequential API**.
- If you need **more flexibility** (e.g., multiple inputs/outputs, skip connections, shared layers), go with the **Functional API**.

For deep learning research or complex architectures, **Functional API** is preferred. However, for quick prototyping, **Sequential API** is a great choice.

Would you like a more detailed example of a complex model using the Functional API? 🚀

---



```
Defining the model: A shaollow neural network

Building/compliitn gthe model:
- define objective function and optimizer
define metric to evalvuate train/validation
build the network connections, weight matrices, initializes, etc.

Gradient descent determine if you optimize well, (Its fare more important in deep learning)
Gradient Descent Optimiazers:
- 

Loss
- 'categorical_crossentropy'

```

EarlyStopping 
-Monitor training lass and set imporoivment threshold (min_delta)
Waiting certain number of epochs if no improvements (patience)
Terminate training 
