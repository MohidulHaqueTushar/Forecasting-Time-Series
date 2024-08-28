# Foundations of Deep Learning
Deep learning involves automatic feature learning from raw data. It seeks to exploit the unknown structure
in the input distribution to discover a good representation of the Data Generating process of the input data.
Building a deep learning model is assembling parameterized into dynamic graphs and optimizing it with
gradient-based methods. The key points are:
  1. Assembling parameterized modules: Deep learning systems composed of a few submodules with a few
parameters assembled into a graph-like structure.
  2. Optimizing it with gradient-based methods: it can be empirically seen that most successful deep
learning systems are trained using gradient-based optimization methods

## Reasons behind the popularity
Deep learning has gained a lot of popularity because of the two main reasons:
  1. Increase in computing availability: <br>
    • Computer hardware showed more improvement <br>
    • The introduction of GPUs
  2. Increase in data availability: <br>
    • Amount of the data increasing because of the cheap cost of data storage <br>
    • Data collection is easier than before <br>
    • With more data, deep learning starts to outperform traditional machine learning <br>

## Perceptron
The fundamental unit of the human brain is called a neuron.<br> 
![neuron](https://github.com/MohidulHaqueTushar/Forecasting-Time-Series/blob/main/Blocks-of-Deep-Learning/Image/neuron.JPG)
<br>Perceptron is the fundamental building block
of all neural networks. Perceptron is designed to mimic a neuron. <br>
![Perceptron](https://github.com/MohidulHaqueTushar/Forecasting-Time-Series/blob/main/Blocks-of-Deep-Learning/Image/perceptron.JPG)
<br>
Perceptron has the following components: <br>
  - Inputs: Real-valued inputs that are fed to a Perceptron. This is like the dendrites in neurons that
collect the input.<br>
  - Weighted Sum: Each input is multiplied by a corresponding weight and summed up. The weights
determine the importance of each input in determining the outcome. It is like soma in a neuron. <br>
  - Non-linearity: The weighted sum goes through a non-linear function. This is like Axon Hillock in a
neuron. <br>
  - Output: It supplies the output as Synapses. <br>

## Deep Learning System Components
Deep learning can be thought of as a system that takes in raw input data through a series of linear and
non-linear transforms to provide us with an output. It also can adjust its internal parameters to make the
output as close as possible to the desired output through learning. Deep learning is a modular system. Deep
learning is not just one model, but rather a language to express any model in terms of a few parametrized
modules with these specific properties: <br>
  1. It should be able to produce an output from a given input through a series of computations. <br>
  2. If the desired output is given, they should be able to pass on information to its inputs on how to
change, to arrive at the desired output. <br>
To optimize these kinds of systems, we predominantly use gradient-based optimization methods. These
parameterized modules should be differentiable functions. The most popular deep learning system paradigm
starts with raw input data. The raw input data goes through blocks of linear and non-linear functions.<br>
![DLSystem](https://github.com/MohidulHaqueTushar/Forecasting-Time-Series/blob/main/Blocks-of-Deep-Learning/Image/deep_learning_system.JPG)

## Representation Learning Block
This is the first block of the deep learning system, consisting of linear transformation and non-linear activation
functions. The overall function of this block is to learn a function, which transforms the raw input into good
features that make non-linear problems linearly separable. Learns the best features by which we can make
the problem linearly separable. The representation learning block may have multiple linear and non-linear
functions stacked on top of each other and the overall function of the block is to learn a function, which
transforms the raw input into good features that make the problem linearly separable. The representation
learning block learns the right transformations, which makes the task easier.We can see there is a linear transformation and a non-linear activation function in a representation learning block. <br>
![RL](https://github.com/MohidulHaqueTushar/Forecasting-Time-Series/blob/main/Blocks-of-Deep-Learning/Image/non-linear_to_linear.JPG)

### Linear Transformation
Linear transformations are simply matrix multiplications that transform the input vector space, the heart
of any neural network or deep learning system today. Linear transformation in a neural network context
means affine transformations. A linear transformation fixes the origin, but an affine transformation moves
the vector space. A linear transformation fixes the origin, but something like a translation, which moves
the vector space, is an affine transformation. Non-linearity becomes essential because when we stack linear
transformations on top of each other, we can see that it all works out to be a single linear transformation.
This kind of defeats the purpose of stacking N layers. Multiple linear transformations work as a single
transformation. Non-linearities are introduced by using a non-linear function, called the activation function. <br>

### Non-Linear Activation Functions
Activation functions are non-linear differentiable functions. It transforms linearly inseparable input vector
space to a linearly separable vector space. It works as an axon hillock in a neural network, and are key to the
neural network’s ability to model non-linear data. It transforms input vector space to a linearly separable
vector space. By non-linear transformations linearly inseparable points become linearly separable. Examples
of popular activation functions. <br>

#### Example : Sigmoid
Characteristics:
  - Most common, and known as the logistic function
  - Continuous function and therefore it is differentiable everywhere
  - The derivative is also computationally simpler to calculate.
  - Squashes the input between 0 and 1 <br>


Main drawback: <br>
  - Saturating of the activation: the gradients tend to zero on the flat portions of the sigmoid <br>

#### Example : Hyperbolic Tangent (tanh)
Characteristics: <br>
- Can express tanh as a function of sigmoid
- Outputs value is in between -1 and 1, and symmetrical around the origin
Main drawback: <br>
- Saturating function <br>

#### Example : ReLU 
Characteristics:<br>
- A linear function, with a kink at zero
- Any value greater than zero is retained as is, but all values below zero are squashed to zero
- This squashing is what gives the non-linearity to the activation function
- Squashes the input between 0 and 1
- Non-saturating function
- Computations of activation function and its gradients are cheap
- Training is faster with this function
- Sparsity in the network: having the activation as zero, a large majority of neurons in the network can
be turned off <br>
Main drawbacks: <br>
- Dead ReLUs: if the input is less than zero, then the gradients become zero, and the unit will not learn
anymore
- Average output of a ReLU unit is positive and when we stack multiple layers, this might lead to a
positive bias in the output.<br>

#### Example : Leaky ReLU
Characteristics: <br>
- No dead ReLUs: makes sure the gradients are not zero when input is less than zero
Main drawback:
- Sparsity is lost: no zero output that turns off a unit <br>

## Linear Classifier
This is the second block of the deep learning system, consisting of linear transformation and output activation
functions.

### Output Activation Functions
Functions that enforce a few desirable properties to the output of the network. This kind of function
has a deeper connection with maximum likelihood estimation (MLE) and the chosen loss function. A
linear activation function is used for regression. Sigmoid and tanh activation functions are used for binary
classification, whereas the softmax activation function is more suitable for multiclass classification. <br>

If we want the neural network to predict a continuous number in the case of regression, we just use a
linear activation function (which is like saying there is no activation function). The raw output from the
network is considered the prediction and fed into the loss function. <br>

For classification, the desired output is a class out of all possible classes. For binary classification, sigmoid
and tanh activation functions are preferred. <br>

For multiclass classification, the softmax activation function is a good fit. <br>
#### Example : Softmax
Characteristics: <br>
- Standard output activation for multiclass classification problems
- Converts the raw output from a network into something that resembles a probability across possible
classes <br>

## Loss Function
The last major component in the deep learning system. The loss function is a way to tell how good the
predictions of the model are. Loss function should be differentiable in the deep learning context. <br>

## Train Deep Learning System
Forward propagation gets the output, and by using the loss function we can measure how far we are from the
desired output. Then backward propagation is used to calculate the gradient concerning all the parameters.
With the gradient of the loss function, gradient descent helps us to optimize the loss function. Negative
gradient is used to minimize a loss function, which will point us in the direction of the steepest descent.
The learning rate multiplied by the gradient defines the step we take in each iteration. After each iteration,
all the parameters in the network are updated to get the optimal loss. There are three popular variants of
gradient descent: <br>
  1. Batch gradient descent
  2. Stochastic gradient descent (SGD)
  3. Mini-batch gradient descent

# Using this System to Forecast Time Series Data

At first we need an idea about:
  - Encoder-decoder paradigm <br>
  ![EnDe](https://github.com/MohidulHaqueTushar/Forecasting-Time-Series/blob/main/Blocks-of-Deep-Learning/Image/encoder_decoder.JPG)
  
The following special structure can be used as an Encoder or a Decoder: <br> 
  - Feed-forward neural networks
  - Recurrent neural networks
  - Long short-term memory (LSTM) networks
  - Gated recurrent unit (GRU)
  - Convolution networks (CNN)

The output result of the project [FFNs for Air Passanger Data](https://github.com/MohidulHaqueTushar/Forecasting-Time-Series/blob/main/Blocks-of-Deep-Learning/FFn_AirPassangerDataset.ipynb) are shown below:

<img src="https://github.com/MohidulHaqueTushar/Forecasting-Time-Series/blob/main/Blocks-of-Deep-Learning/Image/Original%20and%20Predicted%20Trends.png" alt="output" width="500"/>

