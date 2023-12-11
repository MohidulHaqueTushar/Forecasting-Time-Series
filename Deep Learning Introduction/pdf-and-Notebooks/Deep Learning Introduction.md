# Deep Learning Introduction

## Take Away Message

Understanding why deep learning is so popular and the different components of deep learning. How a deep learning system learn from data.

- Deep learning compoents
- Representation learning
- Linear layers
- Activation functions
- Gradient descent

Two main reasons why deep learning has gained a lot of ground in the last two decades:

- Computational availability: the introduction of GPUs provided the much-needed boost to the widespread use of deep learning and accelerated the progress in the field. <br>

- Data availability: if we keep increasing the data that we provide to a deep learning model, the model will be able to learn better and better functions. With more data, deep learning starts to outperform traditional machine learning.

![Model Performances vs Data](Performance_vs_data.jpg)


## What is deep learning?

A model is called deep learning when it involves automatic feature learning from raw data. <br>
key points are:
- Assembling parametrized modules : composed of a few submodules with a few parameters assembled into a graph-like structure.
- Optimizing it with gradient-based methods : empirically see that most successful deep learning systems are trained using       gradient-based methods.

## Perceptron : first neural network

The fundamental unit of a human brain is something we call a neuron, shown here: <br>
![neuron](neuron.jpg)

The biological neuron has the following parts:
- Dendrites : are branched extensions of the nerve cell that collect inputs from surrounding cells or other neurons.
- Soma : collects these inputs, joins them, and is passed on.
- Axon hillock : connects the soma to the axon, and it controls the firing of the neuron. If the strength of a signal exceeds a threshold, the axon hillock fires an electrical signal through the axon.
- Axon : is the fiber that connects the soma to the nerve endings. It is the axon’s duty to pass on the electrical signal to the endpoints.
- Synapses : are the end points of the nerve cell and transmit the signal to other nerve cells.
<br>

McCulloch and Pitts (1943) were the first to design a mathematical model for the biological neuron. But the McCulloch-Pitts model had a few limitations:

- It only accepted binary variables.
- It considered all input variables equally important.
- There was only one parameter, a threshold, which was not learnable.

In 1957, Frank Rosenblatt generalized the McCulloch-Pitts model and made it a full model whose parameters could be learned.

Fundamental building block of all neural networks : perceptron
![perceptron](perceptron.PNG)

Perceptron has the following components:
- Inputs : real-valued inputs that are fed to a Perceptron. This is like the dendrites in neurons that collect the input.
- Weighted sum : each input is multiplied by a corresponding weight and summed up. The weights determine the importance of each   input in determining the outcome.
- Non-linearity : the weighted sum goes through a non-linear function. For the original Perceptron, it was a step function with   a threshold activation. The output would be positive or negative based on the weighted sum and the threshold of the unit.       Modern-day Perceptrons and neural networks use different kinds of activation functions.

Perceptron in the mathematical form:
![Perception](perception_as_mathematically.PNG)

The Perceptron output is defined by the weighted sum of inputs, which is passed in through a non-linear function.

**Note:** importance of using linear algebra
   -  Help understand neural networks faster.
   -  Make the whole thing feasible because matrix multiplications are something that our modern-day computers and GPUs are          really good at. Without linear algebra, multiplying these inputs with corresponding weights would require us to loop            through the inputs, and it quickly becomes infeasible.

## Linear algebra intuition

### Vectors and vector spaces
In linear algebra, a **vector** is an entity that has both magnitude and direction. For example,

                                    a = [[1,2]], where a is an array (2 dimensional).
This is an array of numbers. If we plot this point in the two-dimensional **coordinate space**, we get a point. And if we draw a line from the origin to this point, we will get an entity with direction and magnitude. This is a vector. A two-dimensional vector space, informally, is all the possible vectors with two entries.This two-dimensional coordinate space is called a vector space.

Extending it to n-dimensions, an n-dimensional vector space is all the possible vectors with n entries.

**Note:** a vector is a point in the n-dimensional vector space.

### Matrices and transformations
Genarally a matrix is a rectangular arrangement of numbers. Matrix specifies a linear transformation of the vector space it resides in.We multiply a vector with a matrix, we are essentially transforming the vector, and the values and dimensions of the matrix define the kind of transformation that happens. We also apply these transformation matrices to vector spaces to develop intuition on how matrix multiplication can rotate and warp the vector spaces.

### Deep learning system
Deep learning is not just one model, but rather a language to express any model in terms of a few parametrized modules with these specific properties:
- Able to produce an output from a given input through a series of computations.
- If the desired output is given, they should be able to pass on information to its inputs on how to change, to arrive at the  desired output. For instance, if the output is lower than what is desired, the module should be able to tell its inputs to change in some direction so that the output becomes closer to the desired one.

To optimize these kinds of systems, we predominantly use gradient-based optimization methods. Therefore, condensing the two properties into one, we can say that these parameterized modules should be differentiable functions

A deep learning system:
![system](deep_learning_system.PNG)

Deep learning can be thought of as a system that takes in raw input data through a series of linear and non-linear transforms to provide us with an output. It also can adjust its internal parameters to make the output as close as possible to the desired output through learning.

#### Representation learning
Learns the best features by which we can make the problem linearly separable. Linearly separable means when we can separate the different classes with a straight line. 

Transforming non-linearly separable data into linearly separable using a function:
![representation](represemtraion_learning.PNG)

The representation learning block may have multiple linear and non-linear functions stacked on top of each other and the overall function of the block is to learn desired function, which transforms the raw input into good features that make the problem linearly separable. The representation learning block learns the right transformations, which makes the task easier.

A dataset that is not linearly separable, trained a neural network on the problem to classify, and then visualized how the input space was transformed by the model into a linearly separable representation.

Non-linearly separable space:
![non-linear](vector_space.PNG)

Transformed to linearly seperable:
![linear](linierity.PNG)

Inside the representation learning block. We can see,

- A linear transformation
- A non-linear activation function
