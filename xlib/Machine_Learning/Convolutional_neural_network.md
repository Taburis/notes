# Convolutional Neural Network (CNN)

## Architecture

A good website for ML: [Dive into deep learning](http://d2l.ai/index.html#)

### Convolutional layer
Refering to this [introduction](https://cs231n.github.io/convolutional-networks/).
Given an input tensor $x$ with the shape $W_0\times H_0\times D_0$, a 2D convolutional layer with a $W\times H\times D$ kernel $K$ has $(W\times H\times D_0+1)\times D$ the independent variables (including biases) based on the sharing parameter strategy. They are $D$ filters and each of them has $D_0$ layers ($L^j_i$, the i-th layer of the j-th filter) with shape $W\times H$. One bias is assigned to each filter. 

Denote the convolution operation between $A$ and $B$ by $A\circ B$:
$$
(A\circ B)(j,k)= \sum_{a,b}A(a,b) B(i-a,j-b).
$$

Then the j-th depth convolution layer output can be expressed as
$$
O^j=b_j+\sum_{i=1}^{D_0}L^j_i\circ x_i,
$$

### Softdrop

The function of this layer is to prevent the over-fitting. It will randomly disable some of the elements from the input, and the probablity for disabling the elements are called drop-rate.

### Max pool layer

The function of the max pool layer is preventing the over-fitting by reducing the outputs from selecting the maximum value among the inputs within a loca region as the outputs. For instance, for a size $W\times H$ max pooling layer, it will slice the input into a many small region with shape $W\times H$, (padding or stride may apply). And only the maximum value of each slices will be the output from that region.

### Transpose convolutional layer
More conceptial detials could be found [here](http://d2l.ai/chapter_computer-vision/transposed-conv.html)


## Loss function and accuracy

### cross-entropy

#### standard
Suppose the prediction output is a tensor $p_{i}$ and the truth is $x_{i}$, then the cross entroy py is defined as

$$
H(x,p) = \frac{1}{N}\sum_{i=1}^N-x_i\log p_i,
$$

where the truth $x_i$ can be either 0 or 1 and $p_i<1$. The summary of the feagures:

* Only the preidiction on the truth bit will affect the loss, no penlty/reward for the prediction on other bits.
* Exponentially sensetive for the predictoin on the truth bit.

#### Binary cross-entropy
Suppose the prediction $p_i$ is a binary variable (either 0 or 1), the cross-entropy could be modified to be

$$
H'(x,p)= \frac{1}{N}\sum_{i=1}^N-x_i\log p_i-(1-x_i)\log (1-p_i),
$$

which makes the wrong prediction takes penalty as well.

%#### categorical cross-entropy