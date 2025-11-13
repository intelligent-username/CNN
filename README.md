# Convoluational Neural Networks

![The Orrery by Joseph Wright of Arbey, 1766](imgs/cover.jpg)

*Note: This writeup assumes background knowledge about [backpropogation and dense networks](https://github.com/intelligent-username/Backpropagation)

A Convolutional Neural Network (CNN) is a type of deep learning algorithm primarily used for image recognition and processing. It consists of multiple layers that automatically learn hierarchical features from input images through convolutional operations. Unlike **Dense** networks, **Convolutional** networks use local connectivity and weight sharing via convolutional layers, allowing them to detect patterns that are spatially invariant across the image. CNNs are widely used in computer vision tasks such as object detection, classification, and segmentation.

## Motivation

While regular feed-forward Dense networks are good for good for finding connections between unrelated/linearly independent data points, alot of the tasks that we need to do involve patterns that are spatially related and more complex. If we want to classify or segment a video, for example, it won't be enough to look at it's pixels in sequence and figure out what they have in common. CNNs are designed to take advantage of the spatial structure in data by extracting local features through convolutional layers. For example, if you have an image that you know represents some English character, you may take out it's vertical and horizontal lines using a convolutional operation. You may then extract the image's luminence at certain points, and train the model on these 3 features in order to teach it how to classify the characters that you're working with.

When working with unspecialized MLPs (Dense networks), the model has to learn all of these spatial relationships from scratch. Also, it's growth compliexty is $O(n^2)$, which makes it very expensive to construct. In order to make a regular Dense network work for image recognition, you would have to flatten the image into a 1D array, which would destroy all of the spatial relationships between pixels. The model would then need many, many layers and neurons to be able to learn these relationships, which would make it very slow and inefficient.

By building this feature-specific complexity into your model, you prepare it to be specialized for the specific context that it's being used for. Of course, this makes CNNs a lot more complicated to construct, but it's usually worth it. As a result, CNNs tend to not only be more efficient in terms of the number of parameters, but they are also more accurate.

## Terminology

- A **feedforward network** is just a neural network that that feeds the data 'forward', i.e. from input to output, passing through functions smoothly without any loops or cycles.
- A **convolution** the function resulting from the operation $(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau$ that tells us how the shape of one function is modified by another. In the context of CNNs, convolutions are used to extract features from images by applying filters (kernels) to the input pixels, so we're often working with discrete convolutions instead of continuous integrals. More on this later.

## Math

In addition to the math used in Dense neural networks, CNNs use convolution operations. Understanding them will clarify how CNNs work and can be adjusted.

### Convolutions

The convolution is the backbone of all of the math that we'll be doing in this topic. In essence, a convolution measures how much one function (or pattern) overlaps with another as it slides across it. Mathematically, the **continuous convolution** is defined as

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau), g(t - \tau), d\tau
$$

With this integrael, we find how much the two functions that we have overlap with some given shift, $t$. We integrate with respect to $\tau$ in order to apply this to the entire domain.

This may be a bit abstract, so let's break it down. Imagine $f$ is a signal that represents some data, and $g$ is a filter that we're using to detect certain features in that data. As we slide $g$ across $f$, we multiply the two functions pointwise and integrate the result, which will tell us how much $g$ matches $f$ at each position $t$.

Below is a simple example with

$$
f =
\begin{cases}
    1 & 0 \leq x \leq 1
    \\
    0 & \text{otherwise}
\end{cases}
$$

(in green)

and

$$
g =
\begin{cases}
    1 & 0 \leq x \leq 1
    \\
    0 & \text{otherwise}
\end{cases}
$$

(in purple)

![Example #1 Illustrated](imgs/f1.png).

When the two align, their product is large, and the integral accumulates that alignment.

Put together, this is what the graph of $h(x) = f(x) * g(x)$ (red) looks like:

![Example #1 Overlap](imgs/f2.png)

Let's take a more complicated example to visualize this better.

For example, imagine if $f(x) = e^{-x^2}$ (in green) and $g(x) = e^{-(x-2)^2}$ (in purple). So, $g$ is the same as $f$ shifted 2 units to the right.

![Example #2 Illustrated](imgs/f3.png)

Their overlap, $h(x) = f(x) * g(x)$ (in red), peaks when $g$ is centered over $f$ at $x=2$:

![Example #2 Overlap](imgs/f4.png)

Now, the point of these convolutions, in this context, isn't necessairly to apply them continuously to some existing function and see the result. Instead, we apply the transformation to a series of discret edata points (like pixels) to see how well certain features (like edges) align with the data. For example, if we have the feature of a vertical edge, we can convolve that with an image to see where vertical edges occur.

In **discrete** form, we can write the convolution as:

$$
(f * g)(i, j) = \sum_m \sum_n f[m, n], g[i - m, j - n]
$$

#### Kernels

When applying these convolutions to an image, $f$ would be the matrix representing the pixels and $g$ (sometimes denoted $k$) is the **kernel** or **filter** that we're applying to it.

The kernel slides across the image, applying it's operation to every single pixel. When we multiply and sum overlapping pixels, we find how much of that part of the image matches the feature we need. This is because we're taking a dot product, and, geometrically, the dot product represents how much two vectors align with each other.

In mathematical form, we can write this as:

$$
y_{i,j} = \sum_{m,n} x_{i+m,, j+n} \cdot k_{m,n}
$$

Upon doing this, we have whatever local patterns that the kernel is designed to detect highlighted in the output. This may be the edges, corners, or something else.

#### Feature Maps

The feature map is the outputted image after applying the kernel to the original image. It simply stores the feature that we need, and is later used by the neural network for processing.

For example, this is the resultant feature map after we apply a Laplacian edge-detection kernel to an image:

![Laplacian Edge Detection Example](imgs/laplace.png)

### Size, Strides, Padding, and Pooling

Just like other machine learning tasks, CNNs have hyperparameters that we can tune to improve performance and efficiency. The hyperparameters that are unique to CNNs are **size**, **strides**, **padding**, and **pooling**.

The **size** of the kernel are its dimensions. For example, a $3 \times 3$ kernel has a height and width of 3 pixels each. If we have a $1 \times 1$ kernel, then we're simply multiplying each pixel by a weight, which is equivalent to a Dense layer. Larger kernels can capture more complex features, but they also increase computational cost and may lead to overfitting. They may also give rise to overfitting.

**Strides** determine how far the kernel moves at each step across the input. For example, if the stride is 1, we itereate through every pixel, whereas if the stride is 2, we skip every other pixel, and so forth. Increasing the stride will reduce the dimensions of the feature map, which can help reduce computational cost. Of course, it comes at the cost of detail.

Whenever our kernel size is greater than one, we are bound to go 'outside' of the input's dimensions whenever we are at or near the edges. This may result in literal program crashes if we're not careful or strange, undefined behaviour if we choose to, say, wrap around the edges when we reach them. One possible solution is to add **padding**. Padding pixels are simply just a border around the original image that have some sort of pre-defined behaviour. **Zero-padding** is when we add rows and columns of zeros around the input. Without padding, we risk having the feature map be smaller than the input, which means that we have lost information.

If we happen to have too much data, or data that is redundant, we use pooling as a sort of 'compression'. **Pooling** is when we look at some window of data in an image and extract the parts of it that we need, discarding the rest. For example, when looking at a pixel, we may only care about the value of the $3 \times 3$ grid surrounding it, and so we may take the average of those 9 pixels and write them all as a single one. Pooling is similar to changing the stride in that it helps us get rid of unnecessary data, but in this case it is an intentional and specific transformation of the inputs.
Note that pooling is not part of the convolution operation, rather it's a step applied before/after it to reduce or standardize the size of the data.

### Backprop & Review

So now, we have all of these different operations: A network composed of layers, which are composed of neurons, which in turn have their own connections, activations, weights and convolutions, as well as some hyperparameters that we can tune. Once we find the optimal combination of these things, we can use backpropogation to find the actual weights that the model will use.

Of course, use [gradient descent](https://www.github.com/intelligent-username/Gradient-Descent) to minimize the loss function, add whatever [regularization](https://www.github.com/intelligent-username/Regularization) we may need, etc. etc. Let's quickly review how it works in order to see what's going on behind the scenes.

First, there is the forward pass. At this stage, we take our current weights and biases, and use them to calculate the output of the network. This involves applying the convolutional operations, activations, and any other transformations that are part of the network architecture.

Then, we use that to find the loss. This is just to get an idea of hwo good our model is so far.

Now, importantly, we find the partial derivatives of the loss function with respect to the weights, and we use them to construct the gradient. This is where the *backpropogation* itself actually comes in. The technique here is to use the chain rule "inside out", wherein we start from the output layer and work our way back to the input layer, calculating the gradients at each step. It's not inherently faster than calculating the gradient "outside in", but, since we can cache the intermediate results, we can save some time since neurons in earlier layers share the same downstream paths.
Finally, we use the gradient to update the weights using whatever optimization algorithm that we're using.

### Finally

Let's take these ideas and see how we can use a convolution to transform an image.

Take the following $630 \times 630$px image:

<img src="imgs/tree.png" alt="Original Image" width="300px">

Let's apply a grayscale filter and then trace it's vertical lines using the following kernel:

$$
% Matrix
\begin{bmatrix}
-1, 0, 1 \\
-1, 0, 1 \\
-1, 0, 1 \\
\end{bmatrix}
$$

We also have padding of 2 pixels, kernel size of 5 by 5, and a stride of 1.

The processing goes as follows:

![Processing](imgs/convoluting.gif)

<!-- Original image source: openclipart -->
<!-- https://openclipart.org/detail/310230/rudolph-christmas-tree-construction-paper -->

Now, this processed image would be ready to passed into the rest of the network for processing. (Also, ignore the low-quality gif and the compression).

## Architecture

![Summarative Diagram](imgs/summary.png)

Now that we understand all this math, let's quickly go over how a CNN is actually implemented in code. The key is to think of a CNN as a feedforward network that has some number of initial layers responsible for feature extraction, followed by some number of Dense layers for learning/regression and a final layer for [classification](https://www.github.com/intelligent-username/Classification).

We simply implement this in code and we're done!

---

This project is licensed under the MIT License.
