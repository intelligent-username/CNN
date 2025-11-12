# CNN

![The Plains of heaven by John Martin, 1853](imgs/cover.jpg)

*Note: This writeup assumes background knowledge about [backpropogation and dense networks](https://github.com/intelligent-username/Backpropagation)

A Convolutional Neural Network (CNN) is a type of deep learning algorithm primarily used for image recognition and processing. It consists of multiple layers that automatically learn hierarchical features from input images through convolutional operations. Unlike **Dense** networks, **Convolutional** networks use local connectivity and weight sharing via convolutional layers, allowing them to detect patterns that are spatially invariant across the image. CNNs are widely used in computer vision tasks such as object detection, classification, and segmentation.

## Motivation

While regular feed-forward Dense networks are good for good for finding connections between unrelated/linearly independent data points, alot of the tasks that we need to do involve patterns that are spatially related and more complex. If we want to classify or segment a video, for example, it won't be enough to look at it's pixels in sequence and figure out what they have in common. CNNs are designed to take advantage of the spatial structure in data by extracting local features through convolutional layers. For example, if you have an image that you know represents some English character, you may take out it's vertical and horizontal lines using a convolutional operation. You may then extract the image's luminence at certain points, and train the model on these 3 features in order to teach it how to classify the characters that you're working with.

When working with unspecialized MLPs (Dense networks), the model has to learn all of these spatial relationships from scratch. Also, it's growth compliexty is $O(n^2)$, which makes it very expensive to construct. In order to make a regular Dense network work for image recognition, you would have to flatten the image into a 1D array, which would destroy all of the spatial relationships between pixels. The model would then need many, many layers and neurons to be able to learn these relationships, which would make it very slow and inefficient.

By building this feature-specific complexity into your model, you prepare it to be specialized for the specific context that it's being used for. Of course, this makes CNNs a lot more complicated to construct, but it's usually worth it. As a result, CNNs tend to not only be more efficient in terms of the number of parameters, but they are also more accurate.

## Terminology

- A **feedforward network** is just a neural network that that feeds the data 'forward', i.e. from input to output, passing through functions smoothly without any loops or cycles.
- A **convolution** the function resulting from the operation $(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau$ that tells us how the shape of one function is modified by another. In the context of CNNs, convolutions are used to extract features from images by applying filters (kernels) to the input pixels, so we're often working with discrete convolutions instead of continuous integrals. More on this later.
- **Kernel**: A small matrix used to apply effects such as blurring, sharpening, edge detection, etc. to images.
- **Feature Map**: The output of a convolutional layer after applying the kernel to the input image.
- **Stride**: The number of pixels by which the kernel moves across the image.

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

### Strides , Padding, and Pooling

### Strides, Padding, and Pooling

In convolutional neural networks, controlling the spatial behavior of the kernel and managing feature map size is crucial. This is done through **strides**, **padding**, and **pooling**.

**Strides** determine how far the kernel moves at each step across the input. A stride of 1 shifts the kernel one pixel at a time, producing a densely sampled output. Larger strides skip pixels, generating smaller feature maps and reducing computation.

**Padding** addresses the problem of shrinking feature maps at the borders. Without padding, the kernel cannot fully cover edge pixels, causing the output to be smaller than the input. **Zero-padding** adds rows and columns of zeros around the input, preserving spatial dimensions and allowing edge information to contribute fully to the output.

**Pooling** is a downsampling operation applied to feature maps to reduce dimensionality while retaining essential information. Common types include:

- **Max pooling**: selects the maximum value within each patch (e.g., (2 \times 2)), emphasizing the strongest activations.
- **Average pooling**: computes the mean within each patch, smoothing the representation.

Pooling increases **translation invariance**: small shifts in features do not strongly affect the output, and reduces computational cost.

Together, strides, padding, and pooling allow CNNs to extract spatially meaningful features efficiently, controlling both resolution and computation across layers.

### Backprop Review

## Architecture

### Convolutional Layers

### Pooling Layers

### Fully Connected Layers

### Feature Extraction

### Classifiers

## Algorithm

---

This project is licensed under the MIT License.
