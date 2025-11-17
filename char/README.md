# English Character Recognizing CNN

In this part of the project, we use the EMNIST dataset to train a CNN that'll recognize letters and numbers. In the next project, we'll make one that recognizes entire words. Then, we'll compare the upsides and downsides of each.

This README contains some notes on the mini-project, it will expand over time if and as needed.

## Architecture

We'll be using a VGG-style CNN. The convolution block

## Dataset

The dataset being used here is EMNIST, which is an extension of the classic MNIST dataset. EMNIST includes handwritten letters as well as digits. There are 26 balanced classes of uppercase and lowercase letters (a total of 52 classes) in addition to the 10 digit classes from MNIST. In total, we have 814,255 training samples and 135,000 test samples, each represented as a 28×28 grayscale image. This data is to be pre-processed and then trained on to train the CNN.
