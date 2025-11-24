# English Word Recognition

Here, an end-to-end CNN model is built to recognize entire words from images. This is a step up from recognizing individual characters, as it involves understanding the context and structure of words.

This type of CNN trains directly on words, not on letters. If we had made a model that was really good at recognizing letters, we would have to take each picture, split it into a series of letters, recognize each letter individually, and then piece them back together into words. Although in theory this sounds like a simpler approaach than training on massive amounts of whole words, in practice it often leads to errors compounding and worse overall performance.

One issue with the former approach is the segmentation: there are countless different fonts, styles, spacings, and sizes. Another issue is that it's simply slower.

Recent advances, like [DeepSeek-OCR](https://deepseek.ai/blog/deepseek-ocr-context-compression)'s convolutional token aggregation, extend this by bundling words into compact visual representations for multimodal decoding, enabling 200K+ pages/day processing on single GPUs without explicit segmentation.

## Dataset

The Dataset being used here is Synthwave90k, from the [Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition]((https://arxiv.org/abs/1406.2227)) paper by Jaderberg et al. (2014).
It consists of a bunch of synthetically generated images of words in various fonts, colors, backgrounds, and distortions. For future implementations, one can take inspiration from this paper to, for example, create a more efficient version of the same dataset, make the data more complex, or even create the same kind of dataset for a different language.

In total, we have a vocabulary of about 90,000. The dataset is about 12 GB in size. Because there's so much data, we can give the model tons of examples to learn from, which should help it generalize better to new words.

The data is downloaded from HuggingFace, stored in shards of Arrow files in the data/Synth90k directory. Upon running `import_s9.py`, you should see them start to show up. Note that the download process may take a while.

Each sample has an image and its corresponding text label. The images vary in size, so we'll need to preprocess them (like resizing and normalizing) before feeding them into the CNN for training. When training, PyTorch can only work with PIL/NumPy images, so we'll convert the images from their original format to that during preprocessing. This shouldn't be too hard with the HuggingFace datasets library.

## Architecture

The CNN architecture itself is the focus of *this* project, and we follow the next paper by Jaderberg et al, [Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition
](https://arxiv.org/abs/1507.05717) (2014). It three introduces different architectures: DICT, CHAR, and NGRAM. DICT is the main focus, and so it'll be the one that I'm implementing here. The DICT architecutre consists of:

- 4 Convolutional Layers
  - Conv1: 64 filters, 5×5, stride 1, pad; ReLU → LRN → MaxPool(2×2)
  - Conv2: 128 filters, 5×5; ReLU → LRN → MaxPool(2×2)
  - Conv3: 256 filters, 3×3; ReLU
  - Conv4: 512 filters, 3×3; ReLU → MaxPool(2×2)
- 2 Fully Connected Layers, both with ReLU activations and Dropout
- Softmax for Classifcation

Notice that this is already a *massive* neural network to train. On most laptops, it'll take hundreds of hours. You may even want to rent a cloud GPU to speed this up.
