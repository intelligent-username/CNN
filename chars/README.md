# English Word Recognition

Here, an end-to-end CNN model is built to recognize entire words from images. This is a step up from recognizing individual characters, as it involves understanding the context and structure of words.

This type of CNN trains directly on words, not on letters. If we had made a model that was really good at recognizing letters, we would have to take each picture, split it into a series of letters, recognize each letter individually, and then piece them back together into words. Although in theory this sounds like a simpler approaach than training on massive amounts of whole words, in practice it often leads to errors compounding and worse overall performance.

One issue with the former approach is the segmentation: there are countless different fonts, styles, spacings, and sizes. Another issue is that it's simply slower.

Recent advances, like [DeepSeek-OCR](https://deepseek.ai/blog/deepseek-ocr-context-compression)'s convolutional token aggregation, extend this by bundling words into compact visual representations for multimodal decoding, enabling 200K+ pages/day processing on single GPUs without explicit segmentation.

## Architecture

The architecture follows the dictionary-based (DICT) CNN from [Jaderberg et al.](https://arxiv.org/abs/1406.2227) (2014):

- grayscale images resized to 32×100 pixels feed into four conv layers (64 5×5, 128 5×5, 256 3×3, 512 3×3 filters; stride 1, padded)
- 2×2 max pooling after the first three
- followed by a 4096-unit ReLU fully connected layer
- a final 90k-neuron softmax for direct word classification

This end-to-end design will process whole-word images without character segmentation, using multinomial logistic loss to minimize compounding errors from variable fonts and spacings.

## Dataset

The Dataset being used here is Synthwave90k, which consists of a bunch of synthetically generated images of words in various fonts, colors, backgrounds, and distortions. The dataset is about 12 GB in size. It's composed of folders that each usually contain only 1 image. Because there's so much data, we can give the model tons of examples to learn from, which should help it generalize better to new images.

The data is downloaded from HuggingFace, stored in shards of Arrow files in the data/Synth90k directory. Upon running `import_s9.py`, you should see them start to show up. Note that the download process may take a while.

Each sample has an image and its corresponding text label. The images vary in size, so we'll need to preprocess them (like resizing and normalizing) before feeding them into the CNN for training. When training, PyTorch can only work with PIL/NumPy images, so we'll convert the images from their original format to that during preprocessing. This shouldn't be too hard with the HuggingFace datasets library.
