# Image-Based Cancer Diagnosis Using a CNN

This is my solution for Coding Assignment 3. For this one I built a
Convolutional Neural Network from scratch using PyTorch to classify breast
ultrasound images as either benign or malignant. Most of the work is in the
image pipeline and training the CNN, and then evaluating it with a few plots.

## About the Dataset

I used the Breast Ultrasound Images (BUSI) dataset from Kaggle. It has three
folders: benign, malignant, and normal. Since this is a binary classification
task, I only used the benign and malignant images. The dataset also has mask
images (files with "_mask" in the name), and I skipped those because they are
segmentation masks, not the actual ultrasound images.

The dataset is downloaded with the Kaggle API, so you need your own Kaggle
credentials to run the download cell. **Do not put your real Kaggle key in the
notebook if you upload it somewhere public.** Instead set it in your own
environment:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

or put your `kaggle.json` file in the `~/.kaggle/` folder. The token line in
the notebook is just a placeholder.

## What I Did

**Data pipeline:** I collected the image paths and labels (0 for benign, 1 for
malignant), then split the data into 70% training, 15% validation, and 15%
test using stratified splitting so the class balance stays the same in each
set. I wrote a custom Dataset class so PyTorch can load the images one by one.

I calculated the mean and standard deviation from the training images only and
used them to normalize all the images. For the training set I added
augmentation (random horizontal and vertical flips and small rotations) to
help with overfitting. The validation and test sets are only resized and
normalized. All images are resized to 128x128.

**The CNN:** I built the network with three convolutional blocks. Each block
has a convolution layer, batch normalization, a ReLU activation, and a
max-pooling layer. After the three blocks I flatten the output and pass it
through a fully connected layer with dropout, and then to a single output node
with a sigmoid activation for binary classification.

The rough structure is:

```
Input (3 x 128 x 128)
Conv(3->32)   -> BatchNorm -> ReLU -> MaxPool
Conv(32->64)  -> BatchNorm -> ReLU -> MaxPool
Conv(64->128) -> BatchNorm -> ReLU -> MaxPool
Flatten -> Linear(128) -> ReLU -> Dropout(0.5) -> Linear(1) -> Sigmoid
```

**Training:** I used Binary Cross-Entropy loss (BCELoss) and the Adam
optimizer with a learning rate of 0.001. I trained the model for 25 epochs and
recorded the training and validation loss and accuracy each epoch so I could
plot the learning curves later.

## Plots

Each plot is also saved as a PNG file:

- Learning curves showing training vs validation loss and accuracy across all
  epochs, to check for overfitting or underfitting.
- A ROC curve on the test set with the AUC score.
- A confusion matrix on the test set showing the correct and incorrect
  predictions for each class.

## How to Run

Install the packages first:

```bash
pip install -r requirements.txt
```

Set your Kaggle credentials (see above), then open the notebook:

```bash
jupyter notebook Coding_Assignment_3.ipynb
```

It uses a GPU automatically if one is available, otherwise it runs on the CPU
which is slower. Training the 25 epochs is the part that takes the most time.

## Packages Used

```
torch
torchvision
numpy
pillow
matplotlib
seaborn
scikit-learn
kaggle
```

## Results

The CNN was able to learn to tell benign and malignant images apart. The
learning curves show how the training and validation performance changed over
the epochs, the ROC curve shows how well it separates the classes with the AUC
score, and the confusion matrix shows the errors on the test set. Like in the
other assignments, the false negatives are the most important errors here,
because predicting a malignant image as benign would be the most dangerous
mistake.

## Files

- `Coding_Assignment_3.ipynb` - the notebook with all the code
- `README.md` - this file
