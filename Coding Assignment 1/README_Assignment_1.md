# Comparative Analysis of ML Classifiers for Medical Diagnosis

This is my solution for Coding Assignment 1. The goal was to go through the
full machine learning process on a classification problem, starting from data
preparation and ending with model evaluation. I used the Breast Cancer
Wisconsin Diagnostic dataset and trained three different models to predict
whether a tumor is malignant or benign.

## About the Dataset

I used the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn, so
there is nothing to download. It has 569 samples and 30 features. The target
has two classes, where 0 means malignant and 1 means benign.

## What I Did

First I loaded the dataset and converted it from the raw Bunch object into a
Pandas DataFrame with the correct feature names, and added the target column.
Then I checked for missing values (this dataset is already clean, but the
check is included as required).

After that I used correlation to find the 5 features that are most related to
the target. I split the data into 80% training and 20% testing using a
stratified split so the class balance stays the same in both sets, and scaled
the features with StandardScaler. I fit the scaler only on the training data
so the test data does not leak into training.

Then I trained three models on the same data:

- Logistic Regression (the baseline model)
- Random Forest (an ensemble method)
- Support Vector Machine

## Plots

I made three plots to evaluate the models:

- A bar chart comparing accuracy, precision, and recall for all three models.
- A confusion matrix for the best model, so I can see the false positives and
  false negatives.
- A ROC curve with the AUC score for Logistic Regression.

## Packages Used

```
pandas
numpy
matplotlib
scikit-learn
```

## Results

Logistic Regression and SVM ended up with the same highest accuracy, so both
worked well on this dataset. The confusion matrix and ROC curve also show the
models separate the two classes well. For a medical problem like this, I think
the false negatives matter the most, because saying a malignant tumor is
benign would be the worst kind of mistake.

## Files

- `Coding_Assignment_1.ipynb` - the notebook with all the code
- `README.md` - this file
