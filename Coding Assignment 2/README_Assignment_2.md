# Advanced Ensemble Learning and Evaluation for Cancer Prediction

This is my solution for Coding Assignment 2. This one focuses more on ensemble
methods, hyperparameter tuning, and evaluating the models properly. I used the
Breast Cancer Wisconsin Diagnostic dataset again, but this time I only trained
on the 5 most important features and tuned each model before comparing them.

## About the Dataset

The dataset comes straight from scikit-learn, so there is no download needed.
It has 569 samples and 30 features, and the target is either malignant (0) or
benign (1). Instead of using all 30 features, I selected the 5 features that
are most correlated with the target and trained the models on those.

## What I Did

I loaded the dataset into a Pandas DataFrame and checked for missing values.
For this check I used a conditional statement so the code prints a clear
message depending on whether anything is missing.

Then I found the top 5 features using correlation, split the data into 80%
training and 20% testing (stratified), and scaled the features with
StandardScaler fitted only on the training data.

For the models, I trained three classifiers and tuned each one with
GridSearchCV using 5-fold cross-validation:

- Decision Tree - the baseline model, tuned on max_depth and min_samples_split
- Gradient Boosting - the boosting ensemble, tuned on n_estimators,
  learning_rate, and max_depth
- SVM with an RBF kernel, tuned on C and gamma

Using GridSearchCV on all of them makes the comparison fair, since each model
gets its best settings.

## Plots

All three plots are also saved as PNG files when the notebook runs:

- A hyperparameter impact plot showing how the Decision Tree's training and
  validation accuracy change as max_depth increases. This shows the point
  where the tree starts to overfit.
- A grouped bar chart comparing Accuracy, F1-Score, and ROC-AUC across the
  three tuned models.
- A confusion matrix heatmap for the Gradient Boosting model, showing the
  false positives and false negatives.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Coding_Assignment_2.ipynb
```

Run the cells in order. The plots show up inline and are also saved to the
folder as PNG files.

## Packages Used

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Results

After tuning, the SVM got the highest accuracy and F1-Score, while Gradient
Boosting got the highest ROC-AUC. Both were clearly better than the baseline
Decision Tree, which shows that tuning and using stronger models actually
helps. The confusion matrix for Gradient Boosting had only a few false
negatives, which is good because missing a malignant case is the most serious
error in this kind of problem.

## Files

- `Coding_Assignment_2.ipynb` - the notebook with all the code
- `README.md` - this file
