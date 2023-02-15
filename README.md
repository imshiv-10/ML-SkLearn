## ML-SkLearn

> Machine Learning through SKLearn

![ml-engineering](https://user-images.githubusercontent.com/112423329/219110844-a5b0652a-add0-4276-abf8-dcd1b8097b28.jpg)


Introduction to supervised machine learning, decision trees, and gradient boosting using Python.

# Linear regression

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. The goal is to minimize the difference between the predicted and actual values.


# Logistic regression

Logistic regression, on the other hand, is used for predicting a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables. The logistic function is used to model the probability of the outcome being 1 or 0 given the independent variables. Logistic regression is used in classification problems.

In short, Linear Regression is used for predicting a continuous outcome, while Logistic Regression is used for predicting a binary outcome.

# Decision trees

Decision trees are a type of machine learning algorithm that uses a tree-like model to make predictions based on input data. The tree consists of internal nodes that represent test conditions on the feature, and the leaves represent predictions. The algorithm starts at the root node, performs the test and moves down the tree until it reaches a leaf node, where it outputs the prediction. Decision trees can be used for classification or regression tasks and are widely used due to their interpretability and ability to handle both continuous and categorical data.

## Gini Index

Gini Index is a measure of inequality used in decision tree algorithms to determine the best split at each node of the tree. The Gini Index measures the probability that a randomly selected sample from the data would be misclassified if it were randomly labeled based on the class distribution in the split.

The Gini Index is calculated as the sum of the squared probability of each class, with a value of 0 indicating perfect equality (all samples belong to the same class) and a value of 1 indicating maximum inequality (all samples belong to different classes). The Gini Index is then used to evaluate the quality of a split by calculating the weighted sum of the Gini Index for each split, with the weights being the size of the split relative to the total size of the data.

In decision tree algorithms, the split with the lowest Gini Index is chosen as the best split, as it results in the most homogeneous groups. The Gini Index is a commonly used measure for decision tree algorithms, although other measures such as the information gain or the reduction in variance can also be used.


# Decision Trees vs Random Forests

Decision trees and random forests are two related machine learning algorithms used for both classification and regression tasks.

A decision tree is a tree-like model that makes predictions by starting at the root node, evaluating a test condition on the input features, and moving down the tree based on the result of the test until it reaches a leaf node, which represents a prediction. The tree is constructed by recursively splitting the data into smaller groups based on the feature that results in the best split, as measured by a impurity criterion such as the Gini Index or information gain.

A random forest is an ensemble of decision trees, where multiple decision trees are trained on random subsets of the data and the predictions are combined to produce a final prediction. This can lead to improved accuracy and reduced overfitting compared to a single decision tree, as the randomness in the training data and the combining of predictions helps to reduce the impact of overfitting in any one tree.

In summary, decision trees are a simple and interpretable algorithm, but can easily overfit the data. Random forests are an extension of decision trees that improve accuracy and reduce overfitting by combining the predictions of many decision trees trained on different subsets of the data.

# bootstrap

The bootstrap technique is a key component of the random forest algorithm. In a random forest, each decision tree is trained on a different random subset of the data, which is generated using the bootstrap technique.

The bootstrap technique involves randomly sampling the data with replacement, meaning that some data points may be repeated in the sample while others may not be included at all. This allows each decision tree to be trained on a different subset of the data, capturing different aspects of the relationships between the features and the target.

By training each decision tree on a different subset of the data, the random forest is able to capture a diverse set of relationships between the features and the target, which can improve the overall performance of the algorithm compared to a single decision tree. The final prediction made by the random forest is then a combination of the predictions made by each decision tree, typically using a majority vote in the case of classification or an average in the case of regression.

`Generate the first bootstrap sample
sample_1 = np.random.choice(1000, 1000, replace=True)

Generate the second bootstrap sample
sample_2 = np.random.choice(1000, 1000, replace=True)

Continue this process for the remaining 8 bootstrap samples
...
Use each bootstrap sample to train a separate decision tree in the random forest
tree_1 = DecisionTreeClassifier().fit(X[sample_1], y[sample_1])
tree_2 = DecisionTreeClassifier().fit(X[sample_2], y[sample_2])
...

Use the random forest to make predictions
predictions = (tree_1.predict(X_test) + tree_2.predict(X_test) + ...) / 10

# Dataset with 1000 observations
#  ________
# |_1_|_2_|_3_|_4_|_5_|...|_1000_|

# Bootstrap Sample 1
#  ______
# |_2_|_2_|_4_|_4_|_4_|...|_1_|

# Bootstrap Sample 2
#  ______
# |_3_|_1_|_1_|_3_|_2_|...|_3_|

# Bootstrap Sample 3
#  ______
# |_4_|_4_|_2_|_2_|_1_|...|_1_|

# Bootstrap Sample 4
#  ______
# |_1_|_3_|_3_|_1_|_3_|...|_2_|
`

# ensembling

Suppose we have a dataset with 1000 observations and we want to build a random forest with 4 decision trees. We'll use the bootstrap technique to create 4 different subsets of the data, each with the same size as the original dataset (1000 observations), but with some random sampling with replacement.

`
# Dataset with 1000 observations
#  ________
# |_1_|_2_|_3_|_4_|_5_|...|_1000_|

# Bootstrap Sample 1
#  ______
# |_2_|_2_|_4_|_4_|_4_|...|_1_|

# Bootstrap Sample 2
#  ______
# |_3_|_1_|_1_|_3_|_2_|...|_3_|

# Bootstrap Sample 3
#  ______
# |_4_|_4_|_2_|_2_|_1_|...|_1_|

# Bootstrap Sample 4
#  ______
# |_1_|_3_|_3_|_1_|_3_|...|_2_|

`

Each bootstrap sample has the same size as the original dataset (1000 observations), but with some random sampling with replacement. Each bootstrap sample is used to train a separate decision tree in the random forest.

# Decision Tree 1 (trained on Bootstrap Sample 1)
`#  ________
# |_x_|_x_|_x_|_x_|_x_|...|_x_|

# Decision Tree 2 (trained on Bootstrap Sample 2)
#  ________
# |_x_|_x_|_x_|_x_|_x_|...|_x_|

# Decision Tree 3 (trained on Bootstrap Sample 3)
#  ________
# |_x_|_x_|_x_|_x_|_x_|...|_x_|

# Decision Tree 4 (trained on Bootstrap Sample 4)
#  ________
# |_x_|_x_|_x_|_x_|_x_|...|_x_|`


Each decision tree in the random forest makes a prediction for each observation in the original dataset. The final prediction is made by averaging the predictions made by each decision tree.

`# Final Prediction (Average of Predictions Made by Each Decision Tree)
#  ________
# |_x_|_x_|_x_|_x_|_x_|...|_x_|
`

# Hyper parameter tuning in random forest

`# Random Forest 1
# Hyperparameters:
#  - Number of Trees: 100
#  - Maximum Depth of Trees: 5
#  - Minimum Samples per Leaf: 10
#  - Number of Features to Consider at Each Split: sqrt(total number of features)
#
# Decision Tree 1 (trained on Bootstrap Sample 1)
#  ________
# |_x1_|_x2_|_x3_|_x4_|_x5_|...|_x10_|
#
# Decision Tree 2 (trained on Bootstrap Sample 2)
#  ________
# |_x1_|_x2_|_x3_|_x4_|_x6_|...|_x10_|
#
# Decision Tree 3 (trained on Bootstrap Sample 3)
#  ________
# |_x1_|_x2_|_x5_|_x6_|_x7_|...|_x10_|
#
# Decision Tree 4 (trained on Bootstrap Sample 4)
#  ________
# |_x1_|_x3_|_x4_|_x5_|_x7_|...|_x10_|
#

# Random Forest 2
# Hyperparameters:
#  - Number of Trees: 200
#  - Maximum Depth of Trees: 10
#  - Minimum Samples per Leaf: 5
#  - Number of Features to Consider at Each Split: log2(total number of features)
#
# Decision Tree 1 (trained on Bootstrap Sample 1)
#  ________
# |_x1_|_x2_|_x3_|_x4_|_x5_|...|_x10_|
#
# Decision Tree 2 (trained on Bootstrap Sample 2)
#  ________
# |_x1_|_x2_|_x3_|_x4_|_x6_|...|_x10_|
#
# Decision Tree 3 (trained on Bootstrap Sample 3)
#  ________
# |_x1_|_x2_|_x5_|_x6_|_x7_|...|_x10_|
#
# Decision Tree 4 (trained on Bootstrap Sample 4)
#  ________
# |_x1_|_x3_|_x4_|_x5_|_x7_|...|_x10_|
`
In this example, we see two random forests each with different hyperparameters. The number of trees, maximum depth of each tree, minimum samples per leaf, and number of features to consider at each split are all hyperparameters that can be tuned to optimize the performance of the random forest on a specific dataset. The visual example shows that, while both random forests have the same basic structure, they differ in the specific hyperparameters used.


## Overfitting vs underfitting

Overfitting and underfitting are two common problems in machine learning.

Overfitting occurs when a model fits the training data too well, capturing even the noise and random fluctuations in the data. As a result, the model has high accuracy on the training data but poor accuracy on unseen data, which is known as generalization error.

Underfitting occurs when a model is too simple to capture the underlying patterns in the data. As a result, the model has low accuracy on both the training and unseen data.

To prevent overfitting, techniques such as cross-validation, regularization, early stopping, and ensembling can be used. To prevent underfitting, the model complexity can be increased, more data can be collected, or more features can be engineered.

## Nonlinearity

In machine learning, nonlinearity refers to the presence of nonlinear relationships between input features and target variables. Nonlinear models are used to handle such relationships and improve prediction accuracy. Nonlinear models can capture complex relationships in data and provide more accurate predictions compared to linear models, which assume linear relationships between features and target variables.

Examples of nonlinear models in machine learning include decision trees, random forests, support vector machines, artificial neural networks, k-nearest neighbors, and many others. These models use various techniques, such as nonlinear activation functions, polynomial features, and kernel methods, to handle nonlinear relationships in data.

However, modeling nonlinear relationships can also lead to overfitting and decreased interpretability, as the relationship between input features and target variables can become complex and difficult to understand. Therefore, careful selection of nonlinear models, proper model tuning, and regularization methods are often necessary to achieve good performance and interpretability in nonlinear models.




