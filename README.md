# Introduction
This project of Advanced Data Mining for Data-Driven insights and Predictive Modeling will guide through different key stages of the data mining process as mentioned below in 4 different deliverable parts.

* Deliverable 1: Data Collection, Cleaning, and Exploration
* Deliverable 2: Regression Modeling and Performance Evaluation
* Deliverable 3: Classification, Clustering, and Pattern Mining
* Deliverable 4: Final Insights, Recommendations, and Presentation

# Group Members

* Nischal Joshi
* Rutu Shah
* Murali Krishna
* Santhosh Ramachandran


# Deliverable 1: Data Collection, Cleaning, and Exploration

## Data-Driven-Insights-and-Predictive-Modeling

### Dataset Summary:

The data set consists of medical attributes gathered from patients in order to forecast the severity of heart disease. There are 16 main attributes like age, sex, chest pain type (cp), blood pressure (trestbps), cholesterol(chol), fasting blood sugar (fbs), resting ECG results (restecg), maximal heart rate achieved (thalach), exercise-induced angina (exang), ST depression (old peak),  slope of the peak exercise (slope), major number of blood vessles (ca) and thalassemia (thal),  predicated attribute(num).

**Key Insights from Analysis**

- The majority of patients belong to the age group of between 40 and 60 years.
- Males showed a higher frequency of  heart disease.
- Columns with *few missing values* (e.g., restecg, chol) can be imputed (mean/median/mode).
- Columns with *moderate missingness* (e.g., oldpeak, thalch, exang) can also be imputed if meaningful.
- Columns with *high missingness* (e.g., ca, thal, slope) may need to be dropped or carefully analyzed since much of the data is missing.
  
- **Rows**: 920
- **Columns**: 16
- **Source**: Kaggle

### Data Cleaning Used:

* **For Duplicates**:

  * For data cleaning the data set was diagnosed for missing values using  using `df.isnull().sum()`
  * Following are the column with missing values and their count:
    * trestbps : 59
    * chol: 30
    * fbs: 90
    * restecg: 2
    * thalach: 55
    * exang: 55
    * oldpeak: 62
    * slope: 309
    * ca: 611
    * thal: 486
* **For Duplicates**:

  * Detected using `df.duplicated()`
  * Removed duplicates using `df.drop_duplicates()`
* **For Data Exploration**:

  * Verified and corrected the data where necessary using `df.info()` and `df.astype()`.
* **Outliers and Noise**:

  * Visualized distributions using histograms and boxplots
  * Noted some outliers in cholesterol and max heart rate which were retained for modeling context
* **Challenges Faced:**

  * Selection of  the appropiate dataset from the kaggle.
    * Initial we worked in titanic dataset however it lacked the numerical attributes which is essential for further computation.
    * Later we worked with heart disease data set with a lot of numerical data.
  * Several columns in the dataset (e.g., trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) contained missing values — some with very high missing rates (e.g., ca, thal, slope). This posed a  risk of losing too much data if all rows with missing values were dropped.
  * Outliers and unrealistic values were present in the numeric columns (age, chol) could bias the models and reduce performance.
 
# Deliverable 2: Regression Modeling and Performance Evaluation
 
# Deliverable 3: Classification, Clustering, and Pattern Mining
For this deliverable we have covered following concepts as mentioned below

* Classification Models
* hyperparameter tuning
* Evaluation of Classification model
* Clustering model and its visualization
* Association of Rule Mining Techniques
* Insights on how these patterns can be applied in real world

## Classification Models

### What is Classification?
Classification is a type of data analysis that extracts models describing data classes. These models are used to predict the class labels for new, unseen data points based on their features. In this project, we used a heart disease dataset to determine whether a patient is likely to have heart disease or not (binary classification: 0 = No Disease, 1 = Disease).

Models implemented here are KNN (K Nearest Neighbors) and Decision Tree. 

### k-Nearest Neighbors (k-NN)
* KNN is a distance-based classifier that assigns a data point to the majority class among its k nearest neighbors.
* Operates on the principle that similar data points tend to belong to the same class.
* Tested for multiple k values to find optimal accuracy
* Sample code
  ```python
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

* Best accuracy achieved: ~84% (at k=5 or higher)
* Evaluated using accuracy, confusion matrix, and ROC curve.

### Decision Tree
A Decision Tree is a flowchart-like tree structure where:
* Each internal (non-leaf) node represents a decision based on an attribute.
* Each branch represents the outcome of the test.
* Each leaf node represents a class label (target value).
* The root node is the starting point of the decision process.

* We have used the decision tree classifier from scikit-learn.
  ```python
  from sklearn.tree import DecisionTreeClassifier

  classifier = DecisionTreeClassifier()

* Train the classifier on the training dataset as mentioned below
    ```python
      classifier.fit(X_train, y_train)
    
* Predicted the outcomes for the test dataset.
  ```python
  classifier_y_pred = classifier.predict(X_test)

* Used evaluation metrics such as the confusion matrix and accuracy score.
  ```python
  from sklearn.metrics import confusion_matrix, accuracy_score

  print(confusion_matrix(y_test, classifier_y_pred))
  print('Accuracy:', accuracy_score(y_test, classifier_y_pred))

* Printed the decision tree rules in text form.
  ```python
    from sklearn import tree
    text_representation = tree.export_text(classifier)
    print(text_representation)

* Tree plot visualization
  ```python
  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(50,45))
  _ = tree.plot_tree(
      classifier,
      feature_names=feature_names,
      class_names=target_name,
      filled=True
  )

Summary of decision tree,
* We achieved accuracy of: ~79%
* Confusion matrix, ROC curve, and tree visualization were generated to evaluate performance.

### 📈 Model Performance Comparison

| Model                  | Accuracy |
|-------------------------|----------|
| Decision Tree           | ~79%     |
| k-Nearest Neighbors (k-NN) | ~84%     |

Both models were effective in identifying patients at risk of heart disease,  
with **k-NN showing slightly higher accuracy** in this dataset.

## Hyperparameter Tuning of Classification Models

We performed **hyperparameter tuning** for both **k-Nearest Neighbors (k-NN)** and **Decision Tree** classifiers using `GridSearchCV` to optimize their performance.

### k-Nearest Neighbors (k-NN)

- Parameter grid tested:
  - `n_neighbors`: [3, 5, 7, 9]
  - `weights`: ['uniform', 'distance']
  - `metric`: ['euclidean', 'manhattan']

- Best Parameters:
  - `n_neighbors`: 7
  - `weights`: distance
  - `metric`: manhattan

- Best cross-validated score: **~0.877**

- Final evaluation on test data:
  - **Accuracy:** 0.841
  - **Precision:** 0.882
  - **Recall:** 0.882
  - **F1 Score:** 0.882

### Decision Tree

- Parameter grid tested:
  - `max_depth`: [3, 5, 10, None]
  - `min_samples_split`: [2, 5, 10]
  - `criterion`: ['gini', 'entropy']

- Best Parameters:
  - `max_depth`: 3
  - `min_samples_split`: 2
  - `criterion`: gini

- Best cross-validated score: **~0.831**

- Final evaluation on test data:
  - **Accuracy:** 0.781
  - **Precision:** 0.856
  - **Recall:** 0.814
  - **F1 Score:** 0.834

---

###  Insights for hyperparameter
- Hyperparameter tuning improved the performance of both models.
- k-NN achieved higher test accuracy and F1 score compared to Decision Tree on this dataset.
- ROC curves and confusion matrices were plotted to visualize model performance.




