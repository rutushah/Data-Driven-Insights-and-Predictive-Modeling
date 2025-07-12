# Group Members

* Nischal Joshi
* Rutu Shah
* Murali Krishna
* Santhosh Ramachandran

# Data-Driven-Insights-and-Predictive-Modeling

## Dataset Summary:

The data set consists of medical attributes gathered from patients in order to forecast the presence or absence of heart disease. There are 16 main attributes like age, sex, chest pain type (cp), blood pressure (trestbps), cholesterol(chol), fasting blood sugar (fbs), resting ECG results (restecg), maximal heart rate achieved (thalach), exercise-induced angina (exang), ST depression (old peak),  slope of the peak exercise (slope), major number of blood vessles (ca) and thalassemia (thal),  predicated attribute(num).

**Key Insights from Analysis**

- The majority of patients belong to the age group of between 40 and 60 years.
- Males show a higher frequency of heart disease in this dataset.
- Chest pain type (`cp`), maximum heart rate achieved (`thalach`), and exercise-induced angina (`exang`) show strong correlation with heart disease.
- Cholesterol and resting blood pressure show weak correlation with the target variable.
- Class distribution of the `target` variable is relatively balanced, allowing for straightforward binary classification.

**Rows**: 303

- **Columns**: 16
- **Source**: Kaggle

## Data Cleaning Used:

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
  *
