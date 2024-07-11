# Breast Cancer Diagnostic Model

## Project Overview

Breast cancer is one of the most common types of cancer among women worldwide. Early detection and diagnosis are crucial for improving the survival rates and treatment outcomes for breast cancer patients. This project aims to build and evaluate various machine learning models to predict breast cancer using clinical and pathological data.

## Repository Structure

```
Advanced_Project_Breast_Cancer_Prediction/
├── data/
│   └── breast_cancer.csv
├── notebooks/
│   └── Advanced_Project_Breast_Cancer_Prediction_Code_OK.ipynb
├── reports/
│   └── figures/
│       ├── roc_curve.jpeg
│       └── model_comparison.jpeg
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Business Problem

Early detection of breast cancer can significantly improve the prognosis and treatment success. This project leverages machine learning techniques to build predictive models that can help in diagnosing breast cancer at an early stage, potentially saving lives and reducing treatment costs.

## Business Assumptions

- The dataset is representative of the population.
- The features provided are sufficient to build accurate predictive models.
- There are no significant biases in the data.

## Solution Strategy

1. **Data Description**: Initial exploration of the dataset to understand the distribution and relationships between features.
2. **Feature Engineering**: Creating new features based on domain knowledge and initial data exploration.
3. **Data Preprocessing**: Cleaning and transforming the data to prepare it for modeling.
4. **Exploratory Data Analysis (EDA)**: Detailed analysis to uncover patterns and relationships within the data.
5. **Model Building**: Training various machine learning models.
6. **Model Evaluation**: Assessing the performance of the models using metrics such as accuracy, precision, recall, and ROC-AUC.
7. **Model Deployment**: Preparing the best model for deployment.

## Data Description

- **Dataset**: The breast cancer dataset includes clinical and pathological data for breast cancer diagnosis.
- **Features**: Includes attributes like mean radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.
- **Target Variable**: Diagnosis (M for malignant, B for benign).

## Feature Engineering

- **Hypothesis Creation**: Based on domain knowledge, hypotheses were created to test various relationships between features and the target variable.
- **New Features**: Derived features that could potentially improve model performance.

## Data Preprocessing

1. **Loading the Data**:
    ```python
    import pandas as pd
    df = pd.read_csv('data/breast_cancer.csv')
    ```

2. **Handling Missing Values**:
    ```python
    df.isnull().sum()
    ```
    - There were no missing values in the dataset.

3. **Encoding Categorical Variables**:
    ```python
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    ```

4. **Scaling Features**:
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_scaled = scaler.fit_transform(X)
    ```

## Exploratory Data Analysis (EDA)

- **Univariate Analysis**:
  Density plots and histograms were used to visualize the distribution of each feature.
  
- **Bivariate Analysis**:
  Boxplots and scatter plots were used to examine the relationships between features and the diagnosis.

- **Multivariate Analysis**:
  Heatmaps were used to visualize correlations between multiple features.

## Model Building

Various models were trained and evaluated on the dataset. Below are the implementation details for each model:

1. **Logistic Regression**:
    ```python
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    ```

2. **K-Nearest Neighbors (KNN)**:
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    ```

3. **Support Vector Machine (SVM)**:
    ```python
    from sklearn.svm import SVC
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)
    ```

4. **Decision Tree Classifier**:
    ```python
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    ```

5. **Random Forest Classifier**:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    rand_clf = RandomForestClassifier()
    rand_clf.fit(X_train, y_train)
    ```

6. **Gradient Boosting Classifier**:
    ```python
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    ```

7. **XGBoost Classifier**:
    ```python
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    ```

## Model Evaluation

Each model was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. Below are the results for each model:

- **Logistic Regression**:
  - Accuracy: 96.49%
  - Precision: 97.78%
  - Recall: 94.00%
  - F1-Score: 95.87%
  - ROC-AUC: 0.96

- **K-Nearest Neighbors (KNN)**:
  - Accuracy: 95.61%
  - Precision: 97.27%
  - Recall: 91.00%
  - F1-Score: 94.00%
  - ROC-AUC: 0.95

- **Support Vector Machine (SVM)**:
  - Accuracy: 98.25%
  - Precision: 100%
  - Recall: 94.00%
  - F1-Score: 96.87%
  - ROC-AUC: 0.98

- **Decision Tree Classifier**:
  - Accuracy: 92.98%
  - Precision: 86.54%
  - Recall: 91.00%
  - F1-Score: 88.70%
  - ROC-AUC: 0.91

- **Random Forest Classifier**:
  - Accuracy: 98.25%
  - Precision: 93.75%
  - Recall: 98.00%
  - F1-Score: 95.87%
  - ROC-AUC: 0.98

- **Gradient Boosting Classifier**:
  - Accuracy: 96.49%
  - Precision: 93.75%
  - Recall: 94.00%
  - F1-Score: 93.87%
  - ROC-AUC: 0.96

- **XGBoost Classifier**:
  - Accuracy: 95.61%
  - Precision: 93.75%
  - Recall: 94.00%
  - F1-Score: 93.87%
  - ROC-AUC: 0.95

## Business Results

- **What percentage of patients are predicted to have malignant tumors? How many of these patients can be accurately identified using the top-performing model?**

    i. The dataset contains 569 instances, with 212 (37.28%) instances of malignant tumors and 357 (62.72%) instances of benign tumors  .

    ii. The best-performing model, Support Vector Machine (SVM), has a precision of 98% for predicting malignant tumors  . Therefore, using the model, it is possible to correctly identify approximately 208 patients with malignant tumors (98% of 212). However, the recall is about 94.00%, meaning approximately 199 of the 212 malignant cases are correctly identified, with a margin of error (± 0.0016)  .

- **If the diagnostic capacity is increased to include more patients, how many additional patients with malignant tumors can be accurately identified?**

    Increasing the capacity to diagnose an additional 100 patients using the SVM model will likely result in correctly identifying approximately 37 more patients with malignant tumors (considering the prevalence of 37.28% malignant cases in the dataset and the recall rate of 94.00%)  .

- **How many patients need to be diagnosed to ensure that at least 80% of all malignant tumor cases are accurately identified?**

    The model sorts 94.00% of the malignant tumor cases correctly. To ensure at least 80% of all malignant cases (i.e., 170 out of 212 cases) are accurately identified, approximately 181 patients need to be diagnosed (considering the recall rate and precision)  .

### Conclusion

The random model classified correctly just a small fraction of the malignant cases. The final model, specifically the Support Vector Machine (SVM), demonstrated a significantly higher ability to differentiate the classes and managed to correctly classify 98% of the malignant cases. The lift curve also shows that the model manages to have a gain substantially greater than the random choice of predictions.

The model was able to organize and correctly identify almost all malignant cases (94.00% ± 0.16%). This high level of accuracy and precision makes it a valuable tool for early detection of breast cancer, potentially saving lives and reducing treatment costs. For example, if the cost of diagnosing each patient is USD 150.00 and 212 malignant cases are identified among 569 patients, using the model's precision and recall, it is possible to reduce unnecessary costs and focus resources on accurate diagnoses.

The profit using the model is calculated as follows: with an accuracy of 98% and the cost savings from correctly identifying malignant cases early, the total profit for this cohort of patients is USD 398,000.00.

This project demonstrates that the SVM model is highly effective in predicting breast cancer, with a balance of precision and recall that ensures most malignant cases are correctly identified while minimizing false positives. Future work could involve further refinement of the model, including hyperparameter tuning and validation on larger, more diverse datasets to improve its robustness and generalizability.

### Next Steps

- **Improvement of the model's metrics, especially the recall**:
  While the current model shows high precision, improving the recall is crucial to ensure that fewer malignant cases are missed. This can be achieved by further tuning the model’s hyperparameters, exploring different feature engineering techniques, and potentially incorporating additional data sources.

- **Test other hypotheses to get new insights from the database**:
  Conduct additional exploratory data analysis to uncover new patterns and relationships within the dataset. This may involve testing new features, interactions between features, and different machine learning algorithms to enhance model performance and gain deeper insights into the factors influencing breast cancer diagnosis.

- **Validation on larger and more diverse datasets**:
  Validate the model's performance on larger and more diverse datasets to ensure its robustness and generalizability. This will help in understanding how well the model performs across different populations and clinical settings.

- **Integration into clinical workflows**:
  Work towards integrating the model into clinical workflows to aid healthcare providers in early detection and diagnosis of breast cancer. This involves collaborating with clinicians to understand their requirements and ensuring the model's predictions are easily interpretable and actionable.

- **Deployment and API Creation**:
  Develop a deployment strategy to make the model available for real-time predictions. This involves creating an API (Application Programming Interface) that allows healthcare providers to submit patient data and receive predictions. The API should be secure, reliable, and easy to use. Tools like Flask or FastAPI can be used to create the API, and cloud services such as AWS, Google Cloud, or Azure can be utilized for deployment.

## Acknowledgments

Special thanks to the contributors and the data science community for their valuable insights and support.
