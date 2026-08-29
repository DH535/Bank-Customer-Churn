# Bank Customer Churn Prediction

An end-to-end machine learning project that predicts whether a bank customer is likely to leave the bank (churn).

The project covers the complete ML workflow, from data exploration and preprocessing to model selection, tuning, evaluation, and deployment with Streamlit.

---

## Project Overview

Customer churn is an important challenge for banks because retaining existing customers can be more cost-effective than acquiring new ones.

This project uses customer demographic, financial, behavioral, and engagement information to predict the likelihood of customer churn.

The target variable is:

* `Exited = 0` → Customer stays
* `Exited = 1` → Customer leaves

The dataset contains **10,000 customer records** and 18 original columns.

---

## Objectives

* Explore and understand customer data.
* Identify important factors associated with churn.
* Prepare the data for machine learning.
* Handle class imbalance.
* Compare multiple classification algorithms.
* Select and tune the best-performing model.
* Evaluate the final model using appropriate classification metrics.
* Tune the classification probability threshold.
* Save the trained model pipeline.
* Deploy the model through a Streamlit application.

---

## Dataset

The dataset contains information about bank customers, including:

| Feature              | Description                              |
| -------------------- | ---------------------------------------- |
| `CreditScore`        | Customer's credit score                  |
| `Geography`          | Customer's country                       |
| `Gender`             | Customer's gender                        |
| `Age`                | Customer's age                           |
| `Tenure`             | Number of years with the bank            |
| `Balance`            | Customer's bank balance                  |
| `NumOfProducts`      | Number of bank products used             |
| `HasCrCard`          | Whether the customer has a credit card   |
| `IsActiveMember`     | Whether the customer is an active member |
| `EstimatedSalary`    | Estimated customer salary                |
| `Exited`             | Churn target variable                    |
| `Satisfaction Score` | Customer satisfaction score              |
| `Card Type`          | Customer's card type                     |
| `Point Earned`       | Customer reward points                   |

`RowNumber`, `CustomerId`, and `Surname` were removed because they do not provide useful predictive information for the model.

The `Complain` feature was also removed during feature preparation.

---

## Exploratory Data Analysis

The dataset was examined for:

* Missing values
* Duplicate records
* Data types
* Unique values
* Numerical distributions
* Categorical distributions
* Target-class distribution
* Relationships between features and churn

The target distribution was:

| Exited | Customers |
| -----: | --------: |
|      0 |     7,962 |
|      1 |     2,038 |

This represents an imbalanced classification problem, with significantly fewer customers who churned.

---

## Data Preprocessing

The preprocessing workflow included:

### Removing unnecessary features

The following columns were removed:

```text
RowNumber
CustomerId
Surname
Complain
```

### Encoding categorical variables

`Geography` and `Gender` were one-hot encoded.

`Card Type` was encoded as:

```text
SILVER   → 1
GOLD     → 2
PLATINUM → 3
DIAMOND  → 4
```

### Feature Scaling

`StandardScaler` was used to standardize numerical features.

### Handling Class Imbalance

Several approaches were investigated:

* No balancing
* SMOTE
* Random Oversampling
* Random Undersampling

SMOTE provided a better balance between churn precision and recall and was therefore used during model experimentation.

---

## Models Evaluated

The following classification algorithms were compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* AdaBoost
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* XGBoost

Model evaluation used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Because churn detection is the main objective, **F1-score, Recall, and ROC-AUC** were given more attention than accuracy alone.

---

## Model Selection

After cross-validation and hyperparameter tuning, **Gradient Boosting** achieved the strongest overall performance among the tested models.

The final cross-validation results were approximately:

| Metric    | Score  |
| --------- | -----: |
| Accuracy  | 0.8417 |
| Precision | 0.6083 |
| Recall    | 0.6285 |
| F1-score  | 0.6181 |
| ROC-AUC   | 0.8559 |

The final model was then evaluated on the unseen test set.

### Final Test Results

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.855 |
| Precision |  0.65 |
| Recall    |  0.63 |
| F1-score  |  0.64 |
| ROC-AUC   | 0.869 |

The results show that the model can identify a meaningful portion of customers at risk of churn while maintaining reasonable precision.

---

## Model Deployment

The final preprocessing and prediction pipeline was saved as:

```text
Deployment/
└── best_gb.pkl
```

The saved pipeline contains the necessary preprocessing and trained Gradient Boosting model.

This allows the deployment application to receive new customer information and perform the required transformations before generating a prediction.

---

## Streamlit Application

A Streamlit application is included/planned for deployment.

The application provides:

### Customer Risk Assessment

Users can enter customer information and receive:

* Churn prediction
* Churn probability
* Stay probability
* Risk level

### Batch Prediction

Users can upload a CSV containing multiple customers and receive predictions for the entire dataset.

### Model Insights

The application provides model performance information and feature importance to help users understand the model.

### Prediction Explanation

Where supported, the application can provide insights into which customer characteristics pushed the model's prediction toward higher or lower churn risk.

These explanations represent **model behavior**, not certainty about the customer's actual intentions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Seaborn
* Plotly
* Joblib
* Streamlit

---

## Project Structure

```text
Bank-Customer-Churn/
│
├── Deployment/
│   └── best_gb.pkl
│
├── app.py
│
├── notebooks/
│   └── bank_customer_churn.ipynb
│
├── data/
│   └── bank_customer_churn.csv
│
├── requirements.txt
│
└── README.md
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd Bank-Customer-Churn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

---

## Important Notes

This model is a machine learning prediction system and should not be interpreted as knowing whether a customer will definitely leave.

The predicted probability represents the model's estimated likelihood based on patterns learned from the training data.

Model performance may differ when applied to new data from a different population, time period, or banking environment.

---

## Key Learning Outcomes

This project demonstrates practical experience with:

* Exploratory Data Analysis
* Feature engineering
* Categorical encoding
* Feature selection
* Imbalanced classification
* SMOTE
* Feature scaling
* Cross-validation
* Model comparison
* Hyperparameter tuning
* Classification evaluation
* Model interpretation
* Model serialization
* Machine learning deployment with Streamlit

---

## Author

**Doha Hesham**

This project was developed as an end-to-end machine learning practice project focused on understanding the complete journey from raw customer data to a deployable ML application.
