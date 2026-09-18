# 🏥 Hospital Readmission Prediction

A machine learning project that predicts whether a patient will be *readmitted to the hospital within 30 days* using *Logistic Regression with L2 regularization*.

## 📌 Objective

The goal of this project is to build a binary classification model that predicts:

- Yes → Patient will be readmitted within 30 days
- No → Patient will not be readmitted within 30 days

## 📊 Dataset

The dataset contains *30,000 patient records* and *12 features*.

### Features

- patient_id
- age
- gender
- blood_pressure
- cholesterol
- bmi
- diabetes
- hypertension
- medication_count
- length_of_stay
- discharge_destination
- readmitted_30_days — Target variable

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab
- Logistic Regression

## 🔄 Workflow

1. Load the dataset
2. Perform basic data exploration
3. Check for missing values
4. Remove the patient_id column
5. Separate features and target variable
6. Encode categorical variables using one-hot encoding
7. Split the dataset into training and testing sets
8. Standardize features using StandardScaler
9. Train Logistic Regression with L2 regularization
10. Generate predictions and probabilities
11. Evaluate the model
12. Plot the ROC curve

## 🤖 Model

The project uses Logistic Regression:

```python
LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=2000,
    class_weight="balanced"
)

hospital
