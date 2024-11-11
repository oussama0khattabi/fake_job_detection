# 📊 Fraudulent Job Postings Detection Project

## 📝 Project Overview
This project aims to build a machine learning model to detect **fraudulent job postings** using data-driven techniques. The goal is to develop a classifier that can accurately identify fake job listings to protect job seekers from scams. We explored several models, including **Logistic Regression**, **Support Vector Machines (SVM)**, **Random Forest**, and **Multinomial Naive Bayes**.

## 📁 Dataset Source
The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com) under the dataset titled **"Fake Job Postings Prediction"**. This dataset includes information on various job postings with features that can help identify fraudulent listings. The dataset includes the following key columns:
- **job_id**: Unique identifier for each job posting
- **title**: Job title
- **location**: Location of the job (country, city)
- **department**: Department name (if provided)
- **company_profile**: Description of the company
- **description**: Detailed job description
- **requirements**: Required qualifications for the job
- **benefits**: Benefits offered to the employee
- **employment_type**: Type of employment (Full-time, Part-time, etc.)
- **required_experience**: Required experience level
- **required_education**: Education level needed
- **fraudulent**: Target variable (1 = Fraudulent, 0 = Legitimate)

The dataset is relatively balanced, with approximately **17,880** entries.


## ⚙️ Data Preparation
The data preparation phase involved the following steps:

1. **Loading and Understanding the Data**:
   - Inspected the dataset for missing values, duplicates, and data types.
   - Performed **exploratory data analysis (EDA)** to understand feature distributions.

2. **Data Cleaning**:
   - Removed duplicates and filled in missing values for key columns.
   - Extracted and standardized location information to analyze job postings by country.

3. **Text Preprocessing**:
   - Applied **TF-IDF Vectorization** on the `description` and `requirements` columns to convert text data into numerical format.
   - Tokenized, lemmatized, and removed stop words to clean the text data.
   - Encoded categorical features like `employment_type` and `required_experience` using **One-Hot Encoding**.

4. **Feature Engineering**:
   - Created new features, such as the **length of job descriptions** and **word counts**.
   - Analyzed correlations between text length and fraudulence to discover patterns.

5. **Train-Test Split**:
   - Split the dataset into training (70%) and testing (30%) sets using stratified sampling to maintain class balance.

## 🚀 Model Training
The following models were trained on the processed dataset:

1. **Logistic Regression**
   - A simple yet effective linear classifier that predicts probabilities based on logistic functions.
   - Tuned using `GridSearchCV` to optimize hyperparameters like `C` and `class_weight`.

2. **Support Vector Machine (SVM)**
   - A robust classifier that uses hyperplanes to separate data points.
   - The best model used a **linear kernel** with `C=0.1` and `class_weight='balanced'`.

3. **Random Forest Classifier**
   - An ensemble learning method using multiple decision trees to improve classification accuracy.
   - Tuned with `n_estimators=200`, `max_depth=10`, `min_samples_split=10`, and `max_features='sqrt'`.

4. **Multinomial Naive Bayes**
   - A probabilistic classifier particularly effective for text data.
   - Applied after transforming text features using **TF-IDF**.

## 🔍 Model Evaluation
We evaluated the models based on **accuracy**, **precision**, **recall**, and **F1-score** to find the best model. Given the importance of minimizing **false negatives** (to catch all fraudulent postings), we prioritized recall during the evaluation.

### Evaluation Results Summary
| Model                     | Recall | Precision | F1 Score |
|---------------------------|--------|-----------|----------|
| Logistic Regression       | 0.69   | 0.98      | 0.76     |
| Support Vector Machine    | 0.94   | 0.79      |   0.85   |
| Random Forest             | 0.82   | 0.99      | 0.89     |
| Multinomial Naive Bayes   |        | 0.95      |          |

### Selected Model: Support Vector Machine (SVM)
- The **SVM model** with a linear kernel and balanced class weights achieved the highest recall while minimizing false negatives. This was critical for our project since missing fraudulent postings would have serious consequences.

