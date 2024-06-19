# Lone_approval_prediction

Loan Approval Prediction Model

This repository contains a Python script that develops a model to predict loan approval based on various applicant attributes. The dataset used includes features such as applicant income, loan amount, credit history, and more.

File: Loan Approval prediction model.py

Description

The Loan Approval prediction model.py script performs the following tasks:

1. *Data Loading and Initial Exploration:*
   - Imports necessary libraries (pandas, numpy, matplotlib, seaborn, sklearn).
   - Loads the dataset (lone.csv) and performs initial data exploration, including displaying the first few rows and basic statistics.

2. *Data Preprocessing:*
   - Handles missing values by filling them with appropriate measures (mean, mode).
   - Creates new features (loanamount_log, Totalincome, Totalincome_log).

3. *Data Visualization:*
   - Uses Seaborn to create count plots for categorical variables like Gender, Married, and Dependents.

4. *Data Encoding and Splitting:*
   - Encodes categorical features using LabelEncoder.
   - Splits the data into training and testing sets using train_test_split.

5. *Feature Scaling:*
   - Scales features using StandardScaler.

6. *Model Training and Evaluation:*
   - Trains multiple models including:
     - Random Forest Classifier
     - Gaussian Naive Bayes
     - Decision Tree Classifier
     - K-Neighbors Classifier
   - Evaluates models using accuracy score.

Usage

To run this script, ensure you have the required libraries installed. You can install them using pip:
bash
pip install pandas numpy matplotlib seaborn scikit-learn

Load the dataset lone.csv into the appropriate directory, then execute the script.
