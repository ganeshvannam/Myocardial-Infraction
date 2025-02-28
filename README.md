# Myocardial Infarction Complications Prediction - R Project

I worked on this project as part of my Predictive Modelling course final project. This course was a game-changer for me, offering deep insights into R and the mathematics behind predictive algorithms.

## Why This Project?
The dataset we chose is open-source and available [here](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications).  
Despite heart disease prediction being a common theme in the data science world, this dataset presented its own complexities and challenges, making it an exciting learning experience.

## Project Workflow:

### 1️⃣ Preprocessing the Data:
- Removed degenerate columns.
- Dealt with missing values, skewness, and inconsistencies.
- Handled categorical predictors and created dummy variables.
- Reduced dimensionality to streamline the dataset.

### 2️⃣ Data Splitting:
- Performed a stratified split due to the highly imbalanced target variable (presence/absence of complications).
- Used a standard 80%-20% split for training and testing datasets.

### 3️⃣ Evaluation Metric:
- Selected **Kappa** as the classification metric to ensure fairness for the imbalanced dataset.

### 4️⃣ Model Building:
- Developed both linear and non-linear classification models.
- The **Partial Least Squares Discriminant Analysis (PLS-DA)** model emerged as the best performer with the highest Kappa value.

## Files Included:
- **Myocardial_infarction_complications_Database**: Contains predictors and target variables.
- **Descriptive statistics**: Detailed explanations of each feature.
- **Final Report**: Comprehensive analysis of our approach and results.
- **Linear_Models**: Contains R code for Linear Models.
- **Non_linear_models**: Contains R code for Non-Linear Models.
- **Predictive_Modelling_PPT**: Summary of the project for our class presentation.

## Collaboration:
This project was a collaborative effort with my teammate **Nandhika Rajmanikandan**. Working together made the entire process enjoyable and insightful.

## Key Learnings:
- Tackling imbalanced datasets in medical data science.
- The power of R in predictive modeling.
- How to use metrics like Kappa effectively for model evaluation.

For more details, check out our files!

Happy Predicting!
