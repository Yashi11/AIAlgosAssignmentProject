# AIAlgosAssignmentProject

In this project, we aim to predict the marks of various students on th basis of their nuber of courses and total study hours. Same has been iplemented using SparkML algorithm:
DecisionTreeRegressionModel and GeneralizedLinearRegressionModel

# Decision Tree Algorithm in Spark

Decision trees are a popular family of classification and regression methods. More information about the spark.ml implementation can be found further in the section on decision trees.
# Generalized Linear Regression Algorithm in Spark

Contrasted with linear regression where the output is assumed to follow a Gaussian distribution, generalized linear models (GLMs) are specifications of linear models where the response variable Yi follows some distribution from the exponential family of distributions. Spark’s GeneralizedLinearRegression interface allows for flexible specification of GLMs which can be used for various types of prediction problems including linear regression, Poisson regression, logistic regression, and others. Currently in spark.ml, only a subset of the exponential family distributions are supported and they are listed below.

# Our Dataset

The data consists of Marks of students including their study time & number of courses. The dataset is downloaded from UCI Machine Learning Repository.
### Dataset
<img width="260" alt="Dataset" src="https://user-images.githubusercontent.com/73705143/206892837-543f1c3a-e986-4e77-92d4-ca6f2ec5fda5.png">

Properties of the Dataset: \
Number of Instances: 100\
Number of Attributes: 3 including the target variable.
### Dataset Summary
<img width="484" alt="DatasetSummary" src="https://user-images.githubusercontent.com/73705143/206892821-a41cd4bf-934a-426d-9090-85129a465fa5.png">


The project is simple yet challenging as it is has very limited features & samples. Can you build regression model to capture all the patterns in the dataset, also maitaining the generalisability of the model?
Objective:

    Understand the Dataset & cleanup (if required).
    Build Regression models to predict the student marks wrt multiple features.
    Also evaluate the models & compare their respective scores like R2, RMSE, etc.

# Outputs
### Dataset After Feature Assembling
<img width="344" alt="DatasetAfterFeatureAssembling" src="https://user-images.githubusercontent.com/73705143/206892839-a622e022-e31e-4fe0-bcd3-2631a99efee8.png">

### Decision Tree Predictions
<img width="476" alt="Predictions" src="https://user-images.githubusercontent.com/73705143/206892834-d8d66300-5afd-421b-a4b4-d423ef50b7d3.png">

### Generalized Linear Regression Predictions 
<img width="476" alt="Predictions" src="https://user-images.githubusercontent.com/73705143/206892836-77ada236-c389-406a-bd3a-e01382dbfe67.png">

### Decision Tree Results
<img width="474" alt="ErrorMetricsResult" src="https://user-images.githubusercontent.com/73705143/206892826-7ef6f8fc-7a33-46ef-88fe-a1a13cc091f9.png">

### Generalized Linear Regression Results
<img width="474" alt="ErrorMetricsResult" src="https://user-images.githubusercontent.com/73705143/206892827-f95f1cf3-90af-4f9d-b85f-b1e69bf49b03.png">
<br>
<img width="474" alt="ErrorMetricsResult" src="https://user-images.githubusercontent.com/73705143/206892830-7feae1ce-f3d9-4e03-ba81-b9aa3d0181e3.png">
<br>
<img width="474" alt="ErrorMetricsResult" src="https://user-images.githubusercontent.com/73705143/206892831-63c983f0-d230-4e97-80f9-7ae636f834b4.png">
<img width="474" alt="ErrorMetricsResult" src="https://user-images.githubusercontent.com/73705143/206920407-3a78b692-176b-43e8-819d-baeed64181b6.png">

<!---
<img width="484" alt="DatasetSummary" src="https://user-images.githubusercontent.com/73705143/206892829-92b366c6-4d41-4746-80e6-29bc6ff27f5a.png">
 <img width="344" alt="DatasetAfterFeatureAssembling" src="https://user-images.githubusercontent.com/73705143/206892839-a622e022-e31e-4fe0-bcd3-2631a99efee8.png"> 
--->
# Steps to run this project
1. Set up spark on your device.
2. Create a Maven Project on Eclipse or IntelliJ IDEA.
3. Copy java files from this project to your project.
4. Give the correct path of the dataset.
<br/>
