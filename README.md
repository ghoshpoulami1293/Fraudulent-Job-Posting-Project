# Fraudulent-Job-Posting-Project
Predict whether a job posting is real or fake

Goal:
Train a model to predict whether a job posting is real or fake on the provided dataset.

Solution Implemented: 
Leveraged the SGDClassifier in unsupervised learning to determine job posting authenticity. 
Applied stochastic gradient descent optimization and NLP techniques to predict whether a job posting is genuine or fake.


Restrictions set : 
    Can only install packages: scikit-learn, gensim, pandas, numpy


Data Preprocessing
The data in the dataset  is cleaned. 
    Stop words, punctuations, special characters and digits are removed and text is converted to lowercase.
    Missing values in the input training data (X_trainingData) are filled with empty strings.
    Specifically removed the pattern #NAME\?
    New columns are created by concatenating values from the text features

Model Training
    Model used : Stochastic Gradient Descent Classifier with balanced class weights 
    Performed TF-IDF vectorization on the combined text data and combines it with numerical features.
    Imputed missing values using the mean.
    Constructed a pipeline, including a preprocessor and the defined model.
    Perform hyperparameter tuning using GridSearchCV with Stratified K-fold cross-validation.
    Access the best model found and set it as the final model

Prediction
    Apply the same pre-processing and vectorizing technique in the predict method (as applied while training in the fit method) , and predict using fitted model , whether a job is fake or not.

Model Evaluation:
    Sample output using dataset:
        F1 score: 0.693467
        Runtime: 58.3 seconds





