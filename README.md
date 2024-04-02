# Fraudulent-Job-Posting-Project
Project Title: Job Posting Authenticity Prediction

Description:

The project aims to train a model to predict whether a job posting is real or fake using the provided dataset. The solution implemented leverages Stochastic Gradient Descent (SGD) Classifier, which preprocesses and vectorizes textual data with TF-IDF and numerical features. The model undergoes hyperparameter tuning via GridSearchCV for optimal performance.

Restrictions Set:
Only packages allowed for installation are scikit-learn, gensim, pandas, and numpy.

Data Preprocessing:
The data in the dataset is cleaned by removing stop words, punctuations, special characters, and digits. Additionally, the text is converted to lowercase. Missing values in the input training data (X_trainingData) are filled with empty strings. Specifically, the pattern "#NAME?" is removed. New columns are created by concatenating values from the text features.

Model Training:
Model Used: Stochastic Gradient Descent Classifier with balanced class weights
TF-IDF vectorization is performed on the combined text data and combined with numerical features.
Missing values are imputed using the mean.
A pipeline is constructed, including a preprocessor and the defined model.
Hyperparameter tuning is performed using GridSearchCV with Stratified K-fold cross-validation.
The best model found is accessed and set as the final model.

Prediction:
The same pre-processing and vectorizing technique applied during training in the fit method are applied in the predict method. The fitted model is then used to predict whether a job is fake or not.

Model Evaluation:
Sample output using the dataset:
F1 Score: 0.7693467
Runtime: 58.3 seconds

The F1 score ranges from 0.74 - 0.8 marking an average accuracy of approximately 76%