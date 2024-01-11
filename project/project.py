from pdb import set_trace
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

class my_model:

    #constructor method 
    def __init__(self, max_features=1000):
            # When initializing, we define the maximum number of features for TF-IDF vectorization
            self.max_features = max_features
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features, stop_words='english')        
            # Set class_weight to 'balanced' to handle class imbalance
            self.model = SGDClassifier(random_state=42, class_weight='balanced')  

    #method to fit model 
    def fit(self, X_trainingData, y):
    
        # Preprocess and clean the data
            # Fill missing values in X_trainingData with an empty string.
            X_trainingData.fillna('', inplace=True) 
            # Create new column 'combined_text' in X_trainingData by concatenating values from columns
            X_trainingData['combined_text'] = X_trainingData['title'] + ' ' + X_trainingData['location'] + \
                ' ' + X_trainingData['description'] + ' ' + X_trainingData['requirements']
            # Create new column 'cleaned_text' by applying the clean_text method to 'combined_text'.
            X_trainingData['cleaned_text'] = X_trainingData['combined_text'].apply(
                self.clean_text)                
            

        # Vectorize the text data using TF-IDF
            X_tfidf = self.tfidf_vectorizer.fit_transform(X_trainingData['cleaned_text'])
            
            #Create a DataFrame from the TF-IDF matrix.
            X_tfidf_dataframe = pd.DataFrame(
                X_tfidf.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())

            # Combine TF-IDF features with the numerical features
            numerical_features = X_trainingData[['telecommuting',
                                        'has_company_logo', 'has_questions']]
            X_combinedFeatures = pd.concat(
                [X_tfidf_dataframe, numerical_features.reset_index(drop=True)], axis=1)

            # Impute missing values using strategy as mean
            imputer = SimpleImputer(strategy='mean')
            X_combinedFeatures_imputed = imputer.fit_transform(X_combinedFeatures)


        # Use Pipelines for a cleaner workflow
            #Creating a pipeline with a preprocessor and the defined model.Pass the numerical features as columns
            pipeline = Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[
                        #('num', 'passthrough', [3, 4, 5]),
                        ('imputer', imputer, [0, 1, 2])
                    ],
                    remainder='passthrough'
                )),
                ('model', self.model)
            ])

        # Hyperparameter Tuning using GridSearchCV with StratifiedKFold
            # Defining a parameter grid for hyperparameter tuning
            param_grid = {
                'model__alpha': [0.0001, 0.001, 0.01, 0.1],
                'model__max_iter': [150, 200, 250],
                'model__loss': ['hinge', 'log_loss', 'modified_huber']
            }
            
            # Create GridSearchCV object with the pipeline and parameter grid and fit grid search on the input data and target variable.
            grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1')        
            grid_search.fit(X_combinedFeatures_imputed, y)

            # Access the best model found by the grid search
            best_modelDetermined = grid_search.best_estimator_

            # Set the final model
            self.model = best_modelDetermined

    #method to make predictions
    def predict(self, test_data):
            
        # Preprocess and clean new data
            #Fill missing values in test_data with an empty string.
            test_data.fillna('', inplace=True)

            # Create new column 'combined_text' in X_trainingData by concatenating values from columns
            test_data['combined_text'] = test_data['title'] + ' ' + test_data['location'] + \
                ' ' + test_data['description'] + ' ' + test_data['requirements']
            
            # Create new column 'cleaned_text' by applying the clean_text method to 'combined_text'.
            test_data['cleaned_text'] = test_data['combined_text'].apply(
                self.clean_text)
            
        #Transforming the text data using the pre-fit TF-IDF vectorizer
            X_tfidf = self.tfidf_vectorizer.transform(test_data['cleaned_text'])

            #Creating a DataFrame from the TF-IDF matrix.
            X_tfidf_dataframe = pd.DataFrame(
                X_tfidf.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
            
            # Combine TF-IDF features with the numerical features
            numerical_features = test_data[['telecommuting',
                                        'has_company_logo', 'has_questions']]            
            X_combinedFeatures = pd.concat(
                [X_tfidf_dataframe, numerical_features.reset_index(drop=True)], axis=1)
            
        # Predict using the fitted model
            predictions = self.model.predict(X_combinedFeatures)
            return predictions
    
    #method to clean text data.
    def clean_text(self, text):
            text = re.sub(r'#NAME\?', '', text)
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
