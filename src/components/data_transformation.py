import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTranformationConfig:
    # Configuration for data transformation
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTranformation:
    def __init__(self):
        # Initialize data transformation configuration
        self.data_transformation_config = DataTranformationConfig()
        
    def get_data_transformation_object(self):
        try:
            # Define numerical columns
            numerical_colums=["writing_score", "reading_score"]
            
            # Define categorical columns
            categorical_columns=["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            
            # Create pipeline for numerical columns
            # 1. Impute missing values with median
            # 2. Scale values using StandardScaler
            num_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values
                    ('scaler', StandardScaler(with_mean=False))  # Scale values
                ]
            )
            
            # Create pipeline for categorical columns
            # 1. Impute missing values with most frequent value
            # 2. One-hot encode categorical values
            # 3. Scale values using StandardScaler
            cat_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
                    ('one_hot_encoder', OneHotEncoder()),  # One-hot encode categorical values
                    ('scaler', StandardScaler(with_mean=False))  # Scale values
                ]
            )
            
            # Log categorical and numerical columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_colums}")
            
            # Create ColumnTransformer to apply pipelines to respective columns
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipline, numerical_colums),  # Apply numerical pipeline
                    ('cat_pipeline', cat_pipline, categorical_columns)  # Apply categorical pipeline
                ]
            )
            return preprocessor
        except Exception as e:
            # Raise custom exception with error message and system information
            raise CustomException(e, sys)
        
    def initiate_data_tranformation(self , train_path,test_path):
        try:
            # Read train and test data from csv files
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Log completion of data reading
            logging.info("Read train and test data completed")
            
            # Log start of preprocessor object creation
            logging.info("Obtaining preprocessor object")

            # Get preprocessor object
            preprocessing_obj=self.get_data_transformation_object()
            
            # Define target column name
            target_column_name = "math_score"
            
            # Define numerical columns
            numerical_colums =['writing_score', "reading_score"]
            
            # Split data into input features and target features
            input_feature_train_df =train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df =test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Log start of preprocessing
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            
            # Apply preprocessing object on training data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # Apply preprocessing object on testing data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine input features and target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Log completion of data transformation
            logging.info("Data transformation completed")
            
            # Save preprocessor object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj =preprocessing_obj
            )
            
            # Return transformed data and preprocessor object file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            # Log exception
            logging.exception(e)
            # Raise custom exception with error message and system information
            raise CustomException(e, sys)