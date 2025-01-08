# Import necessary libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

# Import necessary modules from scikit-learn library
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTranformation
from src.components.data_transformation import DataTranformationConfig

# Define a dataclass to store data ingestion configuration
@dataclass
class DataIngestionConfig:
    # Define the path for train data
    train_data_path: str = os.path.join('artifacts','train.csv')
    # Define the path for test data
    test_data_path: str = os.path.join('artifacts','test.csv')
    # Define the path for raw data
    raw_data_path: str = os.path.join('artifacts','data.csv')
    
# Define a class for data ingestion
class DataIngestion:
    # Initialize the data ingestion configuration
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    # Method to initiate data ingestion
    def initiate_data_ingestion(self):
        # Log the start of data ingestion
        logging.info("Entering data ingestion")
        try:
            # Read the data from the specified CSV file
            # This line reads the data from a CSV file named 'stud.csv' in the 'notebook/data' directory
            df = pd.read_csv('notebook/data/stud.csv')
            # Log the completion of data reading
            logging.info("Reading data")
            # Create the directory for test data if it does not exist
            # This line creates the directory for test data if it does not already exist
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path) , exist_ok=True)
            
            # Save the raw data to the specified CSV file
            # This line saves the raw data to a CSV file named 'data.csv' in the 'artifacts' directory
            df.to_csv(self.ingestion_config.raw_data_path,index=False , header=True)
            
            # Log the start of train test split
            logging.info("Train test split initiated")
            # Split the data into train and test sets
            # This line splits the data into train and test sets with a test size of 0.2 and a random state of 42
            train_set , test_set = train_test_split(df ,test_size=0.2,random_state=42)
            
            # Save the train data to the specified CSV file
            # This line saves the train data to a CSV file named 'train.csv' in the 'artifacts' directory
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            # Save the test data to the specified CSV file
            # This line saves the test data to a CSV file named 'test.csv' in the 'artifacts' directory
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            
            # Log the completion of data ingestion
            logging.info("Ingestion of the data is completed")
            
            # Return the paths for train and test data
            # This line returns the paths for the train and test data
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            # Raise a custom exception if any error occurs
            # This line raises a custom exception if any error occurs during data ingestion
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj =DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTranformation()
    data_transformation.initiate_data_tranformation(train_data, test_data)