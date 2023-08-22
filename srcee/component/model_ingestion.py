import os
import sys
from srcee.exception import CustomException
from srcee.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
from srcee.component.model_transformation import DataTransformation
from srcee.component.model_training import ModelTrainer

#Iitialize data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion method starts")
            df = pd.read_csv(os.path.join('notebook/data/insurance.csv'))
            logging.info('Dataset read as pandas Dataframe')

            logging.info('Make directories')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train test split")
            train_set, test_set = train_test_split(df,test_size=.20,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info('Data Ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info("Error raise in data ingestion")
            raise CustomException(e,sys)
        

