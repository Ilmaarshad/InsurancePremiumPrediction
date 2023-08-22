import os
import sys
from srcee.logger import logging
from srcee.exception import CustomException
import pandas as pd

from srcee.component.model_ingestion import DataIngestion
from srcee.component.model_transformation import DataTransformation
from srcee.component.model_training import ModelTrainer




if __name__ == '__main__':
    ingestion = DataIngestion()
    train_data,test_data = ingestion.initiate_data_ingestion()
    trans = DataTransformation()
    train_arr,test_arr,_ = trans.initiate_data_transformation(train_data,test_data)
    trainer = ModelTrainer()
    trainer.initiate_model_training(train_arr,test_arr)
