import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from srcee.exception import CustomException
from srcee.logger import logging
from dataclasses import dataclass
from srcee.utils import save_object

@dataclass
class DataTransformationConfig:
    pickle_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('Data Transformation initiated')

            categorical_feature = ['region','sex','smoker']
            numerical_feature = ['age', 'bmi', 'children']


            logging.info('Pipeline initiated')

            numerical_pipeline = Pipeline([
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                        ])


            categorical_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehotencode', OneHotEncoder(sparse=False,handle_unknown="ignore")),
                        ('scaler',StandardScaler(with_mean=False))

                        ])

            preprocessor = ColumnTransformer([  
                        ('numerical_pipeline',numerical_pipeline,numerical_feature),
                        ('categorical_pipeline',categorical_pipeline,categorical_feature)
                        ])

            return preprocessor
        
            logging.info('Pipeline completed')


        except Exception as e:
            logging.info('Error raised in get data transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocesser_obj = self.get_data_transformation_obj()


            train_feature_df = train_df.drop(columns='expenses',axis=1)
            target_train_df = train_df['expenses']

            test_feature_df = test_df.drop(columns='expenses',axis=1)
            target_test_df = test_df['expenses']

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_feature_arr = preprocesser_obj.fit_transform(train_feature_df)
            test_feature_arr = preprocesser_obj.transform(test_feature_df)
            
        
            train_arr = np.c_[train_feature_arr, np.array(target_train_df)]
            test_arr = np.c_[test_feature_arr, np.array(target_test_df)]



            save_object(
                file_path = self.data_transformation_config.pickle_path,
                obj=preprocesser_obj

            )
            logging.info('pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.pickle_path
            )

        except Exception as e:
            logging.info('Error raise in initiate data transformation')
            raise CustomException(e,sys)
