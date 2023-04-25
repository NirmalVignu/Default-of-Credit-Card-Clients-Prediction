import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def train_model():
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    report=model_trainer.initate_model_training(train_arr,test_arr)
    return report

# if __name__=='__main__':
#     try:
#         logging.info('Training Pipeline Started')
#         report=train_model()
#         logging.info('Training Pipeline Completed')
#         print(report)
#     except Exception as e:
#         logging.info('Exception occured during training pipeline')
#         raise CustomException(e,sys)


    


