import os
import sys
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

## Intitialize the Data Ingetion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create class for Data Ingetion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingetion method starts")
        try:
            df=pd.read_csv(os.path.join('notebook/data','UCI_Credit_Card.csv'))
            df.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)
            df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)

            ## For the Education column we have unique value [2,1,3,4,5,6,0] but in our given data description we have only Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). so we need to change them to 4 which is equal to **others** so we are changing 5,6,0 to 4
            fil = (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | (df['EDUCATION'] == 0)
            df.loc[fil, 'EDUCATION'] = 4

            ## For the Marriage column we have unique value [1,2,3,0] but in our given data description we have only Marriage (1 = married; 2 = single; 3 = others). so we need to change them to 3 which is equal to **others** so we are changing 0 to 3
            fil = df['MARRIAGE'] == 0
            df.loc[fil, 'MARRIAGE'] = 3
            
            logging.info('Dataset read as pandas Dataframe')
            


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            

            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of Data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingetion stage')
            raise CustomException(e,sys)
        
