import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        performance = classification_report(y_test,y_pred)
        auc = roc_auc_score(y_test,model.predict(X_test))
        cm = confusion_matrix(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)

        report = {}
        report['accuracy'] = accuracy
        report['performance'] = performance
        report['auc'] = auc
        report['cm'] = cm
        report['precision']=precision
        report['recall']=recall
        report['f1']=f1
        print(report)
        logging.info('Model evaluation completed')
        logging.info(report)

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)