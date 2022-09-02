import sys
import os
from ensurepip import version
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from sklearn.model_selection import train_test_split
from sklearn import ElasticNet
from urllib.parse import urlparse
import mlflow
#import mlflow.sklearn
import logging
import warnings

logging.basicConfig (level = logging .WARN )
logger= logging.getLogger (__name__ )

#Get url from DVC
import dvc.api

path = "data/AdSmartABdata.csv"
repo = 'C:/Users/Ekubay/Documents/smart_ad_ABtesting'
version - 'v3'
data_url = dvc.api.get_url (
 path = path ,
 repo = repo,
 rev= version
)
mlflow.set_experiment('advert')
def eval_metrics ( actual , pred ) :
    rnse = np.sqrt ( mean_squared_error ( actual , pred ) )
    mae = mean_absolute_error ( actual , pred )
    r2 = r2_score ( actual , pred )
    return rnse , mae , r2

# main 
if __name__ == "__main__":
    warnings.filterwarnings ( " ignore " )
    np.random.seed ( 40 )
# Read the samrt ad from the remote repository
data = pd . read_csv ( data_url , sep = " , " )
# Log data params
mlflow.log_param ( ' data_url ' , data_url )
mlflow.log_param ( ' data version ' , version )
mlflow.log_param ( ' input_rows ' , data.shape [0] )
mlflow.log_param ( ' input_cols ' , data.shape [ 1 ])


# Split the data into training and test sets . ( 0.75 , 0.25 ) split .
train , test = train_test_split( data )
#The predicted column is " response " which is a scalar from 13 , 91
train_x = train.drop ( [ "response" ] , axis = 1 )
test_x= test.drop ( [ " response "] , axis = 1)
train_y= train [[ " response "]]
test_y = test [[ " response " ]]

#Log artifacts : columns used for modeling
cols_x = pd . DataFrame ( list ( train_x.columns ) )
cols_x.to_csv ('features.csv ' , header = False , index = False )
mlflow.log_artifact ('features.csv ' )

cols_y = pd.DataFrame ( list ( train_y.columns ) )
cols_y.to_csv (' targets.csv ' , header = False , index = False )
mlflow.log_artifact ( ' targets.csv ' )

alpha = float ( sys.argv [ 1 ] ) if len ( sys.argv ) > 1 else 0.5
l1_ratio = float ( sys.argv [ 2 ] ) if len ( sys.argv ) > 2 else 0.5
lr = ElasticNet(alpha = alpha , l1_ratio = l1_ratio , random_state = 42 )
lr.fit ( train_x , train_y )
