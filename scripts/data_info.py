# import 
import pandas as pd

class DataInfo:
    def __init__(self, df):
        self.df = df.copy()
# shape of the dataframe
    def shape_df(self):
         '''
         Display number of rows and columns in the given Dataframe
         '''
         print(f"Dataframe contains {self.df.shape[0]} rows and {self.df.shape[1]} columns")
         #return (self.df.shape[0],self.df.shape[1])