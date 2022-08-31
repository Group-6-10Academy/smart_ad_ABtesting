import sys,os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, '../sctiprs/')
sys.path.append(os.path.abspath(os.path.join('scripts')))

from data_Preprocess import CleanDataFrame
from data_info import DataInfo

df = pd.DataFrame({'numbers': [2,4,6,7,9], 'letters':['a','b','c','d','e'],
                   'floats': [0.2323, -0.23123,np.NaN, np.NaN, 4.3434]})

class Tester(unittest.TestCase):
    
    def test_class(self):  #test for creation of the class for 'data information teller'
        data = DataInfo(df)
        self.assertEqual(df.info(),data.df.info())
        
    def test_drop_duplicates(self):
        data = CleanDataFrame(df)
        data.drop_duplicates(df)
        self.assertEqual(df.info(),data.df.info())
        
    def test_detail_info(self):
        data = DataInfo(df)
        self.assertEqual(data.df.info(),df.info())
        
    def test_list_column_names(self):
        data = CleanDataFrame(df)
        self.assertTrue(data.df.isna().sum().sum() !=0)
        
if __name__ == '__main__':
    unittest.main() 
        
    
    