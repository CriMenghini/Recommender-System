# @ author : Cristina Menghini ;
# StudentID: 1527821

import pandas as pd
import numpy as np


class load_data(object):
    """This class define the tables of data that I'm going to use during the computation."""
    
    def __init__(self, data, separated):
        
       
        """An instance created using this class is characterized by the following attributes:
        - data : text file that contains yout dataset,
        - separated : string of the character that separates columns."""
        
        self.data = pd.read_csv(data, sep = separated)
        self.separated = separated
        
    def adjust_data(self):
        
        """This method returns the data( type : array)[0] mapped in a different way in order to use arrays and solve the following problems:
        - Some MovieIDs do not correspond to a movie due to accidental duplicate entries and/or test entries
        - Movies are mostly entered by hand, so errors and inconsistencies may exist,
        and the dataframe[1]."""
        
        datas = self.data
        
        # Define the new lables for the movies
        new_indices = {i : j for i,j in zip( sorted(list(set(self.data['item id']))),range(len(set(self.data['item id']))) )}
        # Change the column of item id
        self.data['item id'] = self.data['item id'].map(new_indices)
        # Shift the values of *user id* in order to built the matrix of ratings quickly
        try:
            self.data['user id'] = self.data['user id'] - 1
        except:
            pass
        # Transform the dataset into array
        data_array = np.array(self.data)
        
        return data_array , datas
