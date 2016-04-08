# @ author : Cristina Menghini ;
# StudentID: 1527821

import pandas as pd
import numpy as np
import random

class split_dataset(object):
    """ This class defines the similarities between the active user and the others."""
    
    def __init__(self, data, data_array, test):
        
        """An instance created by that class is characterized by the following attributes:
        - data : is the datatest in a DataFrame format,
        - data_array: is the array of the entire dataset, the return of *load_data.adjust_data method*,
        - test : is the percentage of data that compose the test set."""
        
        self.data = data
        self.data_array = data_array
        self.test = test
        
    def train_test(self): 
        
        """This method returns:
        - [0] the train rating matrix,
        - [1] the test data,
        - [2] indices_test,
        - [3] indices_train,
        - [4] N_i : the number of items,
        - [5] N_u : the number of users."""
        
        # Compute the number of Items:
        N_i = len(set(self.data_array[:,1])) 
        # Users
        N_u = max(set(self.data_array[:,0])) + 1 
        
        # Length of test and train set:
        len_test = (len(self.data_array)/100)*self.test
        len_train = len(self.data_array)-len_test
        
        # Choose random row indices
        indices_test = sorted(random.sample(self.data.index, len_test))
        # Obtain the list of train set indices
        indices_train = sorted(set(self.data.index).difference(indices_test))
        
        # Define the empty train rating matrix
        train_rating = np.full((N_i,N_u), 0)
        # Fill in the matrix
        for index in indices_train:
            train_rating[self.data_array[index, 1]][self.data_array[index, 0]] = self.data_array[index, 2]
        
        # Define the test dataset
        data_test = self.data_array[indices_test]
        
        return train_rating , data_test , indices_test, indices_train, N_i, N_u
