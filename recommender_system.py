# @ author : Cristina Menghini ;
# StudentID: 1527821

import numpy as np
import pandas as pd
import random
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
import math

class items_similarity(object):
    """ This class defines the similarities between the items."""
    
    def __init__(self, train_rating):
        
        """An instance created by that class is characterized by the following attributes:
        - train_ratings : is the matrix of ratings related to the training set, return[0] of *split_dataset.train_test*."""
        
        self.train_rating = csr_matrix(train_rating) # Compress the matrix of ratings
        
    def cosine_similarity(self):
        
        """This method returns the cosine similarity matrix."""
        
        # Compute the matrix and subtract it from 1 to get the similarity.
        cosine_similarity_matrix = 1-pairwise_distances(self.train_rating, metric = 'cosine')
        
        return cosine_similarity_matrix


class baseline_predictor(object):
    """ This class defines the baseline predictor for each pair of user-item."""
    
    def __init__(self, train_rating, data_array, indices_train, N_i, N_u):
        
        """An instance created by that class is characterized by the following attributes:
        - train_rating : is the matrix of ratings related to the training set, return[0] of *split_dataset.train_test*,
        - data_array : is the array of the entire dataset, the return of *load_data.adjust_data method*,
        - indices_train : is the list of index which the training set is composed by, it's the return[3] of the *split_dataset.train_test*.
        - N_i : is the number of items, in particular it is the return[4] of the methos *split_dataset.train_test()[4]*,
        - N_u : is the number of users, in particular it is the return[5] of the methos *split_dataset.train_test()[5]*."""
        
        
        self.train_rating = train_rating 
        self.data_array = data_array
        self.indices_train = indices_train
        self.N_i = N_i
        self.N_u = N_u
        
    def baseline(self):
        
        """This method returns:
        - [0] the baseline predictiors for the ratings,
        - [1] the array of item standard deviations,
        - [2] the mean of all trainins set ratings."""
        
        # Compute the mean of the entire training set
        mu = np.mean(self.data_array[self.indices_train][:,2])
        # Define the empty vector of the item standard deviation from the mean
	print "Computing the strandand deviation of items."
        b_i = np.empty(self.N_i)
        # Fill in the vector
        for i in range(self.N_i): # For each item
            num = sum(self.train_rating[i,np.nonzero(self.train_rating[i])][0] - mu) # Compute the sum of the differences btw already given ratings and the mean
            den = 25 + len(np.nonzero(self.train_rating[i])[0]) # Compute the number of users that rated that movie
            b_i[i] = num/den # Item std 
        
        # Define the empty vector of the user standard deviation from the mean
	print "Computing the standard deviation of the users"
        b_u = np.empty(self.N_u)
        # Fill in the array
        for u in range(self.N_u): # For each user
            # Define the indices of already rated movies
            rated = np.nonzero(self.train_rating[:,u])[0]
            # Define the sum of the difference btw the already given ratings, the mean and the std of item
            num = np.sum((self.train_rating[:,u][rated]) - [mu + b_i[r] for r in rated]) 
            # Number of the already rated movies
            den = 10 + len(rated) 
            # User std
            b_u[u] = num/den 
        
        # Define the empty matix of the baseline predictor each of whose element represents the baseline prediction of movie *i* given by user *u*
        b_ui = np.empty((self.N_i,self.N_u))
        # Fill in the matrix        
        for i in range(self.N_i): # For each item
            a = b_i[i]
            # Compute the baseline prediction for each user.
            b_ui[i] = [mu + a + b_u[u] for u in range(self.N_u)] 
        
        return b_ui , b_i, mu


class recommender_system(object):
    """ This class defines the predictions for the unknown ratings."""
    
    def __init__(self, data_test, train_rating, cosine_similarity_matrix, b_ui):
        
        """An instance created by that class is characterized by the following attributes:
        - data_test : is the matrix of the test test set, return[1] of *split_dataset.train_test()*,
        - train_rating : is the matrix of ratings related to the training set, return[0] of *split_dataset.train_test*,
        - cosine_similarity_matrix : is the return of *items_similarity.cosine_similarity()* method,
        - b_ui : is the matrix of baseline predictor and it's the return of *baseline_predictor.baseline()*."""
        
        self.data_test = data_test
        self.train_rating = train_rating
        self.cosine_similarity_matrix =  cosine_similarity_matrix
        self.b_ui = b_ui
        
    def item_based_prediction(self):
        
        """This method returns the array of ratings predictions."""
        
        # Get the user in the test set
        user = sorted(set(self.data_test[:,0]))
        # Built a dictionary { key : user ID , value : [list of index with nonzero elements]}
        non_zero_user = { u : np.nonzero(self.train_rating[:,u])[0] for u in user}
        # Built a dictionary {key : user ID, values : [list of indices of zero elements]}
        zero_user = {u : np.where(self.train_rating[:,u] == 0)[0] for u in user}
        # Create a tuple that zip the user and item IDs
        user_item = zip(self.data_test[:,0], self.data_test[:,1])
        
        # Create the empty matix of predictions
        predictions =  np.empty(len(self.data_test))
        # Fill in the matrix
        k = 0
        for u,i in user_item: # For each element of the test set
            rated = non_zero_user[u] # Define the list of rated items by user *u*
            # Define the numerator : the sum of the weighted differencied btw the observed ratings and the baseline predictors
            num = np.sum(np.multiply((self.train_rating[:,u][rated]-self.b_ui[rated,u]), self.cosine_similarity_matrix[:,i][rated]))*1.0
            # Compute che sum of the weights
            den = (np.sum(abs(self.cosine_similarity_matrix[i]))-np.sum(abs(self.cosine_similarity_matrix[i,zero_user[u]])))
            # If the denominator is not zero
            if den != 0: 
                # Compute the predictions
                predictions[k] = num/den + self.b_ui[i,u] 
            # otherwise
            else: 
                # Use the baseline predictor as prediction
                predictions[k] = self.b_ui[i,u] 
            k += 1
           
            if k % 10000 == 0:
                print "Already processed ratings :" , k
        
        return predictions


class evaluate_method(object):
    """This class define the measure to evaluate the method."""
    
    def __init__(self, data_test, predictions):
        
        """An instance created by that class is characterized by the following attributes:
        - data_test : test set, return[1] of *split_dataset.train_test*,
        - predictions : is the array of prefictions for the unkown ratings."""
        
        self.data_test = data_test
        self.predictions = predictions
        
    def RMSE(self):
        
        """This method returns the prediction error."""
        
        # Define the array of observed ratings
        real = self.data_test[:,2]
        # Compute the sum pf the squared differences btw the observed and predicted ratings
        num = sum((real-self.predictions)**2)
        # Compute the RootMearSquaredError
        err = math.sqrt(num/len(self.data_test))
        
        return err
