# @ author : Cristina Menghini ;
# StudentID: 1527821

import numpy as np

class new_user_baseline_predictor(object):
    
    def __init__(self, random_index, user_array, b_i, N_i, num_items):
        
        """An instance created by that class is characterized by the following attributes:
        - random_index : is the list of rated movies indices, it's the return[1] of the *new_user.define_new_user()* method ,
        - user_array : is the array that represent the ratings of movies given by the user even the already not rated, return[0] of the *new_user.define_new_user()* method,
        - b_i : is the array of item's stds from the mean, return[1] of *baseline_predictor.baseline()* method,
        - N_i : is the number of items, return[4] of *split_dataset.train_test*,
        - num_items : is the number of movies that the user has alrady rated."""
    
        self.random_index = random_index
        self.user_array = user_array
        self.b_i = b_i
        self.N_i = N_i
        self.num_items = num_items
    
    def baseline_new_user(self):
        """This method returns the baseline predictor for a new user."""
        
        mu = np.mean(self.user_array[self.random_index])
        # Compute the user standard deviation from the mean
        numerator = np.sum(self.user_array[self.random_index] - (mu + self.b_i[self.random_index]).reshape(int(self.num_items),1))
        # Divide by the number of rated items
        denominator = 10 + len(self.random_index)
        # Compute the standard deviation array
        b_u = numerator/denominator
        
        # Define the empty vector of baseline predictor for the user
        b_ui = np.empty((self.N_i,1))
        # Fill in the vector       
        for i in range(self.N_i): # For each item
            # Compute the baseline prediction of the user
            b_ui[i] = mu + self.b_i[i] + b_u
            
        return b_ui


def new_user_prediction(not_rated, user_array, random_index, b_ui, cosine_similarity_matrix, numb_items):
    
    """This method returns the array of new user predictions."""
        
    # Define the empty array of predictions
    prediction = np.empty(len(not_rated))
    k = 0
    # Compute the prediction for each nt yet rated movie
    for i in not_rated:
        # Define the numerator as the sum of the weighted differences btw the observed ratings and the baseline predictions
        num = np.sum(np.multiply((user_array[random_index] - b_ui[random_index]),cosine_similarity_matrix[i,random_index].reshape(int(numb_items),1)))*1.0
        # Divide by the sum of weights
        den = np.sum(cosine_similarity_matrix[i])-np.sum(cosine_similarity_matrix[i,not_rated])
        # Whether the denominator is different from 0
        if den != 0 :
            # Make the prediction summing up the fractio num/den and the base line prediction
            prediction[k] = num/den + b_ui[i]
        # Otherwise
        else:
            # Use the base line as a prediction
            prediction[k] = b_ui[i]
        k += 1
        if k % 1000 == 0:
            print "Already processed ratings : ", k
            
    return prediction
