# @ author : Cristina Menghini ;
# StudentID: 1527821

import pandas as pd
import random

class new_user(object):
    
    """This class returns the new user."""
    
    def __init__(self, N_i, num_items):
        
        """An instance created by that class is characterized by the following attributes:
        - N_i : is the number of items, return[4] of *split_dataset.train_test*,
        - num_items : is the number of movies that the user has alrady rated."""
        
        self.N_i = N_i
        self.num_items = num_items
        
    def define_new_user(self):
        """This method creates a csv file where the first column are the movie IDs and the second the ratings given by the user. The first
	line of the csv are the column's names."""
        
        # Define the random ratings of 135 items
        random_items=[random.randrange(1,6,1) for _ in range(self.num_items)]
        # Give an index to each of the 135 items, it will be considered the item ID
        random_index = sorted(random.sample(xrange(self.N_i), self.num_items))
        # Define the sorted list of the not rated movies indices
        not_rated = sorted(set(range(self.N_i)).difference(set(random_index)))
        # Zip the two variables and create a dataframe
        zipped_df = pd.DataFrame(zip(random_index,random_items))
        # Create a csv file for the new user
        zipped_df.to_csv('new_user.csv', sep = ',', index = False)
