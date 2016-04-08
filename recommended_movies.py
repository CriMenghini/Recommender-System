# @ author : Cristina Menghini ;
# StudentID: 1527821


import numpy as np
from collections import OrderedDict
from operator import itemgetter

class make_recommendation(object): 
    """This class returns a list of ten recommendation for a new user."""
    
    def __init__(self, not_rated, pred , movie_data):
        
        """An instance created by that class is characterized by the following attributes:
        - not_rated : is the list of indices of not yet rated items,
        - pred : is the array of predictions, return of the *new_user_prediction* function,
        - movie_data : is the return of *load_data.adjust_data()* method, it contains informations about movies."""
        
        self.not_rated = not_rated
        self.pred = pred
        self.movie_data = movie_data
    
    def recommendations(self):
        """This method returns the list of recommendations."""
        
        # Define the dictionary : {key = indices of not rated movies : value = prediction of key item}
        dic_prediction = {i : j for i,j in zip(self.not_rated,self.pred)}
        # Sort the dictionary 
        sort_prediction = OrderedDict(sorted(dic_prediction.items(), key=itemgetter(1), reverse = True))
        
        # Get the list of the top 10 recommended movies
        recomm = [(self.movie_data[0][i,1], self.movie_data[0][i,2])for i in sort_prediction.keys()[:10]]
        
        return recomm  
