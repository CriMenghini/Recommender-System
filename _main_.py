# @ author : Cristina Menghini ;
# StudentID: 1527821

from create_new_user import new_user
from from_file import load_data
from training_test import split_dataset
import recommender_system
import pandas as pd
import numpy as np
import new_user_pred
from recommended_movies import make_recommendation
import pickle
#from sys import argv


#script , rating_file, perc_test, rated_items = argv


def offline(rating_file, perc_test):
    """ This function represent the offline procedure. The parameters of the function are:
    - rating_file : that is the name of the data file,
    - perc_test : that is the percentage you want to built the test set on.
    It returns:
    - [0] : the rmse of the model,
    - [1] : the baseline predictions,
    - [2] : number of items,
    - [3] : similarity matrix."""
    
    print "1527821 is loading data.."
    # Define a new instance of the class load_data and appy the method adjust_data
    given_ratings = load_data(rating_file, ',').adjust_data()
    # Obtain the data as an array
    data_array = given_ratings[0]
    # And as a DataFrame
    data = given_ratings[1] 
    
    # Split the dataset 
    split_informations = split_dataset(data, data_array, int(perc_test)).train_test()
    print "The training and the test sets have been created."
    
    # Define the ratings matrix of the training set
    training_rating = split_informations[0]
    
    print "1527821 is building the similarity matrix."
    # Compute the cosine similarity matrix
    similarity_matrix = recommender_system.items_similarity(training_rating).cosine_similarity()
    
    # Define the list of items in the training set
    indices_train = split_informations[3]
    # Define the number of items
    N_i = split_informations[4]
    # defne the number of users
    N_u = split_informations[5]
    
    print "The programm is computing the baseline predictions."
    # Compute the baseline predictions for the unknown ratings
    baseline_pred = recommender_system.baseline_predictor( training_rating, data_array, indices_train, N_i, N_u).baseline()
    
    # Define the test set
    data_test = split_informations[1]
    # Define the baseline predictions
    b_ui = baseline_pred[0]
    
    print "Initializing the rating prediction procedure.."
    # Compute the prediction for movie in the test set
    predict = recommender_system.recommender_system(data_test, training_rating, similarity_matrix, b_ui).item_based_prediction()
    
    # Measure the goodness of our method
    rmse = recommender_system.evaluate_method(data_test, predict).RMSE()
    print "The RMSE is : ", rmse
    
    return rmse, baseline_pred, N_i, similarity_matrix


def online(n_rated_items, new_user_file, base_pred, len_items, sim_matrix):
    
    print "Loading the new user data..."
    # Read the csv file with user's informations
    user_data = np.array(pd.read_csv(new_user_file, sep = ','))
    
    # Define the indices of rated movies
    random_index = user_data[:,0]
    # Define the ratings of rated items
    rated = user_data[:,1]
    # Define the ratings of not yet rated items
    not_rated = sorted(set(range(len_items)).difference(set(random_index)))
    
    # Create an empty array that represents the user
    user_array = np.full((len_items,1),0)
    # Fill in the array each of whose element is the rating given( or not yet given) by the user
    for k in range(len(random_index)): 
        user_array[random_index[k]] = rated[k] 
        
    # Compute the baseline predictions of the new user
    b_ui = new_user_pred.new_user_baseline_predictor(random_index, user_array, base_pred,len_items, n_rated_items).baseline_new_user()
    
    # Compute the predictions for the new user
    pred = new_user_pred.new_user_prediction(not_rated, user_array, random_index, b_ui, sim_matrix, n_rated_items)
    
    return pred, not_rated


request = raw_input('Press 0 to run the recommandation system and get an evaluation, press any other number to get a movie recommendation: ')


if int(request) == 0:
    rating_file = raw_input('Insert the name of the csv file that contains the data. ')
    perc_test = raw_input('Which is the percentage of the data that had to be considered as the test set? ')
    # Offline procedure
    print "Offline procedure starts."
    off = offline('ratings.csv', 20)
    
    # Save the similarity matrix
    print "Saving the similarity matrix..."
    matrix = pd.DataFrame(off[3])
    matrix.to_csv('similarity_matrix.csv')
    # Save baseline pred
    print "Saving baseline predictions"
    pickle.dump( off[1][1], open('base_user_pred.p', 'wb'))

else:
    rated_items = raw_input('Insert the number of items that the new user already watched. ')
    # Generate the csv file that contains the new user ratings
    new_user = new_user(3706, int(rated_items)).define_new_user()

    # Online procedure
    print "Online procedure starts."
    # Import the similarity matrix
    similarity_mat = np.array(pd.read_csv('similarity_matrix.csv'))
    base = pickle.load(open('base_user_pred.p', 'rb'))
    on = online(rated_items,'new_user.csv', base, 3706, similarity_mat )

    # Load the csv file that contains movies informations
    movies = load_data('movies_2.csv', '::').adjust_data()

    # Make the recommendation
    recoms = make_recommendation(on[1], on[0], movies).recommendations()
    print recoms
