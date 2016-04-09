# Recommender-System(12-17 february 2016)

In this project I have to implement a recommender system for movies. As a dataset I use the MovieLens 1M dataset, available at http://grouplens.org/datasets/movielens. The scripts refer to the 'ratings.csv'.

### Offline procedere

I used the item-based method with some corrections avaiable in the code.
I first evaluate the recommender system offline splitting the dataset (i.e. pairs of user × movie) into a training set (80% of pairs) and test set (20% of pairs). After training the system on the training set, I evaluate it on the test set. My attention was focused on the RMSE that for this system is, on average, 0.89.


### Online procedure

Once I complete the offline procedure, I consider a new user. Assume that you are given a list of movies and scores, corresponding to the preferences of this new users. These will be provided by a file. The user is also interested in some new movies. Recommend to him or her the movies he or she may like.

### How to run the code:

Once you have all the scripts in a folder, type on the bash: python _ main _.py( there are no space btw the _ and main). Then follow the instructions for the Offline or Online procedure. In order to run the code it’s importanto to do first the Offline procedure (typying 0 when requested) and then the Online procedure. That’s because the files that are created during the Offline procedure are useful in the Online part.

### Scripts summary

* create_ new _user.py is the library by which the new user is randomly created.
* from_file.py is a library by which the csv files are loading.
* new_user _pred.py is a library by which are computed the predictions for the new user.
* recommended_movies.py is a library that provides to the user a list of recommended movies. 
* recommender_system.py is the library that contains the code of the method I used.
* traning_test.py is the library by which the dataset is splitted.

