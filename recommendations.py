import random
import numpy as np
import py2neo
import tensorflow.compat.v1 as tf

from load import ReadDataset

tf.disable_v2_behavior()

K = 10  # number of latent factors
# number of movies I have run the model with (prototyping and improving still)
num_movies = 5000

# first x movie_ids (same order as q-matrix)
final_Q_matrix, movie_ids, movie_id_to_index, movies = ReadDataset()

movie_ids = movie_ids[:num_movies]

def getLatentRecommendations(user_movie_ids, user_ratings):
    user_movie_indices = [(lambda movie_id: movie_id_to_index[movie_id])(movie_id) for movie_id in user_movie_ids]
    # print("x")
    # print(user_movie_indices)

    new_user_P_row_initial = np.random.rand(1, K)
    new_user_P_row = tf.Variable(new_user_P_row_initial, dtype=tf.float32)
    new_user_P_row_times_Q = tf.matmul(new_user_P_row, final_Q_matrix)
    # print("1")
    res = tf.gather(new_user_P_row_times_Q, user_movie_indices, axis=1)
    squared_error = tf.square(user_ratings - res)
    # print("2")
    loss = tf.reduce_sum(squared_error)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    predict = optimizer.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(50000):
        sess.run(predict)
        print("{0} / {1}".format(i+1, 50000), end = "\r")

    predicted_movie_ratings = np.around(sess.run(new_user_P_row_times_Q), 3)[0].tolist()

    top_10_movie_ids = [x for y, x in sorted(zip(predicted_movie_ratings, movie_ids))][:10]
    # print("y")
    # print(top_10_movie_ids)

    return [(lambda movie_id: {"movieId": movie_id, **movies[str(movie_id)]})(movie_id) for movie_id in top_10_movie_ids]

def getEnsembleRecommendations(user_movie_ids, user_ratings):
    return getLatentRecommendations(user_movie_ids, user_ratings)

def getMovieQuiz(num_rand_movies):
    selectedMovies = []

    for i in range(0, num_rand_movies):
        selected_movie_index = random.randint(0, num_movies)
        # print("s", selected_movie_index)

        for movie_id in movie_id_to_index:
            if movie_id_to_index[movie_id] == selected_movie_index:
                movie_id = str(int(movie_id))
                # print("m", movie_id)
                selectedMovies.append(
                    {**{"movieId": movie_id}, **movies[movie_id]})
                break

    return selectedMovies
