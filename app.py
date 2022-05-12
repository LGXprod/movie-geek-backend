import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # removes tf informative messages

import json
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

K = 10 # number of latent factors
num_movies = 5000 # number of movies I have run the model with (prototyping and improving still)
final_Q_matrix = None
movie_ids = None # first x movieIds (same order as q-matrix)
movie_id_to_index = None
movies = None

with open('q-matrix.json', 'r') as fp:
    final_Q_matrix = json.load(fp)

with open('movie_ids_list.json', 'r') as fp:
    movie_ids = json.load(fp)

with open('movie_ids.json', 'r') as fp:
    movie_id_to_index = json.load(fp)

with open('movies.json', 'r') as fp:
    movies = json.load(fp)

movie_ids = movie_ids[:num_movies]

def getRecommendations(user_movie_ids, user_ratings):
    user_movie_indices = [(lambda movie_id: movie_id_to_index[str(float(movie_id))]+1)(movie_id) for movie_id in user_movie_ids]
    print(user_movie_indices)

    new_user_P_row_initial = np.random.rand(1, K)
    new_user_P_row = tf.Variable(new_user_P_row_initial, dtype=tf.float32)
    new_user_P_row_times_Q = tf.matmul(new_user_P_row, final_Q_matrix)
    res = tf.gather(new_user_P_row_times_Q, user_movie_indices, axis=1)
    squared_error = tf.square(user_ratings - res)
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
    print(top_10_movie_ids)

    return [(lambda movie_id: {"movieId": movie_id, **movies[str(movie_id)]})(movie_id) for movie_id in top_10_movie_ids]

def getRandomMovies(num_rand_movies):
    selectedMovies = []

    for i in range(0, num_rand_movies):
        selected_movie_index = random.randint(0, num_movies)
        print("s", selected_movie_index)

        for movie_id in movie_id_to_index:
            if movie_id_to_index[movie_id] == selected_movie_index:
                movie_id = str(int(float(movie_id)))
                print("m", movie_id)
                selectedMovies.append({**{"movieId": movie_id}, **movies[movie_id]})
                break

    return selectedMovies

app = Flask(__name__)
CORS(app)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    bodyData = request.form
    movie_ids = json.loads(bodyData["movie-ids"])
    movie_ratings = json.loads(bodyData["movie-ratings"])

    return jsonify(getRecommendations(movie_ids, movie_ratings))

@app.route('/quiz', methods=["GET"])
def quiz():
    movies = getRandomMovies(9)

    for movie in movies:
        movie["genres"] = movie["genres"].replace("|", ", ")

    return jsonify(movies)

@app.route('/saved-movies', methods=['GET'])
def saved_movies():
    return "Hello World!"

# Run the app
if __name__ == "__main__":
   app.run(debug=True)