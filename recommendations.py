import random

import numpy as np
from py2neo import Graph
import tensorflow.compat.v1 as tf

from load import ReadDataset

tf.disable_v2_behavior()

graph = Graph("bolt://localhost:7687", auth=("neo4j", "test"))

K = 10  # number of latent factors
# number of movies I have run the model with (prototyping and improving still)
num_movies = 5000

# first x movie_ids (same order as q-matrix)
final_Q_matrix, movie_ids, movie_id_to_index, movies = ReadDataset()

movie_ids = movie_ids[:num_movies]

allGenres = {'Sci-Fi', 'Documentary', 'Fantasy', 'Musical', 'Animation', 'Children', 'Romance', 'Thriller', 'War', 'IMAX',
             'Horror', 'Drama', 'Adventure', 'Film-Noir', 'Comedy', 'Mystery', 'Western', 'Action', 'Crime'}

def getRecSchema(movie_id, graphFreq, recType):
    return {
        "movieId": movie_id,
        "graphFreq": graphFreq,
        "recType": recType,
        **movies[movie_id]
    }

def getGraphRecommendations(user_movie_ids, user_ratings):
    movie_id_count = {}
    high_rated_movie_ids = []

    for i, movie_id in enumerate(user_movie_ids):
        if user_ratings[i] >= 4:
            high_rated_movie_ids.append(movie_id)

    def getRecByCloseNodes(nodeType):
        try:
            for i, movie_id in enumerate(high_rated_movie_ids):
                for name in movies[movie_id][f"{nodeType.lower()}s"]:
                    name = name.lower() if nodeType == "Tag" else name

                    cypherQuery = f"match (t:{nodeType}) where t.name = '{name}' call {{ with t match (t)-[:{nodeType}_OF]-(m:Movie) return m limit 5 }} call {{ with m match (m)-[:{nodeType}_OF]-(t2:{nodeType}) return t2 limit 2 }} call {{ with t2 match (t2)-[:{nodeType}_OF]-(m2:Movie) return m2 limit 2 }} return m, m2;"
                    movie_results = (graph.run(cypherQuery)).data()
                    # print(nodeType, len(movie_results))

                    for movie in movie_results:
                        movie_1_id = movie["m"]["id"]
                        movie_2_id = movie["m2"]["id"]

                        if movie_1_id in movie_id_count:
                            movie_id_count[movie_1_id]["freq"] += 1
                            movie_id_count[movie_1_id]["recType"].add(
                                f"similar-{nodeType.lower()}")
                        else:
                            movie_id_count[movie_1_id] = {
                                "freq": 1,
                                "recType": {f"similar-{nodeType.lower()}"}
                            }

                        if movie_2_id in movie_id_count:
                            movie_id_count[movie_2_id]["freq"] += 1
                            movie_id_count[movie_2_id]["recType"].add(
                                f"similar-{nodeType.lower()}")
                        else:
                            movie_id_count[movie_2_id] = {
                                "freq": 1,
                                "recType": {f"similar-{nodeType.lower()}"}
                            }
        except:
            pass

    getRecByCloseNodes("Tag")
    getRecByCloseNodes("Genre")

    movie_id_count = [[movieId, meta["freq"] if len(
        meta["recType"]) == 1 else meta["freq"]+10, list(meta["recType"])] for movieId, meta in movie_id_count.items()]
    recommended_movies = sorted(
        movie_id_count, key=lambda x: x[1], reverse=True)[:10]
    # print("r", recommended_movies)

    return recommended_movies


def getLatentRecommendations(user_movie_ids, user_ratings):
    user_movie_indices = [(lambda movie_id: movie_id_to_index[movie_id])(
        movie_id) for movie_id in user_movie_ids]
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
        print("{0} / {1}".format(i+1, 50000), end="\r")

    predicted_movie_ratings = np.around(
        sess.run(new_user_P_row_times_Q), 3)[0].tolist()

    top_10_movie_ids = [x for y, x in sorted(
        zip(predicted_movie_ratings, movie_ids))][:10]
    # print("y")
    # print(top_10_movie_ids)

    # return [(lambda movie_id: {"movieId": movie_id, **movies[str(movie_id)]})(movie_id) for movie_id in top_10_movie_ids]
    return top_10_movie_ids


def getEnsembleRecommendations(user_movie_ids, user_ratings):
    movie_ids_latent_from_model = getLatentRecommendations(user_movie_ids, user_ratings)
    latentRecommendations = [[str(movieId), 0, ["latent-factor-model"]] for movieId in movie_ids_latent_from_model]
    graphRecommendations = getGraphRecommendations(user_movie_ids, user_ratings)

    # remove duplications
    for i in range(len(graphRecommendations)):
        if i < len(movie_ids_latent_from_model):
            if graphRecommendations[i][0] == movie_ids_latent_from_model[i]:
                graphRecommendations[i][2].insert(0, "latent-factor-model")
                del latentRecommendations[i]
        else:
            break

    recommendations = [getRecSchema(movieId, graphFreq, recType) for movieId, graphFreq, recType in latentRecommendations + graphRecommendations]

    return recommendations

# refactor this function later


def getMovieQuiz(min_num_movies):
    selectedMovies = []

    unmarked_genres = allGenres

    while len(unmarked_genres) > 0 or len(selectedMovies) < min_num_movies:
        selected_movie_index = random.randint(0, num_movies)
        # print("s", selected_movie_index)

        for movie_id in movie_id_to_index:
            if movie_id_to_index[movie_id] == selected_movie_index:
                movie_id = str(int(movie_id))
                movie = movies[movie_id]

                if len(unmarked_genres) == 0 and len(selectedMovies) < min_num_movies:
                    selectedMovies.append(
                        {**{"movieId": movie_id}, **movie})
                    break
                else:
                    # check if this movie adds a new genre
                    for genre in movie["genres"]:
                        if genre in unmarked_genres:
                            unmarked_genres.remove(genre)
                            selectedMovies.append(
                                {**{"movieId": movie_id}, **movie})
                            break

                    break

    return selectedMovies
