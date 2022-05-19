import os

from recommendations import getEnsembleRecommendations, getMovieQuiz
from flask_cors import CORS
from flask import Flask, request, jsonify
from dotenv import load_dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # removes tf informative messages

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    movie_ratings = request.json
    return jsonify(getEnsembleRecommendations(list(movie_ratings.keys()), [int(x) for x in list(movie_ratings.values())]))


@app.route('/quiz', methods=["GET"])
def quiz():
    return jsonify(getMovieQuiz(9))


@app.route('/saved-movies', methods=['GET'])
def saved_movies():
    return "Hello World!"

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
