import json

def ReadDataset():
  with open('q-matrix.json', 'r') as fp:
    final_Q_matrix = json.load(fp)

  with open('movie_ids_list.json', 'r') as fp:
      movie_ids = json.load(fp)

  with open('movie_ids.json', 'r') as fp:
      movie_id_to_index = json.load(fp)

  with open('movies.json', 'r') as fp:
      movies = json.load(fp)

  return final_Q_matrix, movie_ids, movie_id_to_index, movies
  