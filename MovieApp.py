import pandas
from sklearn.feature_extraction.text import TfidfVectorizer  # This package allows us to vectorise the text
# The text has to be converted to coefficients and numbers to compare the similarity of words.
# It will highlight the importance of each word in the dataframe.

from sklearn.metrics.pairwise import linear_kernel

movies = pandas.read_csv('./resources/movies_small.csv', sep=";")

tfidf = TfidfVectorizer(stop_words='english')

movies["overview"] = movies["overview"].fillna("")

# This line below creates a matrix mapping words in each film overview section, and their frequencies
tfidf_matrix = tfidf.fit_transform(movies["overview"])
# the following code is used create similarity coefficients between pairing of each film.
# i.e., each film is matched up against each other and this matrix is used to calculate and represent that.
similarity_metrics = linear_kernel(tfidf_matrix, tfidf_matrix)


# sorted(similarity_metrics[5], reverse=True)
# This function calculates similarity using the similarity_metrics above and then returns a list of similar movie titles
def similar_movies(movie_title, nr_movies):
    idx = movies.loc[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity_metrics[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movies_indices = [tpl[0] for tpl in scores[1:nr_movies + 1]]
    similar_titles = list(movies["title"].iloc(movies_indices))
    return similar_titles
