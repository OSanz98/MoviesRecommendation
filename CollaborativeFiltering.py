import pandas
from surprise import Dataset, Reader, SVD, model_selection

# movies = pandas.read_csv('./resources/movies.csv')
ratings = pandas.read_csv('./resources/ratings.csv')[["userId", "movieId", "rating"]]

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(ratings, reader)
training_set = dataset.build_full_trainset()
svd = SVD()
svd.fit(training_set)
svd.predict(15, 1956)
model_selection.cross_validate(svd, dataset, measures=["RMSE","MAE"])
# RMSE stands for Root Mean Square Error and MAE stands for Mean Absolute Error