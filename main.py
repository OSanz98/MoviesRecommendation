import pandas

# Load data
movies = pandas.read_csv('./resources/movies.csv')
credits_data = pandas.read_csv('./resources/credits.csv')
ratings = pandas.read_csv('./resources/ratings.csv')

print(movies.head())

# Calculate a weight rating
# WR = (v / (v+m)) x R + (m / (v+m)) x C
# v - number of votes for a film
# m - minimum number of votes required
# r - average rating of the film
# c - average rating across all films
m = movies['vote_count'].quantile(0.9)
c = movies["vote_average"].mean()


movies_filtered = movies.copy().loc[movies["vote_count"] >= m]


def weight_rating(df, m=m, c=c):
    r = df["vote_average"]
    v = df["vote_count"]
    wr = ((v / (v + m)) * r) + (m / (v + m) * c)
    return wr


movies_filtered["weighted_rating"] = movies_filtered.apply(weight_rating, axis=1)
print(movies_filtered)
print(movies_filtered.sort_values("weighted_rating", ascending=False)[['title', 'weighted_rating']].head(10))
# If we wanted to serve this data to a web app through API then could convert to JSON dict using .to_dict() after the
# .head() option
