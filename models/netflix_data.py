import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Response

class NetflixData:
    def __init__(self, filename='imdb/netflix_titles.csv'):
        # Load the dataset
        self.data_netflix = pd.read_csv(filename)
        # Preprocess the data by renaming columns, filling missing values, etc.
        self.preprocess_data()
        # Cluster the data based on genres
        self.cluster_data()

    def preprocess_data(self):
        """Preprocess the data: rename columns, fill missing values, and convert genres to uppercase"""
        # Renaming columns
        self.data_netflix = self.data_netflix.rename(
            columns={'date_added': 'date_added_platform', 'duration': 'duration_seconds', 'listed_in': 'genre_type',
                     'type': 'movie_or_tv_show'})
        # Dropping unnecessary columns
        self.data_netflix.drop(columns=['show_id'], axis=1, inplace=True)
        # Filling missing values
        self.data_netflix['cast'] = self.data_netflix['cast'].fillna('uninformed cast')
        self.data_netflix['director'] = self.data_netflix['director'].fillna('uninformed director')
        self.data_netflix['country'] = self.data_netflix['country'].fillna('uninformed country')
        # self.data_netflix = self.data_netflix[self.data_netflix['country'] == 'United States']
        # Converting genres to uppercase
        self.data_netflix['genre_type'] = self.data_netflix['genre_type'].apply(lambda x: x.upper())

    def cluster_data(self):
        """Cluster the data based on genres using KMeans clustering"""
        # Splitting the genre strings into separate columns
        df_split = self.data_netflix['genre_type'].str.split(',', expand=True)
        df_split = df_split.fillna('-')
        # Creating dummies for each unique genre
        group_dummies = [pd.get_dummies(df_split[y].apply(lambda x: x.strip()), dtype='int') for y in df_split.columns]
        group_dummies = pd.concat(group_dummies, axis=1)
        group_dummies = group_dummies.fillna(0).astype('uint8')
        # Converting titles to uppercase
        self.data_netflix['title'] = self.data_netflix['title'].apply(lambda x: x.upper())
        # Applying KMeans clustering to the genre data
        X_genre_type = np.array(group_dummies)
        kmeans_model = KMeans(n_clusters=34, random_state=0)
        y_Kmeans34 = kmeans_model.fit_predict(X_genre_type)
        # Storing the cluster labels in the dataframe
        self.data_netflix['clusters_genre'] = y_Kmeans34

    def recommend_by(self, query, top_n=50):
        recommendations_by_actor = self.recommend_by_actor(query, top_n)
        recommendations_by_title = self.recommend_by_title(query, top_n)
        recommendations_by_director = self.recommend_by_director(query, top_n)
        recommendations_by_genre = self.recommend_by_genre(query, top_n)

        # Combine the recommendations
        combined_recommendations = recommendations_by_actor + recommendations_by_title + recommendations_by_director + recommendations_by_genre

        # If you want to remove duplicates, you may need to implement a function for that
        unique_recommendations = self._remove_duplicates(combined_recommendations)

        # Return the top N combined recommendations
        return unique_recommendations[:top_n]

    def _remove_duplicates(self, recommendations):
        seen_titles = set()
        unique_recommendations = []
        for recommendation in recommendations:
            title = recommendation['title']
            if title not in seen_titles:
                unique_recommendations.append(recommendation)
                seen_titles.add(title)
        return unique_recommendations

    def _recommend(self, top_recommendations):
        recommended_movies = []
        for _, row in top_recommendations.iterrows():
            movie_info = {
                'movie_or_tv_show': row['movie_or_tv_show'],
                'title': row['title'],
                'director': row['director'],
                'cast': row['cast'],
                'country': row['country'],
                'date_added_platform': row['date_added_platform'],
                'release_year': row['release_year'],
                'duration_seconds': row['duration_seconds'],
                'genre_type': row['genre_type'],
                'description': row['description'],
                'rating': row['rating']
            }
            recommended_movies.append(movie_info)
        return recommended_movies

    def recommend_by_actor(self, actor_name, top_n=5):
        recommendations = self.data_netflix[self.data_netflix['cast'].str.contains(actor_name, na=False)]
        return self._recommend(recommendations.head(top_n))

    def recommend_by_title(self, movie_title, top_n=5):
        filtered_movies = self.data_netflix[self.data_netflix['title'] == movie_title]
        if filtered_movies.empty:
            print(f"No movie found with title {movie_title}")
            return []
        cluster = filtered_movies['clusters_genre'].iloc[0]
        recommendations = self.data_netflix[self.data_netflix['clusters_genre'] == cluster]
        top_recommendations = recommendations.head(top_n)
        return self._recommend(top_recommendations)

    def recommend_by_genre(self, genre, top_n=5):
        recommendations = self.data_netflix[self.data_netflix['genre_type'].str.contains(genre)]
        return self._recommend(recommendations.head(top_n))

    def recommend_by_director(self, director_name, top_n=5):
        recommendations = self.data_netflix[self.data_netflix['director'].str.contains(director_name, na=False)]
        return self._recommend(recommendations.head(top_n))

    def plot_top_10_genres(self):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(10, 6))
        top_10_genres = self.data_netflix['genre_type'].value_counts().head(10)
        sns.barplot(x=top_10_genres.index, y=top_10_genres.values, palette="colorblind")
        plt.title('Top 10 Genres on Netflix')
        plt.xlabel('Genre')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Return the binary image data
        return Response(img_buffer.read(), content_type='image/png')

    def plot_distribution_of_ratings(self):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data_netflix['rating'], bins=20, kde=True, color='purple')
        plt.title('Distribution of Ratings')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Titles')
        plt.tight_layout()
        plt.show()

    def plot_number_of_titles_by_year(self):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12, 6))
        last_20_years = self.data_netflix[self.data_netflix['release_year'] >= (pd.to_datetime('today').year - 20)]
        last_20_years['release_year'].value_counts().sort_index().plot.bar()
        plt.title('Number of Titles Released by Year (Last 20 Years)')
        plt.xlabel('Release Year')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
# # Example usage:
# netflix_data = NetflixData()
# recommended_movies = netflix_data.recommend_movie('SOME MOVIE TITLE')
