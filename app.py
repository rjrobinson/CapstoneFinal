import os

import requests
from flask import Flask, render_template, request, Response
from sklearn.cluster import KMeans

from models.netflix_data import NetflixData  # Import other necessary modules and functions from your Notebook
import seaborn as sns
import io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Add this line
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Example usage:
netflix_data = NetflixData()


def get_poster_url(title):
    omdb_api_key = '827c464d'
    omdb_url = f'http://www.omdbapi.com/?t={title}&apikey={omdb_api_key}'
    response = requests.get(omdb_url)
    movie_data = response.json()

    if 'Poster' in movie_data:
        print(f"Successfully retrieved poster for '{title}'.")
        return movie_data['Poster']
    else:
        print(f"Failed to retrieve poster for '{title}'.")
        return None



# # Call other methods as needed
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/recommend', methods=['POST'])
def recommend_by():
    query = request.form['query'].upper()
    recommendations = netflix_data.recommend_by(query)

    for movie in recommendations:
        movie['poster_url'] = get_poster_url(movie['title'])

    return render_template('index.html', recommendations=recommendations)


@app.route('/top_10_genres')
def top_10_genres():
    return render_template('top_10_genres.html')


@app.route('/top_10_genres_plot.png')
def top_10_genres_plot():
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 6))

    top_10_genres = netflix_data.data_netflix['genre_type'].value_counts().head(10)
    colors = sns.color_palette("colorblind", n_colors=len(top_10_genres))

    bars = sns.barplot(x=top_10_genres.index, y=top_10_genres.values, palette=colors, ax=ax)

    ax.set_ylabel('Number of Titles')
    ax.tick_params(axis='x', rotation=0, labelsize=8)  # Reset rotation to 0
    ax.set_xticklabels([])  # Remove the x-axis labels

    # Create a legend with color mappings
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(top_10_genres))]
    ax.legend(handles, top_10_genres.index, title="Genres", loc="upper left",
              bbox_to_anchor=(1.05, 1))  # Add the legend

    plt.tight_layout()  # Automatically adjust the subplot parameters

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.clf()

    return Response(img_buffer.getvalue(), content_type='image/png')


@app.route('/release_year_histogram')
def release_year_histogram():
    return render_template('release_year_histogram.html')


@app.route('/release_year_histogram_plot.png')
def release_year_histogram_plot():
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data=netflix_data.data_netflix, x='release_year', bins=20, kde=False, color='purple', ax=ax)

    ax.set_xlabel('Release Year')
    ax.set_ylabel('Number of Titles')
    ax.set_xlim(1970, None)  # Start the x-axis at 1970

    plt.tight_layout()  # Automatically adjust the subplot parameters

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.clf()

    return Response(img_buffer.getvalue(), content_type='image/png')


from sklearn.decomposition import PCA


@app.route('/kmeans_clusters')
def kmeans_clusters():
    return render_template('kmeans_clusters.html')


@app.route('/kmeans_clusters_plot.png')
def kmeans_clusters_plot():
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Applying PCA for dimensionality reduction if needed
    pca = PCA(n_components=2)
    X_genre_type = pca.fit_transform(
        netflix_data.group_dummies)  # Make sure you've set group_dummies as a class attribute

    kmeans_model = KMeans(n_clusters=34, random_state=0)
    y_Kmeans34 = kmeans_model.fit_predict(X_genre_type)

    scatter = ax.scatter(X_genre_type[:, 0], X_genre_type[:, 1], c=y_Kmeans34, cmap='viridis', s=50)

    # Plotting the cluster centers
    centers = kmeans_model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

    ax.set_title('KMeans Clustering of Genres')
    ax.set_xlabel('Genre Direction 1')
    ax.set_ylabel('Genre Direction 2')

    # Create a legend with color mappings
    legend_labels = [f"Cluster {i}" for i in range(34)]
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1))
    ax.add_artist(legend1)

    plt.tight_layout()  # Automatically adjust the subplot parameters

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.clf()

    return Response(img_buffer.getvalue(), content_type='image/png')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
