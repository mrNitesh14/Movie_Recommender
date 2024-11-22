import sqlite3
import json
from fastapi import FastAPI, HTTPException
from typing import List
import bcrypt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize FastAPI app
app = FastAPI()

import sqlite3
import csv

# Database setup
DATABASE_FILE = "movie_recommendation.sqlite"
MOVIES_CSV_FILE = "imdb_top_1000.csv"  # Ensure you have a CSV file named 'movies.csv'

# def initialize_db():
#     """
#     Initialize the SQLite database with tables for users and movies.
#     Populate the movies table with data from a CSV file if it's empty.
#     """
#     with sqlite3.connect(DATABASE_FILE) as conn:
#         cursor = conn.cursor()
#         # Create users table
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 username TEXT NOT NULL UNIQUE,
#                 password_hash TEXT NOT NULL,
#                 favorite_genre TEXT DEFAULT '[]'
#             )
#         """)
#         print("Users table initialized.")
        
#         # Create movies table
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS movies (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 title TEXT NOT NULL,
#                 genre TEXT NOT NULL,
#                 imdb_rating REAL,
#                 overview TEXT,
#                 director TEXT,
#                 released_year TEXT,
#                 runtime TEXT,
#                 meta_score REAL,
#                 votes INTEGER
#             )
#         """)
#         print("Movies table initialized.")
        
#         # Check if movies table is empty
#         cursor.execute("SELECT COUNT(*) FROM movies")
#         movie_count = cursor.fetchone()[0]
#         # Inside initialize_db
#         if movie_count == 0:
#             # Populate the movies table from the CSV file
#             try:
#                 with open(MOVIES_CSV_FILE, "r", encoding="utf-8") as csv_file:
#                     csv_reader = csv.DictReader(csv_file)
#                     movies = [
#                         (
#                             row["Series_Title"],  # Corrected column name
#                             row["Genre"],  # Corrected column name
#                             float(row["IMDB_Rating"]) if row["IMDB_Rating"] else None,
#                             row["Overview"],
#                             row["Director"],
#                             row["Released_Year"],
#                             row["Runtime"],
#                             float(row["Meta_score"]) if row["Meta_score"] else None,
#                             int(row["No_of_Votes"]) if row["No_of_Votes"] else None
#                         )
#                         for row in csv_reader
#                     ]
#                     cursor.executemany("""
#                         INSERT INTO movies (title, genre, imdb_rating, overview, director, released_year, runtime, meta_score, votes)
#                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#                     """, movies)
#                     print(f"{len(movies)} movies imported successfully from {MOVIES_CSV_FILE}.")
#             except FileNotFoundError:
#                 print(f"Error: CSV file '{MOVIES_CSV_FILE}' not found.")
#             except Exception as e:
#                 print(f"Error while importing movies: {e}")

        
#         conn.commit()

def initialize_db():
    """
    Initialize the SQLite database with tables for users, movies, and user_movie_likes.
    Populate the movies table with data from a CSV file if it's empty.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                favorite_genre TEXT DEFAULT '[]'
            )
        """)
        print("Users table initialized.")
        
        # Create movies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                genre TEXT NOT NULL,
                imdb_rating REAL,
                overview TEXT,
                director TEXT,
                released_year TEXT,
                runtime TEXT,
                meta_score REAL,
                votes INTEGER
            )
        """)
        print("Movies table initialized.")
        
        # Create user_movie_likes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_movie_likes (
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                liked BOOLEAN NOT NULL,
                PRIMARY KEY (user_id, movie_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE
            )
        """)
        print("User-Movie Likes table initialized.")
        
        # Check if movies table is empty
        cursor.execute("SELECT COUNT(*) FROM movies")
        movie_count = cursor.fetchone()[0]
        
        if movie_count == 0:
            # Populate the movies table from the CSV file
            try:
                with open(MOVIES_CSV_FILE, "r", encoding="utf-8") as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    movies = [
                        (
                            row["Series_Title"],  # Corrected column name
                            row["Genre"],  # Corrected column name
                            float(row["IMDB_Rating"]) if row["IMDB_Rating"] else None,
                            row["Overview"],
                            row["Director"],
                            row["Released_Year"],
                            row["Runtime"],
                            float(row["Meta_score"]) if row["Meta_score"] else None,
                            int(row["No_of_Votes"]) if row["No_of_Votes"] else None
                        )
                        for row in csv_reader
                    ]
                    cursor.executemany("""
                        INSERT INTO movies (title, genre, imdb_rating, overview, director, released_year, runtime, meta_score, votes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, movies)
                    print(f"{len(movies)} movies imported successfully from {MOVIES_CSV_FILE}.")
            except FileNotFoundError:
                print(f"Error: CSV file '{MOVIES_CSV_FILE}' not found.")
            except Exception as e:
                print(f"Error while importing movies: {e}")
        
        conn.commit()

# Initialize the database
initialize_db()

# Helper function to execute queries
def execute_query(query, params=(), fetch_one=False, fetch_all=False):
    """
    Execute a query on the SQLite database.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        if fetch_one:
            return cursor.fetchone()
        if fetch_all:
            return cursor.fetchall()
        conn.commit()
        
imdb_data = pd.read_csv("imdb_top_1000.csv")

# Populate the movies table from the dataset
def populate_movies_from_dataset(dataframe):
    """
    Populate the movies table with data from the provided DataFrame.
    """
    movies = dataframe.to_dict(orient="records")
    query = """
        INSERT INTO movies (title, genre, imdb_rating, overview, director, released_year, runtime, meta_score, votes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        for movie in movies:
            cursor.execute(query, (
                movie["Series_Title"],
                movie["Genre"],
                movie["IMDB_Rating"],
                movie["Overview"],
                movie["Director"],
                movie["Released_Year"],
                movie["Runtime"],
                movie["Meta_score"] if not pd.isna(movie["Meta_score"]) else None,
                movie["No_of_Votes"]
            ))
        conn.commit()

# Function to hash a password
def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Function to verify a password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify if the plain password matches the hashed password.
    """
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# API Endpoints
@app.post("/signup/")
def signup(username: str, password: str):
    """
    User signup: Create a new user with a username and password.
    """
    # Hash the user's password
    hashed_password = hash_password(password)
    
    # Check if the username already exists
    existing_user = execute_query("SELECT id FROM users WHERE username = ?", (username,), fetch_one=True)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Insert into the database
    execute_query("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                  (username, hashed_password))
    
    return {"message": "User created successfully!"}

@app.post("/login/")
def login(username: str, password: str):
    """
    User login: Validate username and password.
    """
    # Fetch the user from the database
    stored_user = execute_query("SELECT id, password_hash FROM users WHERE username = ?", 
                                (username,), fetch_one=True)
    
    if not stored_user:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    # Verify the password
    user_id, hashed_password = stored_user
    if not verify_password(password, hashed_password):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    return {"message": f"User {username} logged in successfully!", "user_id": user_id}

@app.post("/users/{user_id}/favorite_genre/")
def update_favorite_genre(user_id: int, favorite_genres: List[str]):
    """
    Add or update the user's favorite genres.
    """
    # Ensure the user exists
    user = execute_query("SELECT favorite_genre FROM users WHERE id = ?", (user_id,), fetch_one=True)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update the user's favorite genres
    execute_query(
        "UPDATE users SET favorite_genre = ? WHERE id = ?",
        (json.dumps(favorite_genres), user_id)  # Serialize the list to a JSON string
    )

    return {"message": f"User {user_id}'s favorite genres updated successfully!"}

def get_content_based_recommendations(user_id, movies=None, user_movie_likes=None):
    """
    Content-based filtering: Recommend movies similar to those liked by the user.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        if movies is None:
            movies = pd.read_sql_query("SELECT id, title, genre, overview FROM movies", conn)
        if user_movie_likes is None:
            user_movie_likes = pd.read_sql_query("""
                SELECT user_id, movie_id, liked
                FROM user_movie_likes
            """, conn)

        # Get movies liked by the user
        liked_movies = user_movie_likes[
            (user_movie_likes['user_id'] == user_id) & (user_movie_likes['liked'] == 1)
        ]

        if liked_movies.empty:
            return []  # No liked movies, return an empty list

        # Generate content matrix
        movies['content'] = movies['genre'] + " " + movies['overview']
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['content'])

        # Compute cosine similarity
        liked_movie_ids = liked_movies['movie_id'].values
        liked_indices = movies[movies['id'].isin(liked_movie_ids)].index
        similarity_scores = cosine_similarity(tfidf_matrix[liked_indices], tfidf_matrix)

        # Aggregate and recommend
        movie_scores = np.mean(similarity_scores, axis=0)
        movies['score'] = movie_scores
        recommendations = movies[~movies['id'].isin(liked_movie_ids)].sort_values(by='score', ascending=False)
        return recommendations[['id', 'title', 'score']].head(10).to_dict('records')


def get_collaborative_recommendations(user_id):
    """
    Collaborative filtering: Recommend movies based on other users' activities.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        # Fetch user-movie interaction matrix
        user_movie_matrix = pd.read_sql_query("""
            SELECT user_id, movie_id, liked
            FROM user_movie_likes
        """, conn).pivot(index='user_id', columns='movie_id', values='liked').fillna(0)

        # Compute similarity between users
        user_similarity = cosine_similarity(user_movie_matrix)

        # Get the index of the current user
        user_index = user_movie_matrix.index.get_loc(user_id)

        # Weighted sum of movie scores from similar users
        similar_users = user_similarity[user_index]
        weighted_scores = np.dot(similar_users, user_movie_matrix.fillna(0))
        movie_scores = pd.Series(weighted_scores, index=user_movie_matrix.columns)

        # Exclude movies the user has already interacted with
        user_liked_movies = user_movie_matrix.loc[user_id]
        recommendations = movie_scores[~user_liked_movies.astype(bool)].sort_values(ascending=False)
        movie_ids = recommendations.head(10).index.tolist()

        # Get movie details
        cursor.execute(f"""
            SELECT id, title
            FROM movies
            WHERE id IN ({','.join('?' for _ in movie_ids)})
        """, movie_ids)
        return [{"id": row[0], "title": row[1], "score": recommendations[row[0]]} for row in cursor.fetchall()]

@app.get("/recommendations/hybrid/{user_id}")
def get_hybrid_recommendations(user_id: int):
    """
    Parallel hybrid recommendation system: Combines content-based filtering (CBF), collaborative filtering (CF),
    and a fallback to the most liked movies for new users.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        # Fetch user information
        user = execute_query("SELECT favorite_genre FROM users WHERE id = ?", (user_id,), fetch_one=True)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Fetch movies data
        movies = pd.read_sql_query("SELECT id, title, genre, overview FROM movies", conn)
        
        # Fetch user-movie likes (interaction data)
        user_movie_likes = pd.read_sql_query("""
            SELECT user_id, movie_id, liked
            FROM user_movie_likes
        """, conn)

    # Check if the user has any liked movies
    liked_movies = user_movie_likes[
        (user_movie_likes['user_id'] == user_id) & (user_movie_likes['liked'] == 1)
    ]

    if liked_movies.empty:  # New user or no activity
        # Fallback to most liked movies with detailed information
        most_liked_movies = pd.read_sql_query("""
            SELECT 
                movies.id, 
                movies.title, 
                movies.genre, 
                movies.overview, 
                COUNT(user_movie_likes.movie_id) AS like_count
            FROM movies
            LEFT JOIN user_movie_likes 
                ON movies.id = user_movie_likes.movie_id AND user_movie_likes.liked = 1
            GROUP BY movies.id
            ORDER BY like_count DESC
            LIMIT 10
        """, conn)
        return most_liked_movies.to_dict('records')

    # If the user has liked movies, proceed with hybrid recommendations
    cbf_recommendations = get_content_based_recommendations(user_id, movies, user_movie_likes)
    cf_recommendations = get_collaborative_recommendations(user_id)

    # Combine recommendations using weighted scores
    combined_scores = {}
    for rec in cbf_recommendations:
        combined_scores[rec['id']] = combined_scores.get(rec['id'], 0) + rec['score'] * 0.5
    for rec in cf_recommendations:
        combined_scores[rec['id']] = combined_scores.get(rec['id'], 0) + rec['score'] * 0.5

    # Fetch movie details and sort by combined score
    combined_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT id, title, genre, overview
            FROM movies
            WHERE id IN ({','.join('?' for _ in combined_recommendations)})
        """, [rec[0] for rec in combined_recommendations])
        movies = cursor.fetchall()

    return [
        {"id": movie[0], "title": movie[1], "genre": movie[2], "overview": movie[3], "score": combined_scores[movie[0]]}
        for movie in movies
    ]


@app.post("/users/{user_id}/like_movie/")
def like_movie(user_id: int, movie_id: int):
    """
    Add a like for a movie by a user.
    """
    # Ensure the user exists
    user = execute_query("SELECT id FROM users WHERE id = ?", (user_id,), fetch_one=True)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Ensure the movie exists
    movie = execute_query("SELECT id FROM movies WHERE id = ?", (movie_id,), fetch_one=True)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    # Insert or update the like status in the user_movie_likes table
    execute_query("""
        INSERT INTO user_movie_likes (user_id, movie_id, liked)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, movie_id) 
        DO UPDATE SET liked = ?
    """, (user_id, movie_id, True, True))  # Mark the movie as liked
    
    return {"message": f"User {user_id} liked movie {movie_id}"}

@app.get("/movies/search/")
def search_movies(genre: str = None, title: str = None):
    """
    Search for movies by title or genre.
    """
    query = "SELECT title, genre, imdb_rating FROM movies WHERE 1=1"
    params = []
    if genre:
        query += " AND genre LIKE ?"
        params.append(f"%{genre}%")
    if title:
        query += " AND title LIKE ?"
        params.append(f"%{title}%")

    movies = execute_query(query, tuple(params), fetch_all=True)
    if not movies:
        raise HTTPException(status_code=404, detail="No movies found")
    
    return [{"title": m[0], "genre": m[1], "imdb_rating": m[2]} for m in movies]





# @app.get("/recommendations/{user_id}")
# def get_recommendations(user_id: int):
#     """
#     Recommend movies based on the activity of other users.
#     """
#     # Get the list of movies the user has already liked
#     liked_movies = execute_query("""
#         SELECT movie_id FROM user_movie_likes 
#         WHERE user_id = ? AND liked = 1
#     """, (user_id,), fetch_all=True)
    
#     liked_movie_ids = [movie[0] for movie in liked_movies]

#     # Get all movies liked by other users
#     all_liked_movies = execute_query("""
#         SELECT user_id, movie_id 
#         FROM user_movie_likes 
#         WHERE movie_id NOT IN ({})
#     """.format(','.join('?' for _ in liked_movie_ids)), tuple(liked_movie_ids), fetch_all=True)
    
#     # Count how many users liked each movie
#     movie_like_count = {}
#     for user_id, movie_id in all_liked_movies:
#         if movie_id not in movie_like_count:
#             movie_like_count[movie_id] = 0
#         movie_like_count[movie_id] += 1

#     # Sort movies by popularity (most liked by other users)
#     sorted_movies = sorted(movie_like_count.items(), key=lambda x: x[1], reverse=True)

#     # Get details of the top recommended movies
#     recommended_movies = []
#     for movie_id, _ in sorted_movies[:10]:  # Limit to top 10 recommendations
#         movie = execute_query("SELECT title, genre, imdb_rating, overview FROM movies WHERE id = ?", (movie_id,), fetch_one=True)
#         if movie:
#             recommended_movies.append({
#                 "title": movie[0],
#                 "genre": movie[1],
#                 "imdb_rating": movie[2],
#                 "overview": movie[3]
#             })
    
#     if not recommended_movies:
#         raise HTTPException(status_code=404, detail="No recommendations found")
    
#     return recommended_movies


# @app.post("/users/{user_id}/like_movie/")
# def like_movie(user_id: int, movie_id: int):
#     """
#     Allow a user to like a movie. Add the movie's genre to the user's favorite genres.
#     """
#     # Ensure the user exists
#     user = execute_query("SELECT favorite_genre FROM users WHERE id = ?", (user_id,), fetch_one=True)
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     # Ensure the movie exists
#     movie = execute_query("SELECT genre FROM movies WHERE id = ?", (movie_id,), fetch_one=True)
#     if not movie:
#         raise HTTPException(status_code=404, detail="Movie not found")

#     # Get user's favorite genres and movie genres
#     favorite_genres = json.loads(user[0])  # Deserialize user's favorite genres
#     movie_genres = [g.strip() for g in movie[0].split(",")]  # Split and clean up movie genres

#     # Add new genres to user's favorite genres
#     updated_favorite_genres = set(favorite_genres)  # Use a set to avoid duplicates
#     updated_favorite_genres.update(g.lower() for g in movie_genres)  # Add movie genres in lowercase

#     # Update the user's favorite genres in the database
#     execute_query(
#         "UPDATE users SET favorite_genre = ? WHERE id = ?",
#         (json.dumps(list(updated_favorite_genres)), user_id)
#     )

#     return {"message": f"User {user_id}'s favorite genres updated successfully!", "updated_genres": list(updated_favorite_genres)}
