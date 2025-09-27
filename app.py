import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity

# Cache data loading and processing to avoid recomputing
@st.cache_data
def load_and_process_data():
    # Load data with explicit dtypes
    books = pd.read_csv("Books.csv", sep=";", encoding="latin-1", 
                        usecols=["ISBN", "Title", "Author", "Year", "Publisher"],
                        dtype={"ISBN": str}, on_bad_lines='skip')
    ratings = pd.read_csv("Ratings.csv", sep=";", encoding="latin-1", 
                         usecols=["User-ID", "ISBN", "Rating"],
                         dtype={"User-ID": str, "ISBN": str}, on_bad_lines='skip')
    users = pd.read_csv("Users.csv", sep=";", encoding="latin-1", 
                       usecols=["User-ID", "Age"],
                       dtype={"User-ID": str}, on_bad_lines='skip')

    # Filter active users (>=50 ratings)
    user_counts = ratings['User-ID'].value_counts()
    active_users = user_counts[user_counts >= 50].index
    ratings = ratings[ratings['User-ID'].isin(active_users)]

    # Filter popular books (>=20 ratings)
    book_counts = ratings['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= 20].index

    # Ensure popular_books only includes ISBNs present in books
    popular_books = popular_books[popular_books.isin(books['ISBN'])]

    # Filter ratings and books
    ratings = ratings[ratings['ISBN'].isin(popular_books)]
    books = books[books['ISBN'].isin(popular_books)]

    # Create user-item matrix
    rating_matrix = ratings.pivot(index='User-ID', columns='ISBN', values='Rating').fillna(0)

    # Apply SVD with k latent factors
    k = 50  # Number of latent factors
    U, sigma, Vt = svd(rating_matrix, full_matrices=False)
    U = U[:, :k]
    sigma = np.diag(sigma[:k])
    Vt = Vt[:k, :]

    return books, rating_matrix, Vt

# Load data
books, rating_matrix, Vt = load_and_process_data()

# Recommendation function
def get_similar_books(isbn, rating_matrix, Vt, books, num_similar=5):
    if isbn not in rating_matrix.columns:
        return None  # Return None for invalid ISBN
    book_idx = rating_matrix.columns.get_loc(isbn)
    book_vector = Vt[:, book_idx].reshape(1, -1)
    similarities = cosine_similarity(book_vector, Vt.T)[0]
    similar_indices = similarities.argsort()[::-1][1:num_similar+1]
    similar_isbns = rating_matrix.columns[similar_indices]
    similar_scores = similarities[similar_indices]
    similar_books = pd.DataFrame({
        'ISBN': similar_isbns,
        'Similarity': similar_scores
    })
    similar_books = similar_books.merge(
        books[['ISBN', 'Title', 'Author']],
        on='ISBN',
        how='left'
    )
    return similar_books[['Title', 'Author', 'Similarity']]

# Sidebar: Display sample ISBNs
st.sidebar.title("Sample ISBNs")
sample_isbns = ["000649840X", "0020199600", "0007154615", "0020198906", "002026478X"]
sample_df = pd.DataFrame({'ISBN': sample_isbns})
sample_info = sample_df.merge(books[['ISBN', 'Title', 'Author']], on='ISBN', how='left')
# Verify all ISBNs exist to avoid displaying invalid ones
sample_info = sample_info[sample_info['ISBN'].isin(rating_matrix.columns)]
st.sidebar.write("Copy an ISBN below to try:")
st.sidebar.dataframe(sample_info[['ISBN', 'Title', 'Author']], use_container_width=True)

# Main app
st.title("Book Recommender using SVD by ISBN")

# ISBN input
isbn_input = st.text_input("Enter an ISBN (e.g., 000649840X):", "")

# Display book details
if isbn_input:
    if isbn_input in rating_matrix.columns:
        book_info = books[books['ISBN'] == isbn_input][['Title', 'Author']].iloc[0]
        st.write(f"**Book Found**")
        st.write(f"**Title**: {book_info['Title']}")
        st.write(f"**Author**: {book_info['Author']}")
    else:
        st.warning("ISBN not found in the rating matrix. Try a sample ISBN from the sidebar.")

# Recommend button
if st.button("Recommend"):
    if isbn_input:
        recommendations = get_similar_books(isbn_input, rating_matrix, Vt, books, num_similar=5)
        if recommendations is not None:
            st.write(f"**Books similar to ISBN {isbn_input}:**")
            st.dataframe(recommendations, use_container_width=True)
        else:
            st.error("ISBN not found. Please enter a valid ISBN from the sidebar.")
    else:
        st.error("Please enter an ISBN before recommending.")