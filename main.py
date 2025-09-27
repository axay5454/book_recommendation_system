import pandas as pd

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

# Print initial shapes
print("Initial Books:", books.shape)
print("Initial Ratings:", ratings.shape)
print("Initial Users:", users.shape)

# Check for missing values in raw data
print("\nMissing Values in Books:")
print(books[['ISBN', 'Title', 'Author']].isna().sum())
print("\nMissing Values in Ratings:")
print(ratings[['User-ID', 'ISBN', 'Rating']].isna().sum())
print("\nMissing Values in Users:")
print(users[['User-ID', 'Age']].isna().sum())

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

# Print shapes and stats after filtering
print("\nFiltered Books:", books.shape)
print("Filtered Ratings:", ratings.shape)
print("Rating Matrix Shape:", rating_matrix.shape)
print(f"Books retained: {books.shape[0]/271379*100:.2f}% of original")
print(f"Users retained: {rating_matrix.shape[0]/278859*100:.2f}% of original")

# Check for missing values in rating matrix
print("\nMissing Values in Rating Matrix:")
print(rating_matrix.isna().sum().sum())  # Total NaNs (should be 0)

# Print 5 sample ISBNs from rating matrix with details
sample_isbns = rating_matrix.columns[:5].tolist()
isbn_df = pd.DataFrame({'ISBN': sample_isbns})
isbn_info = isbn_df.merge(books[['ISBN', 'Title', 'Author']], on='ISBN', how='left')
print("\nSample ISBNs from Rating Matrix:")
print(isbn_info)

from numpy.linalg import svd
import numpy as np

# Apply SVD with k latent factors
k = 50  # Number of latent factors
U, sigma, Vt = svd(rating_matrix, full_matrices=False)

# Truncate to k factors
U = U[:, :k]
sigma = np.diag(sigma[:k])
Vt = Vt[:k, :]

# Print shapes to verify
print("U Shape:", U.shape)
print("Sigma Shape:", sigma.shape)
print("Vt Shape:", Vt.shape)

# Part 3: ISBN-Based Recommendations
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_books(isbn, rating_matrix, Vt, books, num_similar=5):
    # Check if ISBN exists in the rating matrix
    if isbn not in rating_matrix.columns:
        return f"ISBN {isbn} not found in the rating matrix"
    
    # Get the index of the ISBN in the matrix
    book_idx = rating_matrix.columns.get_loc(isbn)
    
    # Get the latent vector for the input ISBN
    book_vector = Vt[:, book_idx].reshape(1, -1)
    
    # Compute cosine similarities with all other books
    similarities = cosine_similarity(book_vector, Vt.T)[0]
    
    # Get indices of top similar books (excluding the input book)
    similar_indices = similarities.argsort()[::-1][1:num_similar+1]
    similar_isbns = rating_matrix.columns[similar_indices]
    similar_scores = similarities[similar_indices]
    
    # Create dataframe with similar books
    similar_books = pd.DataFrame({
        'ISBN': similar_isbns,
        'Similarity': similar_scores
    })
    
    # Merge with books dataframe to get titles and authors
    similar_books = similar_books.merge(
        books[['ISBN', 'Title', 'Author']],
        on='ISBN',
        how='left'
    )
    
    # Return relevant columns
    return similar_books[['Title', 'Author', 'Similarity']]

# Test with sample ISBN
test_isbn = "000649840X"  # "Angelas Ashes" by Frank McCourt
print(f"\nBooks similar to ISBN {test_isbn} (Angelas Ashes):")
print(get_similar_books(test_isbn, rating_matrix, Vt, books, num_similar=5))

test_isbn_2 = "0020199600"  # "Great Gatsby"
print(f"\nBooks similar to ISBN {test_isbn_2} (Great Gatsby):")
print(get_similar_books(test_isbn_2, rating_matrix, Vt, books, num_similar=5))