# Book Recommender using SVD by ISBN

The application is live at: [Streamlit App](https://svd-book-recommender.streamlit.app/)

This project implements a book recommendation system that suggests similar books based on a user-provided ISBN. It uses Singular Value Decomposition (SVD) to compute book similarities from user ratings, built from scratch with Python. The project includes a command-line script (`main.py`) for core implementation and a Streamlit web application (`app.py`) for an interactive user interface.

## Project Overview

The recommender system:
- Loads book data (`Books.csv`), user ratings (`Ratings.csv`), and user info (`Users.csv`) from the [Book-Crossing Dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset).
- Filters active users (≥50 ratings) and popular books (≥20 ratings) to create a user-item rating matrix.
- Applies SVD to extract latent factors (`k=50`) for books.
- Computes cosine similarity between books to recommend the top 5 similar books for a given ISBN.
- Provides a Streamlit app to input ISBNs, view book details (title, author), and get recommendations.

## Repository Structure

- `main.py`: Core implementation of the recommender system, including data loading, filtering, SVD, and recommendation logic. Outputs sample ISBNs and recommendations to the console.
- `app.py`: Streamlit web application that wraps `main.py`’s logic. Allows users to input an ISBN, displays title/author, and shows recommendations on a button click. Includes a sidebar with sample ISBNs.
- `requirements.txt`: Lists Python dependencies (e.g., pandas, numpy, scikit-learn, streamlit).
- `Books.csv`, `Ratings.csv`, `Users.csv`: Dataset files (not included in repo; see [Dataset](#dataset)).

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aatifahmad123/svd_book_recommender.git
   cd svd_book_recommender
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Command-Line Script (`main.py`)

`main.py` processes the data, applies SVD, and outputs sample ISBNs with recommendations.

1. Ensure the dataset files are in the root directory.
2. Run the script:
   ```bash
   python main.py
   ```
3. **Output**:
   - Initial and filtered dataset shapes.
   - Missing value checks.
   - Sample ISBNs with titles/authors (e.g., `000649840X` → "Angela's Ashes").
   - Recommendations for a test ISBN (e.g., `000649840X`):
     ```
     Books similar to ISBN 000649840X (Angelas Ashes):
                           Title           Author  Similarity
     0           Shipping News   E Annie Proulx    0.825017
     1       Jitterbug Perfume      Tom Robbins    0.820888
     ...
     ```

### Running the Streamlit App (`app.py`)

`app.py` provides an interactive web interface for recommendations.

1. Ensure the dataset files are in the root directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.
4. **Features**:
   - **Sidebar**: Displays sample ISBNs (e.g., `000649840X`, `0020199600`) with titles and authors.
   - **Main Interface**:
     - Enter an ISBN (e.g., copy from sidebar).
     - View the book’s title and author if valid.
     - Click "Recommend" to see a table of 5 similar books with titles, authors, and similarity scores.
   - **Error Handling**: Warns for invalid/empty ISBNs.
5. **Example Interaction**:
   - Input: `000649840X`
   - Output: 
     ```
     Book Found
     Title: Angela's Ashes
     Author: Frank McCourt
     ```
   - Click Recommend:
     ```
     Books similar to ISBN 000649840X:
     Title                    Author            Similarity
     Shipping News           E Annie Proulx     0.825017
     Jitterbug Perfume       Tom Robbins       0.820888
     ...
     ```

## Dataset

The project uses the [Book-Crossing Dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset), containing:
- `Books.csv`: Book metadata (ISBN, Title, Author, Year, Publisher).
- `Ratings.csv`: User ratings (User-ID, ISBN, Rating).
- `Users.csv`: User info (User-ID, Age).

## Contact

For questions or suggestions, contact Aatif Ahmad at <b22ai002@iitj.ac.in>.

