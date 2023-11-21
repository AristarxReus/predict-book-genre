# predict-book-genre
# Book Recommender System
 This project is a book recommender system that uses Natural Language Processing (NLP) techniques to recommend books based on user input in telegram bot. It utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) matrix and Nearest Neighbors model to find books similar to the user's input.

## Data Loading and Preprocessing
The project starts by loading a dataset of book summaries and genres from a CSV file. It then preprocesses the text data by converting it to lowercase, removing punctuation, tokenizing the text, lemmatizing words, and removing stop words using NLTK (Natural Language Toolkit) library.

## TF-IDF Matrix Creation
After preprocessing the text data, the project creates a TF-IDF matrix using the TfidfVectorizer from scikit-learn. This matrix represents the importance of each word in the corpus of book summaries and genres.

## Nearest Neighbors Model Training
The Nearest Neighbors model is trained using the cosine similarity metric on the TF-IDF matrix. This allows the model to find books that are similar to the user's input based on their summaries and genres.

## Book Recommendation Function
The project provides a recommend_book function that takes a book summary, genre, and the number of books to recommend as input. It then finds the nearest neighbors of the input data in the TF-IDF matrix and returns a list of recommended books based on their similarity to the input.

## Usage
To use the book recommender system, simply call the recommend_book function with a book summary, genre, and the number of books to recommend. The function will return a list of recommended books based on the input. Or insert your token from the telegram bot into this code and make a request to it
recommend_book("A gripping tale of love and betrayal", "Romance", num_books=5)
## Dependencies
- pandas
- nltk
- scikit-learn
- tqdm
- aiogram
