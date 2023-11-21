import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import Dispatcher
import re, sqlite3
from aiogram.types import ReplyKeyboardMarkup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)

TOKEN='YOUR_TOKEN'

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


df = pd.read_csv('data.csv')
df

# Загрузка списка стоп-слов и инициализация лемматизатора
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Функция для предварительной обработки текста
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Применение предварительной обработки к текстам
tqdm.pandas()  # Используем tqdm для отслеживания прогресса
df['processed_text'] = df['summary'] + ' ' + df['genre']
df['processed_text'] = df['processed_text'].progress_apply(preprocess_text)

# Создание TF-IDF матрицы
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

# Обучение модели Nearest Neighbors
model = NearestNeighbors(metric='cosine')
model.fit(tfidf_matrix)

def recommend_book(summary, genre, num_books=5):
    genres = []
    input_text = summary + ' ' + genre
    input_vector = tfidf_vectorizer.transform([preprocess_text(input_text)])

    # Поиск ближайших соседей для введенных данных
    distances, indices = model.kneighbors(input_vector, n_neighbors=num_books+1)

    # Вывод рекомендованных книг на основе близости к введенным данным
    for idx in indices.flatten()[1:]:
        genres.append(f"Book: {df['title'][idx]}")

    return genres
# Начало диалога с ботом
@dp.message_handler(commands=['begin'])
async def start(callback_query: types.CallbackQuery):
    await bot.send_message(
        callback_query.from_user.id,
        text='Добро пожаловать в модель помошник подбора литературы по жанру/описанию!\nДля начала работы введите любой текст',
    )
# Начало диалога с ботом
@dp.message_handler(commands=['start'])
async def start(callback_query: types.CallbackQuery):
    await bot.send_message(
        callback_query.from_user.id,
        text='Добро пожаловать в модель помошник подбора литературы по жанру/описанию!\nДля начала работы введите любой текст',
    )
# Функция для ответа на основе обученной модели
@dp.message_handler()
async def registration_callback(callback_query: types.Message):
    user_input_text = callback_query.text.split(',')

    summary_input = user_input_text[0]
    genre_input = user_input_text[1]
    genres = recommend_book(summary_input, genre_input)
    user_id = callback_query.from_user.id
    request = callback_query.text
    connection = sqlite3.connect('db.sqlite3')
    cursor = connection.cursor()
    cursor.execute("INSERT INTO requests (user_id, request_text) VALUES ('%s', '%s')"%(user_id, request))

    connection.commit()
    connection.close()

    await bot.send_message(
        callback_query.from_user.id,
        text="Подходящие книги по данному жанру и теме:",
    )

    for genre in genres:
        await bot.send_message(
            callback_query.from_user.id,
            text=genre,
        )

executor.start_polling(dp, skip_updates=True)

executor.start_polling(dp, skip_updates=True)
