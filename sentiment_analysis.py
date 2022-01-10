import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

np.random.seed(544)

# Load data and drop rows with missing values
fname = 'amazon_reviews_us_Kitchen_v1_00.tsv.gz'
original_data = pd.read_csv(fname, sep='\t', compression='gzip', on_bad_lines='skip')
data = original_data[['star_rating', 'review_body']].copy()
data.dropna(inplace=True)

# Display statistics of three classes (with comma between them)
count_positive = len(data[ data['star_rating'].isin([4, 5]) ])
count_negative = len(data[ data['star_rating'].isin([1, 2]) ])
count_neutral = len(data[ data['star_rating'].isin([3]) ])
print(f'{count_positive}, {count_negative}, {count_neutral}')

# Discard reviews with rating of 3
filter_index = data[ data['star_rating'] == 3 ].index
data.drop(filter_index, inplace=True)

# Map ratings to sentiment
rating_mapping = {1: 0, 2: 0, 4: 1, 5: 1}
data['star_rating'].replace(rating_mapping, inplace=True)
data['star_rating'] = data['star_rating'].astype('int8')


# Get 100k samples for each class, and then concatenate them making it the data we'll use
positive_reviews = data[ data['star_rating'] == 1 ].sample(100000)
negative_reviews = data[ data['star_rating'] == 0 ].sample(100000)
data = pd.concat([positive_reviews, negative_reviews])


## Data Cleaning

# Create a new column for cleaned reviews, which are lowercase
data['cleaned_reviews'] = data['review_body'].str.lower()

# Function to remove HTML tags and URLs from a string
def sanitize_review(text):
    # remove HTML tags
    text = BeautifulSoup(str(text), 'html.parser').get_text()   
    # remove URLS
    text = re.sub(r'http\S+', '', str(text))
    return text

data['cleaned_reviews'] = data['cleaned_reviews'].apply(sanitize_review)

# Perform contractions on the reviews
import contractions

def fix_contractions(text):
    return contractions.fix(text)

data['cleaned_reviews'] = data['cleaned_reviews'].apply(fix_contractions)

# Remove non-alphabetical characters
data['cleaned_reviews'] = data['cleaned_reviews'].str.replace('[^a-zA-Z\s]', ' ')

# ## Remove the extra spaces between the words
def remove_extra_spaces(text):
    return ' '.join(str(text).split())

data['cleaned_reviews'] = data['cleaned_reviews'].apply(remove_extra_spaces)

# Print cleaning results
avg_before_clean = data['review_body'].apply(lambda x: len(str(x))).mean()
avg_after_clean = data['cleaned_reviews'].apply(lambda x: len(str(x))).mean()
print(f'{avg_before_clean:.0f}, {avg_after_clean:.0f}')


## Pre-processing

# Remove the stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

def remove_stop_words(text):
    return ' '.join([word for word in str(text).split() if word not in (stop)])

data['processed_reviews'] = data['cleaned_reviews'].apply(remove_stop_words)


# Perform lemmatization
from nltk.stem import WordNetLemmatizer

tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in tokenizer.tokenize(text)])

data['processed_reviews'] = data['processed_reviews'].apply(lemmatize)

# Print pre-processing results
avg_before_preprocessing = data['cleaned_reviews'].apply(lambda x: len(str(x))).mean()
avg_after_preprocessing = data['processed_reviews'].apply(lambda x: len(str(x))).mean()
print(f'{avg_before_preprocessing:.0f}, {avg_after_preprocessing:.0f}')

# Training and Testing data split
from sklearn.model_selection import train_test_split

# Perform an 80-20 split for training and testing data (using the cleaned+pre-processed reviews)
review_train, review_test, y_train, y_test = train_test_split(data['processed_reviews'], data['star_rating'], test_size=0.2)

# TF-IDF Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(review_train)
X_test = vectorizer.transform(review_test)


# Helper function to report results for each model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def report_results(y_train_true, y_train_pred, y_test_true, y_test_pred):
    train_accuracy = accuracy_score(y_train_true, y_train_pred)
    train_precision = precision_score(y_train_true, y_train_pred)
    train_recall = recall_score(y_train_true, y_train_pred, average='macro')
    train_f1 = f1_score(y_train_true, y_train_pred)
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_precision = precision_score(y_test_true, y_test_pred)
    test_recall = recall_score(y_test_true, y_test_pred, average='macro')
    test_f1 = f1_score(y_test_true, y_test_pred)
    print(f'{train_accuracy:.3f}, {train_precision:.3f}, {train_recall:.3f}, {train_f1:.3f}, {test_accuracy:.3f}, {test_precision:.3f}, {test_recall:.3f}, {test_f1:.3f}')

# Perceptron
from sklearn.linear_model import Perceptron

model = Perceptron()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

report_results(y_train, y_train_pred, y_test, y_test_pred)

# SVM
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

report_results(y_train, y_train_pred, y_test, y_test_pred)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1500)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

report_results(y_train, y_train_pred, y_test, y_test_pred)

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

report_results(y_train, y_train_pred, y_test, y_test_pred)