import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk, re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from termcolor import colored

campaign = open(str(sys.argv[1]), 'r')
entry_text = campaign.read()

stop_words = stopwords.words('english')
normalizer = WordNetLemmatizer()

articles = [entry_text]*5

def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  pos_counts = Counter()
  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech

def preprocess_text(text):
  cleaned = re.sub(r'\W+', ' ', text).lower()
  tokenized = word_tokenize(cleaned)
  normalized = " ".join([normalizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized if not re.match(r'\d+',token)])
  return normalized

# preprocess articles
processed_articles = [preprocess_text(a) for a in articles]

# initialize and fit CountVectorizer
vectorizer = CountVectorizer()

counts = vectorizer.fit_transform(processed_articles)

# convert counts to tf-idf
transformer = TfidfTransformer(norm=None)

# initialize and fit TfidfVectorizer
tfidf_scores_transformed = transformer.fit_transform(counts)

# check if tf-idf scores are equal
vectorizer = TfidfVectorizer(norm=None)

tfidf_scores = vectorizer.fit_transform(processed_articles)

# get vocabulary of terms
try:
  feature_names = vectorizer.get_feature_names()
except:
  pass

# get article index
try:
  article_index = [f"Article {i+1}" for i in range(len(articles))]
except:
  pass

# create pandas DataFrame with word counts
try:
  df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
  #print(df_word_counts)
except:
  pass

# create pandas DataFrame(s) with tf-idf scores
try:
  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)
  #print(df_tf_idf)
except:
  pass

try:
  df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
  #print(df_tf_idf)
except:
  pass

def aux(article):
     # For the average polarity score
     # print(article)
     print(colored('Analyzing ...','white'))
     import time
     time.sleep(4)
     vader = SentimentIntensityAnalyzer()
     score = vader.polarity_scores(article).get('compound')*100
     ret_text = '[*] Success Campaign percentage: {}%'.format(score)
     print(colored(ret_text, 'green'))

aux(articles[0])
