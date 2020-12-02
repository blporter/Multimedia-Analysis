import nltk
import ssl
import csv
import re
import string
import random
import os

from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier

slang_drug_word = ["mj", "weed", "marijuana", "heroin", "crack", "lit", "dank", "dope", "pot", "adderall", "oxy"]
tweet_fields = ['user_id', 'stat_id', 'creation', 'tweet_body', 'name']
training_fields = ['target', 'ids', 'date', 'flag', 'user', 'text']
positive_tweets = []
negative_tweets = []

tweet_files = [
    "tweets.csv",
    "tweets2.csv",
    "tweets3.csv",
    "tweets4.csv",
    "tweets5.csv",
    "tweets6.csv",
    "tweets7.csv",
    "tweets8.csv",
]


class SentimentAnalysis:
    def __init__(self):
        super(SentimentAnalysis, self).__init__()

    def tokenize_csv(self, file_path):
        print("Tokenizing csv file")
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        with open(file_path, newline='') as file:
            reader = csv.DictReader(file, fieldnames=tweet_fields)
            content = []
            for row in reader:
                content.append(tokenizer.tokenize(row['tweet_body']))
            return content

    def tokenize_training_model(self, tweets):
        print("Tokenizing training model")
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        content = []
        for row in tweets:
            content.append(tokenizer.tokenize(row))
        return content

    def clean_content(self, content):
        print("Lemmatizing and cleaning")
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words("english")
        cleaned_content = []
        for row in content:
            cleaned_sentence = []
            for word, tag in pos_tag(row):
                # remove links (http(s) and everything after)
                word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                              word)

                if tag.startswith('NN'):
                    pos = 'n'
                elif tag.startswith('VB'):
                    pos = 'v'
                else:
                    pos = 'a'

                token = lemmatizer.lemmatize(word, pos)
                if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                    cleaned_sentence.append(token.lower())

            cleaned_content.append(cleaned_sentence)
        return cleaned_content

    def prepare_content_for_model(self, cleaned_content):
        for tokens in cleaned_content:
            yield dict([token, True] for token in tokens)


def download_nltk_libraries():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download()
    print("NLTK download complete.")


# My training dataset from Kaggle is too large to commit to github. Download the dataset from the following link:
# https://www.kaggle.com/kazanova/sentiment140
# and store it in the data folder.
def split_training_file():
    print("Splitting training file into positive and negative")
    with open('data/training.1600000.processed.noemoticon.csv', newline='', encoding='latin-1') as file:
        reader = csv.DictReader(file, fieldnames=training_fields)
        for row in reader:
            if int(row['target']) < 2:
                negative_tweets.append(row['text'])
            else:
                positive_tweets.append(row['text'])


def write_header(file_path):
    with open(file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=['text', 'sentiment'])
        # file doesn't exist yet, write a header
        if not os.path.isfile(file_path):
            writer.writeheader()


def write_cache(file_path, data):
    with open(file_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow(row)
    print("Wrote to cache")


def read_cache(file_path):
    with open(file_path, newline='') as file:
        reader = csv.reader(file)
        content = []
        for row in reader:
            content.append(row)
        return content


def fetch_featured_tweets():
    analysis = SentimentAnalysis()
    tweets_to_return = []
    original_tweets = []

    for tweet_file in tweet_files:
        if os.path.isfile("cache/" + tweet_file):
            data = read_cache("cache/" + tweet_file)
        else:
            data = analysis.tokenize_csv("data/" + tweet_file)
            write_cache("cache/" + tweet_file, data)

        zipped_data = zip(data, read_cache("data/" + tweet_file))

        for tokenized_tweet, og_tweet in zipped_data:
            does_have_drugs_words = len([word for word in tokenized_tweet if word in slang_drug_word]) > 0
            if does_have_drugs_words:
                tweets_to_return.append(tokenized_tweet)
                original_tweets.append(og_tweet[3])

    return tweets_to_return, original_tweets


def main():
    # download_nltk_libraries()
    analysis = SentimentAnalysis()

    # If the cleaned and tokenized data is already cached, pull from that
    if os.path.isfile('cache/cleaned_training_data_negative_cache.csv'):
        cleaned_positive_content = read_cache('cache/cleaned_training_data_positive_cache.csv')
        cleaned_negative_content = read_cache('cache/cleaned_training_data_negative_cache.csv')
        print("Read from cache")
    else:
        # Otherwise, clean and tokenize the data and then cache it.
        split_training_file()

        positive_tokens = analysis.tokenize_training_model(positive_tweets)
        negative_tokens = analysis.tokenize_training_model(negative_tweets)
        cleaned_positive_content = analysis.clean_content(positive_tokens)
        cleaned_negative_content = analysis.clean_content(negative_tokens)

        write_header('cache/cleaned_training_data_positive_cache.csv')
        write_header('cache/cleaned_training_data_negative_cache.csv')
        write_cache('cache/cleaned_training_data_positive_cache.csv', cleaned_positive_content)
        write_cache('cache/cleaned_training_data_negative_cache.csv', cleaned_negative_content)

    positive_content_for_model = analysis.prepare_content_for_model(cleaned_positive_content)
    negative_content_for_model = analysis.prepare_content_for_model(cleaned_negative_content)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_content_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_content_for_model]

    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    # train the first 70%, test the last 30%. We have 1.6 million tweets in our training data.
    train_data = dataset[:1120000]
    test_data = dataset[1120000:]

    print("Training using dataset")
    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

    tokenized_tweets, og_tweets = fetch_featured_tweets()
    assert len(tokenized_tweets) == len(og_tweets)
    cleaned_drug_tokens = analysis.clean_content(tokenized_tweets)

    print("Running network on real tweets")
    for idx, tokens in enumerate(cleaned_drug_tokens):
        original_tweet = og_tweets[idx]
        token_dict = dict([token, True] for token in tokens)
        try:
            classified = classifier.classify(token_dict)
            if classified == 'Positive':
                print(original_tweet, "=>", classified)
        except Exception:
            print("exception")
    return 0


if __name__ == '__main__':
    main()

