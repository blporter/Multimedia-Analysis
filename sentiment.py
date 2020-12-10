__author__ = 'Brooke Porter'
__class__ = 'CSC 664 Multimedia Systems (Undergraduate)'
__school__ = 'San Francisco State University'
__professor__ = 'Dr. Rahul Singh'
__file__ = 'sentiment.py'

"""
This file is used to parse the tweets retrieved from app.py into drug-related tweets with positive sentiment.
It uses a Naive Bayes Classifier to train using an open-source Kaggle dataset from
https://www.kaggle.com/kazanova/sentiment140 to determine whether a given tweet holds a positive or negative sentiment.
First on the training dataset, it tokenizes and then lemmatizes and cleans the tweet bodies into base words, then
again on the real data. The real data is also filtered by tweets relating to the list of drug words. Caching is
implemented on the cleaned and lemmatized data to improve run times.
"""

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

slang_drug_words = ["mj", "weed", "marijuana", "heroin", "crack", "lit", "dank", "dope", "pot", "adderall", "oxy"]
tweet_fields = ['user_id', 'stat_id', 'creation', 'tweet_body', 'name']
training_fields = ['target', 'ids', 'date', 'flag', 'user', 'text']
positive_tweets = []
negative_tweets = []

# For the purposes of running and testing, the list of supplied csv files can be changed. These should be
# in the data folder.
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
        """
        Tokenize a csv file containing real tweets into arrays of words, given the path to the file. This
        is used for real data, and uses the tweet_fields global list to determine the key of the tweet body.
        :param file_path: The required file path to the csv being parsed.
        :return: The array of arrays of string words.
        """
        print("Tokenizing csv file")
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        with open(file_path, newline='') as file:
            reader = csv.DictReader(file, fieldnames=tweet_fields)
            content = []
            for row in reader:
                content.append(tokenizer.tokenize(row['tweet_body']))
            return content

    def tokenize_training_model(self, tweets):
        """
        This function is called after splitting the training file into positive and negative sentiment halves.
        It takes a list of tweet bodies and tokenizes them, returning the list of tokenized lists of tweets.
        :param tweets: A list of positive or negative sentiment tweet bodies.
        :return: A list of tokenized lists of words.
        """
        print("Tokenizing training model")
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        content = []
        for row in tweets:
            content.append(tokenizer.tokenize(row))
        return content

    def clean_content(self, content):
        """
        This function cleans a given list of tokenized lists of words by lemmatizing words into their base
        forms, and removing unnecessary data from the lists of words. HTTP links are removed, as well as @mentions
        and common stop-list words.
        :param content: The list of tokenized lists of words.
        :return: The list of cleaned and lemmatized lists of words.
        """
        print("Lemmatizing and cleaning")
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words("english")
        cleaned_content = []
        for row in content:
            cleaned_sentence = []
            for word, tag in pos_tag(row):
                # Remove links (http(s) and everything after)
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
        """
        This function converts the cleaned and lemmatized tweet tokens into a dict suitable for the model
        to work with.
        :param cleaned_content: The list of cleaned and lemmatized lists of words.
        """
        for tokens in cleaned_content:
            yield dict([token, True] for token in tokens)


def download_nltk_libraries():
    """
    This function should be used to download the nltk libraries for use with this program, if they are not
    already downloaded. It only needs to be done once.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download()
    print("NLTK download complete.")


def split_training_file():
    """
    We use a csv file containing training data supplied from https://www.kaggle.com/kazanova/sentiment140.
    The csv file is too large to commit to Github, and will likely require manual downloading from the above link.
    This function splits that csv file into positive and negative sentiment halves, and uses the training_fields
    list of keys to access the tweet body, adding it to the relevant positive or negative list.
    """
    print("Splitting training file into positive and negative")
    with open('data/training.1600000.processed.noemoticon.csv', newline='', encoding='latin-1') as file:
        reader = csv.DictReader(file, fieldnames=training_fields)
        for row in reader:
            if int(row['target']) < 2:
                negative_tweets.append(row['text'])
            else:
                positive_tweets.append(row['text'])


def write_header(file_path):
    """
    This is a generic function for writing the header for a csv file that does not yet have one.
    :param file_path: The path to the csv file being written to.
    """
    with open(file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=['text', 'sentiment'])
        # The file doesn't exist yet, so write a header
        if not os.path.isfile(file_path):
            writer.writeheader()


def write_cache(file_path, data):
    """
    This is a generic function for writing tokenized data to a csv file for caching. Cleaning and lemmatizing the
    data is the most time consuming step in the program, so saving the data from that step makes the program
    easier to work with.
    :param file_path: The path where the csv file should be created.
    :param data: The list of data to be written to the cache file.
    """
    with open(file_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow(row)
    print("Wrote to cache")


def read_cache(file_path):
    """
    This is a generic function for reading tokenized data from a cached csv file. Cleaning and lemmatizing the
    data is the most time consuming step in the program, so skipping that step by reading cached data makes the
    program easier to work with.
    :param file_path: The path to the cached csv file.
    :return: The list of content from the cache.
    """
    with open(file_path, newline='') as file:
        reader = csv.reader(file)
        content = []
        for row in reader:
            content.append(row)
        return content


def fetch_featured_tweets():
    """
    This is a helper function to handle the real tweet csv data. It first checks the global list of tweet files for
    whether or not that file exists in the cache folder. If that file does exist in the cache, then it reads that
    file and checks it against the slang_drug_words global list. Only tweets containing those tokenized words
    are added to be returned. If the file does not exist in the cache, it is first tokenized and then added to said
    cache.
    :return: The tokenized and relevant tweets, as well as their original counterparts.
    """
    print("Fetching featured tweets")
    analysis = SentimentAnalysis()
    num_original_tweets = 0
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
            num_original_tweets += 1
            does_have_drugs_words = len([word for word in tokenized_tweet if word in slang_drug_words]) > 0
            if does_have_drugs_words:
                tweets_to_return.append(tokenized_tweet)
                original_tweets.append(og_tweet[3])

    return tweets_to_return, original_tweets, num_original_tweets


def main():
    should_download = input("Do you need to download nltk libraries? [y/n] ")
    if should_download == "y":
        download_nltk_libraries()

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

    # The dataset needs to be converted to a dict applicable for training.
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_content_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_content_for_model]

    # The positive and negative sentiment halves to train off of should be combined again, and the order randomized.
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    # train the first 70%, test the last 30%. We have 1.6 million tweets in our training data.
    train_data = dataset[:1120000]
    test_data = dataset[1120000:]

    print("Training using dataset")
    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

    # After training, we can repeat the process using real data.
    tokenized_tweets, og_tweets, num_original_tweets = fetch_featured_tweets()
    assert len(tokenized_tweets) == len(og_tweets)
    cleaned_drug_tokens = analysis.clean_content(tokenized_tweets)

    print("Running network on real tweets")
    num_positives = 0
    for idx, tokens in enumerate(cleaned_drug_tokens):
        original_tweet = og_tweets[idx]
        token_dict = dict([token, True] for token in tokens)
        try:
            # We instruct our network to classify each tweet, and only output Positive sentiment tweets.
            classified = classifier.classify(token_dict)
            if classified == 'Positive':
                num_positives += 1
                print(original_tweet, "=>", classified)
        except Exception:
            print("exception")

    print("\nTotal original tweets:", num_original_tweets)
    print("Total drug related tweets:", len(cleaned_drug_tokens))
    print("Percent of original tweets that are drug related:", len(cleaned_drug_tokens) / num_original_tweets)
    print("Total number of positive sentiment tweets:", num_positives)
    print("Percent of drug related tweets with positive sentiment:", num_positives / len(cleaned_drug_tokens))

    return 0


if __name__ == '__main__':
    main()

