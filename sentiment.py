import nltk
import ssl
import csv
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

fieldnames = ['user_id', 'stat_id', 'creation', 'tweet_body', 'name']


class SentimentAnalysis:
    content = []
    cleaned_content = []

    def __init__(self):
        super(SentimentAnalysis, self).__init__()

    def tokenize_csv(self):
        print("Tokenizing.")
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        with open('data/tweets.csv', newline='') as file:
            reader = csv.DictReader(file, fieldnames=fieldnames)
            for row in reader:
                self.content.append(tokenizer.tokenize(row['tweet_body']))

    def clean_content(self):
        print("Lemmatizing and cleaning.")
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words("english")
        for row in self.content:
            cleaned_sentence = []
            for word, tag in pos_tag(row):
                # remove links (http(s) and everything after)
                word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                              word)
                # word = re.sub("(@[A-Za-z0-9_]+)", "", word)

                if tag.startswith('NN'):
                    pos = 'n'
                elif tag.startswith('VB'):
                    pos = 'v'
                else:
                    pos = 'a'

                token = lemmatizer.lemmatize(word, pos)
                if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                    cleaned_sentence.append(token.lower())

            self.cleaned_content.append(cleaned_sentence)


def download_nltk_libraries():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download()
    print("NLTK download complete.")


def main():
    # download_nltk_libraries()

    analysis = SentimentAnalysis()
    analysis.tokenize_csv()
    analysis.clean_content()

    print(analysis.cleaned_content)
    return 0


if __name__ == '__main__':
    main()
