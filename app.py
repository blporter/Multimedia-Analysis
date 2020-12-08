__author__ = 'Brooke Porter'
__class__ = 'CSC 664 Multimedia Systems (Undergraduate)'
__school__ = 'San Francisco State University'
__professor__ = 'Dr. Rahul Singh'
__file__ = 'app.py'

"""
This file is used for parsing tweets from the Twitter API via the Tweepy library.
It requires adding the necessary environment keys to your configuration.
We first authenticate Tweepy using the environment keys, then instruct it to fetch
tweets for 15 minutes, saving them into a relevant csv file. This works with the pull.yaml
file in the workflows folder to instruct Github to run this file every 30 minutes automatically,
but that is controlled by the repository.
"""

import datetime
import logging
import csv
import tweepy
import os
import time

# Environment keys necessary to access the Twitter API.
consumer_key = os.environ["TWITTER_KEY"]
consumer_secret = os.environ["TWITTER_SECRET"]
access_token = os.environ["TWITTER_ACCESS_TOKEN"]
access_token_secret = os.environ["TWITTER_ACCESS_TOKEN_SECRET"]

fieldnames = ['user_id', 'stat_id', 'creation', 'tweet_body', 'name']
csv_path = "data/tweets8.csv"
file_exists = os.path.isfile(csv_path)


class StreamListener(tweepy.StreamListener):
    def __init__(self, end_time):
        super(StreamListener, self).__init__()
        self.end_time = end_time

    def on_status(self, status):
        """
        This function is called by the Tweepy StreamListener to obtain tweets. We pass the required end_time
        to the above init function, which is then used in this function to determine when to stop looping.
        The parsed tweets are then saved to a csv file, specified by the csv_path global variable.
        :param status: The tweet data fetched by Tweepy, used to access the parsed stream.
        :return: The boolean value to determine whether to stop parsing.
        """
        if time.time() > self.end_time:
            return False

        if status.lang == 'en' and 'RT'.upper() not in status.text:
            stat = status.text
            stat = stat.replace('\n', '')
            stat = stat.replace('\t', '')

            user_id = status.user.id_str
            stat_id = status.id_str
            create = str(status.created_at)
            name = status.user.screen_name

            with open(csv_path, 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({
                    "user_id": user_id,
                    "stat_id": stat_id,
                    "creation": create,
                    "tweet_body": stat,
                    "name": name
                })
                csv_file.close()

    def on_error(self, status_code):
        """
        This function is used to detect whether, given that an error occurred, an error is related
        to Twitter's rate limit on stream data.
        :param status_code: The status code of the error which occurred
        :return: A boolean value indicating whether the error is related to the Twitter API's rate limit.
        """
        print("error fetching tweets with status code ", status_code)
        if status_code == 420:
            date = "Error code 420 at:" + str(datetime.datetime.now())
            logging.info(date)
        return False


def write_header():
    """
    If the header doesn't already exist for the csv_path file, then it needs to be created before the file
    can be written to.
    """
    with open(csv_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames)
        # file doesn't exist yet, write a header
        if not file_exists:
            writer.writeheader()


def app():
    # Tweepy needs to be given the Twitter key and secret obtained from a developer account to authenticate.
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.access_token = access_token
    auth.access_token_secret = access_token_secret
    api = tweepy.API(auth)

    print("Getting tweets")

    # Should only run for 15 minutes at a time
    end_time = time.time() + 60 * 15
    listener = StreamListener(end_time=end_time)
    # Write the header to the csv file if it doesn't exist already
    write_header()
    tweepy.Stream(api.auth, listener=listener).sample()

    print("Done fetching tweets, exiting.")
    return 0


if __name__ == '__main__':
    app()
