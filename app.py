import datetime
import logging
import csv
import tweepy
import os
import time

consumer_key = os.environ["TWITTER_KEY"]
consumer_secret = os.environ["TWITTER_SECRET"]
access_token = os.environ["TWITTER_ACCESS_TOKEN"]
access_token_secret = os.environ["TWITTER_ACCESS_TOKEN_SECRET"]

fieldnames = ['user_id', 'stat_id', 'creation', 'tweet_body', 'name']
csv_path = "data/tweets4.csv"
file_exists = os.path.isfile(csv_path)


class StreamListener(tweepy.StreamListener):
    def __init__(self, end_time):
        super(StreamListener, self).__init__()
        self.end_time = end_time

    def on_status(self, status):
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
        print("error fetching tweets with status code ", status_code)
        if status_code == 420:
            date = "Error code 420 at:" + str(datetime.datetime.now())
            logging.info(date)
        return False


def write_header():
    with open(csv_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames)
        # file doesn't exist yet, write a header
        if not file_exists:
            writer.writeheader()


def app():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.access_token = access_token
    auth.access_token_secret = access_token_secret
    api = tweepy.API(auth)

    print("Getting tweets")

    # Should only run for 15 minutes at a time
    end_time = time.time() + 60 * 15
    listener = StreamListener(end_time=end_time)
    # write the header to the csv if it doesnt have it already
    write_header()
    tweepy.Stream(api.auth, listener=listener).sample()

    print("Done fetching tweets, exiting.")
    return 0


if __name__ == '__main__':
    app()
