### Run Sentiment Analysis

To run the sentiment analysis on the provided datasets, the NLTK libraries must be installed. When running the program, it will prompt you for a "y" or "n" to whether or not you need to download them. I personally downloaded all of the available libraries.

The csv files to be parsed should be included in the `data` folder, and listed in the `tweet_files` array at the top of the file. If those files have corresponding files in the `cache` folder, they will be automatically used instead of re-parsing the data. Otherwise, the program will run through parsing the data and then caching it for later use.

The size of this compressed project was too large to submit when including the training set, so I removed it from the `data` folder and kept the cached files instead. The program should work as-is, using the cache. The training data can be downloaded from https://www.kaggle.com/kazanova/sentiment140 and should be saved in the `data` folder, if needed.

---

The `app.py` file is only used for fetching additional tweets, and should not be used when running the sentiment analysis. It requires setting environment variables to allow the Tweepy library access to the Twitter API.

If you have a Twitter development account handy, the API keys supplied with that account can replace the environment variables listed at the top of the file to be able to run it. The environment variables should have corresponding keys with the development account's API keys.
