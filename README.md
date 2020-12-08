### Run Sentiment Analysis
To run the sentiment analysis on the provided datasets, the NLTK libraries must be installed. In `sentiment.py`, the first line in `main()` should be uncommented: `download_nltk_libraries()`. I personally downloaded all of the available libraries.

After this download has been completed, the line can be removed and the app should automatically run afterwards. In the case that it does not, it can be re-run after commenting out the line again. This is only necessary to download the libraries if they are not already present on the machine.

The csv files to be parsed should be included in the `data` folder, and listed in the `tweet_files` array at the top of the file. If those files have corresponding files in the `cache` folder, they will be automatically used instead of re-parsing the data. Otherwise, the program will run through parsing the data and then caching it for later use.

---

The `app.py` file is only used for fetching additional tweets, and should not be used when running the sentiment analysis. It requires setting environment variables to allow the Tweepy library access to the Twitter API.

If you have a Twitter development account handy, the API keys supplied with that account can replace the environment variables listed at the top of the file to be able to run it. The environment variables should have corresponding keys with the development account's API keys.
