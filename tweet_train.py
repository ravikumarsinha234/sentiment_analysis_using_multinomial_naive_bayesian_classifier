import csv
import util
import pickle as pk
import numpy as np

# What is the probability for unseen words?
DEFAULT_P = 0.5

# Read the word list
## Your code here
vocab_list = pk.load(open("vocablist_tweet.pkl", "rb"))  ## Your code here

# Note how many words are in the vocabulary
veclen = len(vocab_list)  ## Your code here

# Make a dictionary (word->index) for faster lookup
# Convert dictionary for faster lookup
vocab_lookup = {}
for i, word in enumerate(vocab_list):
    vocab_lookup[word] = i

# Create an np array for counting each word
# sums[3][1] will be the total number of vocab_list[3]
# in sentiment 1 tweets
sums = np.zeros((veclen, 3), dtype=np.double)

# Create an array for counting tweets by sentiment
# tweet_counts[0] will be the total number of negative tweets
tweet_counts = np.zeros(3, dtype=int)


# Step through the train.csv file
with open("train_tweet.csv", "r") as f:
    reader = csv.reader(f)

    # Keep track of how many tweets we skipped
    # because they had no words in our vocabulary
    skipped_tweet_count = 0

    for row in reader:

        # Skip rows that don't have two entries
        if len(row) != 2:
            continue

        # Get the tweet and its associated sentiment
        tweet = row[0]
        sentiment = int(row[1])

        # Convert the tweet to a list of words
        wordlist = util.str_to_list(tweet)  ## Your code here (use util.py)

        # Convert the list of words into a word count vector
        word_counts = util.counts_for_wordlist(
            wordlist, vocab_lookup
        )  ## Your code here (use util.py)
        # print(word_counts)
        # print(word_counts.shape)
        # break
        # Skip tweets with no common words
        if word_counts is None:
            skipped_tweet_count += 1
            continue
        # Add the word counts to the sums for the appropriate sentiment
        # (You don't need a loop here)
        ## Your code here
        sums[:, sentiment] = sums[:, sentiment] + word_counts

        # Increment the count of the sentiment
        ## Your code here
        tweet_counts[sentiment] += 1

print(f"Skipped {skipped_tweet_count} tweets: had no words from vocabulary")
# print(sums)
# print(tweet_counts)
# Zeros are draconian
# Replace any zeros in sums with DEFAULT_P
## Your code here
sums[sums == 0] = DEFAULT_P

# From sums, get the total number of counted
# words for each sentiment
# totals = ## Your code here
totals = np.array([sums[:, 0].sum(), sums[:, 1].sum(), sums[:, 2].sum()])
totals.reshape(1, 3)
print(totals)
assert totals.shape == (3,), "totals is an incorrect shape"

# # Compute the word frequencies
# word_frequencies = np.array([sums[:,0]/sums[:,0].sum(),sums[:,1]/sums[:,1].sum(),sums[:,2]/sums[:,2].sum()])## Your code here
# print(word_frequencies.shape)
# word_frequencies = word_frequencies.reshape(2000,3)
word_frequencies = sums / sums.sum(axis=0)
assert np.all(
    np.isclose(word_frequencies.sum(axis=0), np.array([1.0, 1.0, 1.0]))
), "Word frequencies for a sentiment do not sum to one"

# Take the log of the word frequencies
log_word_frequencies = np.log(word_frequencies)  ## Your code here

# # Compute the priors
sentiment_frequencies = tweet_counts / tweet_counts.sum()
assert sentiment_frequencies.shape == (
    3,
), "sentiment_frequencies is an incorrect shape"
assert np.isclose(
    sentiment_frequencies.sum(), 1.0
), "sentiment frequencies do not sum to one"

# Print out the priors
print("*** Tweets by sentiment ***")
for i in range(3):
    print(f"\t{i} ({util.labels[i]}): {sentiment_frequencies[i] * 100.0:.1f}%")

# Compute the log of the priors
log_sentiment_frequencies = np.log(sentiment_frequencies)  ## Your code here

assert log_word_frequencies.shape == (
    2000,
    3,
), "log_word_frequencies is an incorrect shape"
assert log_sentiment_frequencies.shape == (
    3,
), "log_sentiment_frequencies is an incorrect shape"

# Write out the logs of the word and sentiment frequencies in a single pickle file
# Your code here
with open(util.frequencies_path, "wb") as f:
    pk.dump((log_word_frequencies, log_sentiment_frequencies), f)
    # pk.dump(log_sentiment_frequencies, f)

# Just for fun, print out the most positive and most negative words
# by taking the difference between a wols
# rd's frequency in sentiment 0 tweets
# and its frequency in sentiment 2 tweets
happy_angry = word_frequencies[:, 0] - word_frequencies[:, 2]
happiest_to_angriest = np.argsort(happy_angry)
# print(vocab_lookup[0])
print(f"Positive words:{[vocab_list[val] for val in happiest_to_angriest[:10]]}")
print(f"Negative words: {[vocab_list[val] for val in happiest_to_angriest[-1:-11:-1]]}")
# ## Your code here
