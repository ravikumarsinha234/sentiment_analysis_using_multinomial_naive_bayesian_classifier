import csv
import pickle as pk
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import util

# Read in the word listprint(len(gt_sentiments))

## Your code here
vocab_list = pk.load(open("vocablist_tweet.pkl", "rb"))  ## Your code here

# Convert dictionary (str -> index) for faster lookup
vocab_lookup = {}
for i, word in enumerate(vocab_list):
    vocab_lookup[word] = i

## Your code here

# Read in the log of the word and sentiment frequencies
log_word_freqencies = pk.load(open(util.frequencies_path, "rb"))[
    0
].T  ## Your code here (Transpose what you read)
log_sentiment_frequencies = pk.load(open(util.frequencies_path, "rb"))[
    1
]  ## Your code here
assert log_word_freqencies.shape == (3, 2000), "log_word_frequecies is the wrong shape"
assert log_sentiment_frequencies.shape == (
    3,
), "log_sentiment_frequencies is the wrong shape"

# We should compare ourselves to some baseline
most_common_sentiment = np.argmax(log_sentiment_frequencies)  ## Your code here
freq_of_most_common = np.exp(
    log_sentiment_frequencies[np.argmax(log_sentiment_frequencies)]
)  ## Your code here
print(
    f'Would get {freq_of_most_common * 100.0:.1f}% accuracy by guessing "{most_common_sentiment}" every time.'
)

# Gather ground truth and predictions
gt_sentiments = []
predicted_sentiments = []
prediction_confidences = []

# Step through the test.csv file
with open("test_tweet.csv", "r") as f:
    reader = csv.reader(f)

    skipped_tweet_count = 0
    for row in reader:

        # Check to see if the row has two entries
        if len(row) != 2:
            continue

        # Get the tweet and its ground truth sentiment
        tweet = row[0]  ## Your code
        sentiment = int(row[1])  ## Your code (it should be an int)

        # Convert the tweet to a wordlist
        wordlist = util.str_to_list(tweet)  ## Your code here (use util.py)

        # Conver the wordlist to a word_counts vector
        word_counts = util.counts_for_wordlist(
            wordlist, vocab_lookup
        )  ## Your code here (use util.py)

        # Did this tweet have no common words?
        if word_counts is None:
            skipped_tweet_count += 1
            continue

        # Compute the log likelihoods for all three sentiments
        log_likelihoods = np.dot(
            log_word_freqencies, word_counts
        )  ## Your code here (refer to the lecture) (It is a matrix times a vector)
        assert log_likelihoods.shape == (3,), "log_likelihoods is the wrong shape"

        # Add the log priors to the log_likelihoods to get the log_posteriors (unnormalized)
        log_posteriors = log_likelihoods + log_sentiment_frequencies  ## Your code here

        # Get a prediction (0, 1 or 2)
        prediction = np.argmax(log_posteriors)  ## Your code here

        # Move posterior out of "log space"
        unnormalized_posteriors = np.exp(log_posteriors - np.max(log_posteriors))

        # They must add up to 1, so normalize them
        normalized_posteriors = unnormalized_posteriors / np.sum(
            unnormalized_posteriors
        )
        # print(normalized_posteriors)
        # Get your confidence in the prediction
        confidence = normalized_posteriors[prediction]  ## Your code here
        assert (
            confidence > 0.33
        ), "Confidence lower than 33 percent, there are only three sentiments"
        assert confidence <= 1.0, "Confidence is more than 100 percent."

        # Store the result in lists
        gt_sentiments.append(sentiment)
        predicted_sentiments.append(prediction)
        prediction_confidences.append(confidence)

print(f"Skipped {skipped_tweet_count} rows for having none of the common words")

# Convert gathered data into numpy arrays
gt = np.array(gt_sentiments)
predictions = np.array(predicted_sentiments)
confidence = np.array(prediction_confidences)

# Show some basic statistics
tweet_count = len(gt_sentiments)  ## Your code
correct_count = np.sum(gt == predictions)  ## Your code
print(
    f"{tweet_count} lines analyzed, {correct_count} correct ({100.0 * correct_count/tweet_count:.1f}% accuracy)"
)
cm = confusion_matrix(gt, predictions)  ## Your code
print(f"Confusion: \n{cm}")

# Save out a confusion matrix plot
fig, ax = plt.subplots()
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=util.labels)
cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
fig.savefig("confusion_tweet.png")
print('Wrote confusion matrix as "confusion_tweet.png"')

# Plot how many you get right vs confidence thresholds
steps = 32
thresholds = np.linspace(0.33, 1.0, steps)
correct_ratio = np.zeros(steps)
confident_ratio = np.zeros(steps)

for i in range(steps):
    threshold = thresholds[i]
    if np.sum(confidence > threshold) != 0:
        correct_ratio[i] = np.sum(
            (gt == predictions) & (confidence > threshold)
        ) / np.sum(confidence > threshold)
    else:
        correct_ratio[i] = 1
    confident_ratio[i] = np.sum(confidence > threshold) / len(confidence)
    ## Your code here
    # correct_ratio[i] = ## Your code
    # confident_ratio[i] = ## Your code

# Make a plot
fig, ax = plt.subplots()
ax.set_title("Confidence and Accuracy Are Correlated")
ax.plot(
    thresholds, correct_ratio, "blue", linewidth=0.8, label="Accuracy Above Threshod"
)
ax.set_xlabel("Confidence Threshold")
ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100.0:.0f}%")
ax.hlines(
    freq_of_most_common,
    0.33,
    1,
    "blue",
    linestyle="dashed",
    linewidth=0.8,
    label=f"Accuracy Guessing {most_common_sentiment}",
)
ax.plot(
    thresholds,
    confident_ratio,
    "r",
    linestyle="dashed",
    linewidth=0.8,
    label="Tweets scoring above threshold",
)
ax.legend()
fig.savefig("confidence_tweet.png")
