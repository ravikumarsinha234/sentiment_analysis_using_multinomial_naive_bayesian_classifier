# Sentiment Analysis using Multinomial Naive Bayesian Classifier

A Bayesian classifier is a wonderful thing: given an input, you get a probability for every possible
output. A naive Bayesian classifier makes the simplifying assumption that every dimension of the input
vector is independent.  
We will use a naive Bayesian classifier to classify a tweet using the ”Bag of Words”
approach.  
We've a collection of tweets to airlines that have been labeled ”positive”, ”negative”,
or ”neutral”. We will develop a system that will be able to label the sentiment of a tweet with
about 80% accuracy.  

## Preparation of Data

We have Tweets.csv which is a real data set from Kaggle: https://www.kaggle.com/
datasets/crowdflower/twitter-airline-sentiment  
We also have a program tweet prep data.py that:  
• Reads Tweets.csv using the csv library.  
• Discards tweets where the labeler was not at least 50  
• Ignores all stop words and names of airlines  
• Saves out a list of the 2000 most common words that remain.  
• Splits the remaining data into train tweet.csv and test tweet.csv. About 10% of the
tweets will end up in test tweet.csv.  
• Prints out the 32 most common words.  
If you haven’t already, you will need to install nltk and its English stopwords:  
> pip3 install nltk  
> python3  
Type "help", "copyright", "credits" or "license" for more information.  
  
>import nltk  
>nltk.download(’stopwords’)  
  
You do not need to change tweet prep data.py at all. Run it to create vocablist tweet.pkl,
train tweet.csv, and test tweet.csv.  
Here’s what it should look like when you run it:  
> python3 tweet_prep_data.py  
  
Kept 14404 rows, discarded 236 rows  
Most common 32 words are [’flight’, ’get’, ’cancelled’, ’thanks’, ’service’,
’help’, ’time’, ’customer’, ’im’, ’us’, ’hours’, ’flights’, ’hold’, ’amp’,
’plane’, ’thank’, ’cant’, ’still’, ’one’, ’please’, ’need’, ’would’, ’delayed’,
gate’, ’back’, ’flightled’, ’call’, ’dont’, ’bag’, ’hour’, ’got’, ’late’]  
Wrote 2000 words to vocablist_tweet.pkl.  
(Why is ”flightled” on this list? I have no idea. Real data is sometimes weird.)  

## Using the training data
We will need to complete the program called tweet train.py that  
• Reads in vocablist tweet.pkl.  
• Goes through train tweet.csv row by row, counting the words These counts will be used
to create a word frequency vector for each sentiment.  
• It will also count the tweets for each sentiment, so that it can say things like ”63.3% of all
these tweets are negative.”. These frequencies will act as your priors.  
• Take the log of all the word frequencies and the sentiment frequencies. Save them both to a
single file named frequencies tweet.pkl.  
• Print out the 10 most positive words and the 10 most negative (as determined by the difference
between the Sentiment 0 frequency and the Sentiment 2 frequency).  
Words that don’t appear at all for a sentiment should be treated as if they appeared 0.5 times.
When it runs, it should look something like this:  
> python3 tweet_train.py  
  
Skipped 81 tweets: had no words from vocabulary  
*** Tweets by sentiment ***  
0 (negative): 63.5%  
1 (neutral): 20.5%  
2 (positive): 16.0%  
Positive words:  
thanks thank great love awesome best much good amazing guys  
Negative words:  
flight cancelled hours hold delayed call get flightled hour dont  

## Test with the testing data

You will need to complete the program called test tweet.py that  
• Reads in vocablist tweet.pkl and frequencies tweet.pkl.  
• Goes through test tweet.csv row by row, using Bayesian inference to guess if it is positive,
negative, or neutral.  
• At the end, it should give some statistics on its performance, like accuracy and a confusion
matrix. This should include a baseline of ”How many would the system get right if it ignored
the data and just guessed the most common class?”  
• Besides a guess, the Bayesian Classifier gives us a probability that it is correct. If we discard
the results that it is less sure of, we would expect it’s accuracy to increase. Make a plot
showing this.  
When it runs, it should look like this:  

> python3 tweet_test.py  
  
Would get 63.5% accuracy by guessing "0" every time.
Skipped 9 rows for having none of the common words
1468 lines analyzed, 1162 correct (79.2% accuracy)  
Confusion:  
[[843 58 31]  
[118 159 34]  
[ 42 23 160]]  
