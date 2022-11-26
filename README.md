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
