import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sb
import numpy as np
import re
import json

###Dataset Initialization
sub = "worldpolitics"

with open("credentials.json") as f:
    params = json.load(f)

reddit = praw.Reddit(client_id=params["client_id"],
                     client_secret=params["client_secret"],
                     user_agent="suicidalcrocodile's script")

allposts = list()

for post in reddit.subreddit(sub).hot(limit=1000):
  allposts.append([None if post.author is None else post.author.name, 
                   post.name, 
                   post.title, 
                   post.selftext, 
                   post.num_comments, 
                   post.score,
                   post.created_utc	])

dataset = pd.DataFrame(allposts, columns=["Author", 
                                          "Name",
                                          "Title", 
                                          "Text", 
                                          "Num_Comments", 
                                          "Score",
                                          "CreatedUTC"])
dataset['CreatedUTC'] = pd.to_datetime(dataset['CreatedUTC'],unit='s')

###ddataset date distribution
dataset.groupby(pd.Grouper(key="CreatedUTC", freq="1d")).size().plot(kind='bar', rot=0)

###Initializing useful variables
titles_values = dataset["Title"].values.astype("str")

def find_only_whole_word(search_string, input_string):
  ###from https://stackoverflow.com/a/52304664
  # Create a raw string with word boundaries from the user's input_string
  raw_search_string = r"\b" + search_string + r"\b"

  match_output = re.search(raw_search_string, input_string)
  ##As noted by @OmPrakesh, if you want to ignore case, uncomment
  ##the next two lines
  #match_output = re.search(raw_search_string, input_string, 
  #                         flags=re.IGNORECASE)

  no_match_was_found = ( match_output is None )
  if no_match_was_found:
    return False
  else:
    return True
  
###Building titles bag of words
cv = CountVectorizer(stop_words="english")
titles_bow = cv.fit_transform(dataset["Title"].values)
sum_titles_words= titles_bow.sum(axis=0)
sum_map_titles_words = [(word, sum_titles_words[0, i]) for word, i in cv.vocabulary_.items()]
sum_map_titles_words = sorted(sum_map_titles_words, key = lambda x: x[1], reverse=True)
common_words_ds = pd.DataFrame(sum_map_titles_words[:20], columns=["Word", "Frequency"])
common_words_ds.plot.bar(x="Word", y="Frequency", rot=0)
  
##Visualizing titles wordcloud
stopwords = set(STOPWORDS)
titles = dataset["Title"].values
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=1024, height=1024).generate(" ".join(titles))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

##Visualizing Text wordcloud
stopwords = set(STOPWORDS)
titles = dataset["Text"].values
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=1024, height=1024).generate(" ".join(titles))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

###Building titles Bigram bag of words
cv = CountVectorizer(ngram_range=(2,2), stop_words="english")
titles_bow = cv.fit_transform(dataset["Title"].values)
sum_titles_bigrams= titles_bow.sum(axis=0)
sum_map_titles_bigrams = [(word, sum_titles_bigrams[0, i]) for word, i in cv.vocabulary_.items()]
sum_map_titles_bigrams = sorted(sum_map_titles_bigrams, key = lambda x: x[1], reverse=True)
common_bigrams_ds = pd.DataFrame(sum_map_titles_bigrams[:20], columns=["Bigram", "Frequency"])
common_bigrams_ds.plot.bar(x="Bigram", y="Frequency", rot=0)

###Adding PopTerms Columns
most_frequent_terms = list(x[0] for x in sum_map_titles_words[:3])
popular_terms_colls = list()
for title in np.char.lower(titles_values):
  title_popular_terms = np.empty(len(most_frequent_terms),dtype=titles_values.dtype)
  pop_terms_count = 0
  for frequent_term in most_frequent_terms:
    if find_only_whole_word(frequent_term, title):
      title_popular_terms[pop_terms_count] = frequent_term
      pop_terms_count+=1
  popular_terms_colls.append(title_popular_terms)

popular_terms_ds = pd.DataFrame(popular_terms_colls, columns=["PopTerm{0}".format(str(i+1)) for i in range(len(most_frequent_terms))]).astype(str)
popular_terms_ds.replace(r'^\s*$', np.nan, regex=True, inplace=True)
dataset_with_popterms=dataset.join(popular_terms_ds)

###Visualizing Score distribution among popular terms

##PopTerm1 vs Score scatter plot
ax = sb.catplot(x="PopTerm1", y="Score", data=dataset_with_popterms);
ax.set(yscale="log") #setting to log scale if necessary

##PopTerm1 vs Mean Score bar plot
sb.barplot(x="PopTerm1", y="Score", data=dataset_with_popterms, capsize=.2)

##PopTerm1 vs Score box plot
ax=sb.boxplot(x="PopTerm1", y="Score", data=dataset_with_popterms, showfliers=False)
#showFliers can be set to True to show outliers
ax.set(yscale="log")#setting to log scale if necessary

###Visualizing Num_Comment distribution among popular terms
##Num_Comments vs score
ax=sb.lineplot(x="Score", y="Num_Comments", data=dataset_with_popterms, hue="PopTerm1")
ax.set(yscale="log",xscale="log")#setting to log scale if necessary

##Mean Num_Comments vs PopTerm1
sb.barplot(x="PopTerm1", y="Num_Comments", data=dataset_with_popterms, capsize=.2)

##Num_Comments vs PopTerm1
sb.boxplot(x="PopTerm1", y="Num_Comments", data=dataset_with_popterms, showfliers=False)
#showFliers can be set to True to show outliers

###Visualizing Authors distribution among popular terms
print("Unique authors:{0}".format(dataset_with_popterms["Author"].nunique()))
##Plot the 10 most frequent authors 
ax=dataset_with_popterms["Author"].value_counts(sort=True, normalize=False)[:10].plot(kind="bar", rot=0)
ax.set_ylabel("Post Count")
ax.set_xlabel("Author")

##Plot the 10 most frequent authors of popular terms
ax=dataset_with_popterms[dataset_with_popterms["PopTerm1"].notna()]["Author"].value_counts()[:10].plot(kind="bar", rot=0)
ax.set_ylabel("Post Count")
ax.set_xlabel("Author")

##Plot the 10 most frequent authors of each popular term
for pop_term in most_frequent_terms:
  plt.figure()
  dataset_with_popterms[dataset_with_popterms["PopTerm1"]==pop_term]["Author"].value_counts()[:3].plot(kind="bar",rot=0, title="Frenquent authors of posts about {0}".format(pop_term))

##Plot the 10 most frequent authors of each popular term in one figure
sb.countplot(x="Author", data=dataset_with_popterms,order=dataset_with_popterms["Author"].value_counts().iloc[:10].index, hue="PopTerm1")

###Visualizing Authors distribution among popular terms by mean value
##Authors vs Mean Score
dataset_with_popterms.groupby("Author")["Score"].mean().sort_values(ascending=False)[:3].plot(kind="bar", rot=0)
##Authors vs Mean Score among posts about popular terms
dataset_with_popterms[dataset_with_popterms["PopTerm1"].notna()].groupby("Author")["Score"].mean().sort_values(ascending=False)[:3].plot(kind="bar", rot=0)