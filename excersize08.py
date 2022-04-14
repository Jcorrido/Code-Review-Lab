import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Remove stopwords, non alphabetical words and characters, punctuation, smaller than three characters
def cleantext(text, stopwords, wordlength):

    for words in text:
        cleanwords = []
        for word in words:
            newword1 = ''.join(ch for ch in word if ch.isalpha())       # Gets rid of non alpha characters
            newword2 = newword1.encode("ascii", "ignore")               # Gets rid of accented characters
            newword = newword2.decode()
            if newword == "":
                break
            else:
                if len(newword) > wordlength and len(newword) < 15:     # Removes word less than wordlength and urls
                    if newword not in stopwords:                        # Removes stop words
                        cleanwords.append(newword)
        output.append(cleanwords)


#### Prep work ####
# Covid related list
num_articles = 769
covid_related = ["covid", "coronavirus", "vaccine", "vaccination", "antibody", "moderna", "pfizer", "johnson"]
# Create pd dataframe from everything
news = pd.read_csv("News_2021.csv")

# Create pd column of text from description and content
news["description"] = news["description"].fillna(" ")
news["content"] = news["content"].fillna(" ")

news["text"] = news["description"] + news["content"]
news["text"] = news["text"].str.lower()

# Create list from stopwords
stopwords = open('stopwords_en.txt').read().splitlines()

# Create list from text
text = []
old_row = []
for row in news["text"]:
    words = row.split()
    text.append(words)

# Clean text
output = []
cleantext(text, stopwords, 3)

#### Perform Output ####
# Find the 20 most common words
unique = {}
for words in output:
    for word in words:
        try:
           unique[word] = unique[word] + 1
        except:
            unique[word] = 1

unique2 = unique.copy()
top20 = []
count = 0
while count < 20:
    top20.append(max(unique2, key=unique2.get))
    unique2.pop(max(unique2, key=unique2.get))
    count += 1
print("Here are the 20 most common words:\n", top20)

# Covid related articles
covid_related_articles = []
noncovid_related_articles = []
output2 = output.copy()

for row in output2:
    check = any(item in row for item in covid_related)
    if check is True:
        covid_related_articles.append(row)
    else:
        noncovid_related_articles.append(row)

covid_percentage = ((len(covid_related_articles)+1)/num_articles)*100
print("percentage of articles related to covid:", covid_percentage)

# Sentiment of articles not related to covid-19
analyzer = SentimentIntensityAnalyzer()

noncovid_list_text = []
for row in noncovid_related_articles:
    text1 = ' '.join(row)
    noncovid_list_text.append(text1)
noncovid_related_text = ' '.join(noncovid_list_text)
vad_sentiment = analyzer.polarity_scores(noncovid_related_text)

pos_noncovid = vad_sentiment['pos']
neg_noncovid = vad_sentiment['neg']
neu_noncovid = vad_sentiment['neu']

print('\n\n--- Sentiment analysis')

print('\nNonCovid Related')
print('It is positive for', '{:.1%}'.format(pos_noncovid))
print('      negative for', '{:.1%}'.format(neg_noncovid))
print('      neutral for', '{:.1%}'.format(neu_noncovid))

# Sentiment analysis for all articles
output_list_text = []
for row in output:
    text1 = ' '.join(row)
    output_list_text.append(text1)
output_related_text = ' '.join(output_list_text)
vad_sentiment = analyzer.polarity_scores(output_related_text)

pos_output = vad_sentiment['pos']
neg_output = vad_sentiment['neg']
neu_output = vad_sentiment['neu']

print('\nAll articles')
print('It is positive for', '{:.1%}'.format(pos_output))
print('      negative for', '{:.1%}'.format(neg_output))
print('      neutral for', '{:.1%}'.format(neu_output))

# WordClouds
print('\n\n--- Generating the wordcloud')

# Defining the wordcloud parameters
wc1 = WordCloud(background_color="white", max_words=2000, stopwords=set(stopwords))
wc2 = WordCloud(background_color="white", max_words=2000, stopwords=set(stopwords))

# Generate word cloud
wc1.generate(noncovid_related_text)
wc2.generate(output_related_text)

# Store to file
wc1.to_file('non-covid.png')
wc2.to_file('output.png')

# Show the cloud
plt.imshow(wc1)
plt.imshow(wc2)
plt.axis('off')
plt.show()

print('\n--- End of processing')

