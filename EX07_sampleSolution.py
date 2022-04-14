# -*- coding: utf-8 -*-
"""

This code reads 2 text files of comments from the Pros and Con on the question:
    Should recreational marijuana be legal?
It finds the most common words and the most common bigrams.

"""

# importing the required libraries
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# this function is to clean the text
def txt_clean(word_list, min_len, stopwords_list):
    """
    Performs a basic cleaning to a list of words.

    :param word_list: list of words
    :type: list
    :param min_len: minimum length of a word to be acceptable
    :type: integer
    :param stopwords_list: list of stopwords
    :type: list
    :return: clean_words list of clean words
    :type: lists

    """
    clean_words = []
    for line in word_list:
        parts = line.strip().split() # this is creating a list with elements being words (tokens) for the line
        for word in parts:
            word_l = word.lower().strip()
            if word_l.isalpha():
                if len(word_l) > min_len:
                    if word_l not in stopwords_list:
                        clean_words.append(word_l)
                    else:
                        continue

    return clean_words


# opening the files
Pro_file = open("pro_ marijuana_raw.txt","r", encoding='utf8')
Con_file = open("con_ marijuana_raw.txt","r", encoding='utf8')
stopwords_file = open("stopwords_en.txt","r", encoding='utf8')

# initializing the lists
stopwords_lst = []
Pro_lines = []
Con_lines = []
Pro_words = []
Con_words = []

# setting the minimum length for the words
word_min_len = 2

# loading the stopwords in a list
for s_word in stopwords_file:
    stopwords_lst.append( s_word.strip() )
stopwords_lst.extend(['marijuana', 'cannabis', 'drug', 'use', 'legal', 'year', 'people', 'legalization'])
# the word marijuana has been added to the stopwords, being reasonably in most of the comments

# reading the files
for p_line in Pro_file:
    if p_line != '\n' and len(p_line) > 30:
        Pro_lines.append(p_line)

for a_line in Con_file:
    if a_line != '\n' and len(a_line) > 30:
        Con_lines.append(a_line)

# cleaning the input texts
Pro_words = txt_clean(Pro_lines, word_min_len, stopwords_lst)
Con_words = txt_clean(Con_lines, word_min_len, stopwords_lst)

# calculating the top values
top_10_Pro = Counter(Pro_words).most_common(10)
top_10_Con = Counter(Con_words).most_common(10)

# printing the top values
print ('\n\n--- Top words (count)')
print ("\nTop 10 Pro Words:")
for pair in top_10_Pro:
    print (pair[0], "(" + str(pair[1]) + ")")

print ("Top 10 Con Words:")
for pair in top_10_Con:
    print (pair[0], "(" + str(pair[1]) + ")")

##### BIGRAMS #####

# initializing the lists that will contain the bigrams
Pro_bigrams = []
Con_bigrams = []

# calculating the bigrams

for i in range(len(Pro_words)):
    try:
        Pro_bigrams.append( Pro_words[i] + "_" + Pro_words[i+1] )
    except:
        pass # reached end of list

for i in range(len(Con_words)):
    try:
        Con_bigrams.append( Con_words[i] + "_" + Con_words[i+1] )
    except:
        pass # reached end of list

# calculating the top values
common_Pro_bigrams = Counter(Pro_bigrams).most_common(5)
common_Con_bigrams = Counter(Con_bigrams).most_common(5)

# printing the top values
print ('\n\n5 most common bigrams:')
print ('\nPro bigrams:')
for i in range(5):
    pro_bigram = common_Pro_bigrams[i][0].split('_')
    print(pro_bigram[0], pro_bigram[1])

print ("\nCon bigrams:")
for i in range(5):
    con_bigram = common_Con_bigrams[i][0].split('_')
    print (con_bigram[0], con_bigram[1])


##### Sentiment Analysis #####
# calculating the sentiment using vader library
analyzer = SentimentIntensityAnalyzer()

# processing the pro
# vader needs strings as input. Transforming the list into string
clean_text_str_pro = ' '.join(Pro_words)
vad_sentiment = analyzer.polarity_scores(clean_text_str_pro)

pos_pro = vad_sentiment ['pos']
neg_pro = vad_sentiment ['neg']
neu_pro = vad_sentiment ['neu']

# processing the con
# vader needs strings as input. Transforming the list into string
clean_text_str_con = ' '.join(Con_words)
vad_sentiment = analyzer.polarity_scores(clean_text_str_con)

pos_con = vad_sentiment ['pos']
neg_con = vad_sentiment ['neg']
neu_con = vad_sentiment ['neu']

# printing the pro
print ('\n\n--- Sentiment analysis')

print ('\nPRO marijuana')
print ('It is positive for', '{:.1%}'.format(pos_pro))
print ('      negative for', '{:.1%}'.format(neg_pro))
print ('      neutral for', '{:.1%}'.format(neu_pro))


# printing the con
print ('\nCON marijuana')
print ('It is positive for', '{:.1%}'.format(pos_con))
print ('      negative for', '{:.1%}'.format(neg_con))
print ('      neutral for', '{:.1%}'.format(neu_con))

##### Word cloud #####
print ('\n\n--- Generating the wordcloud')

# Transforming the lists of words into strings
Pro_words_string = ' '.join(Pro_words)
Con_words_string = ' '.join(Con_words)

# Defining the wordcloud parameters
wc = WordCloud(background_color = "white", max_words = 2000,
               stopwords = set(stopwords_lst))

# Generate word cloud
wc.generate(Pro_words_string)

# Store to file
#wc.to_file('Pro.png')

# Show the cloud
plt.imshow(wc)
plt.axis('off')
plt.show()

print ('\n--- End of processing')
