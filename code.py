# Import necessary libraries for text processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import glob
import os

# Specify the path to the file(s) to load and join
joined_files = os.path.join("", "hospitality data.csv")

# Return a list of all files matching the specified path pattern
joined_list = glob.glob(joined_files)

# Read and concatenate all files in the list into a single DataFrame
df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
print(df)  # Display the contents of the DataFrame

# Install textblob for sentiment analysis and text processing
import sys
!{sys.executable} -m pip install textblob

# Import TextBlob for text analysis
from textblob import TextBlob

# Download NLTK resources for text processing
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

# Load the English stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Import autocorrect library for correcting misspelled words
from autocorrect import spell

# Initialize a lemmatizer to reduce words to their root forms
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize a stemmer to reduce words to their base forms
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Helper function to map NLTK POS tags to WordNet tags for lemmatization
from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

# Preprocess 'Pros' and 'Cons' columns in the DataFrame
df['Pros_pre'] = df['Pros_pre'].astype(str)
df['Cons_pre'] = df['Cons_pre'].astype(str)
for j in range(len(df)):
    # Tokenize 'Cons' text for the current row
    word_tokens = word_tokenize(df.loc[j].at["Cons"])
    
    # Remove stopwords from the tokenized words
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    
    # Correct spelling errors in filtered words
    for i in filtered_sentence:
        i = spell(i)
    
    # Convert words to lowercase
    for i in filtered_sentence:
        i = i.lower()
    
    # Apply stemming to each word
    stem_w = []
    for w in filtered_sentence:
        stem_w.append(ps.stem(w))
    
    # Reconstruct sentence from stemmed words
    sentence = ' '.join(stem_w)
    
    # Tokenize sentence again for POS tagging
    senP = word_tokenize(sentence)
    nltk.pos_tag(senP)
    
    # Apply lemmatization with POS tagging
    pre_lem = []
    for i in range(len(senP)):
        wor = word_tokenize(senP[i])
        try:
            pre_lem.append(lemmatizer.lemmatize(senP[i], get_wordnet_pos(nltk.pos_tag(wor)[0][1])))
        except:
            pass

    # Filter out words based on POS (e.g., only include nouns)
    post_pos = []
    for i in range(len(pre_lem)):
        wor = word_tokenize(pre_lem[i])
        try:
            a = get_wordnet_pos(nltk.pos_tag(wor)[0][1])
        except:
            pass
        else:
            if a == 'n':  # Only keep nouns
                post_pos.append(pre_lem[i])

    # Join processed words into a final string and save it back to DataFrame
    d = " ".join(post_pos)
    df.at[j, 'Cons_pre'] = d

# Load additional data for further processing
joined_files = os.path.join("", "prepro_data_hospitality.csv")
joined_list = glob.glob(joined_files)
df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
print(df)

# Save the processed DataFrame to a new CSV file
df.to_csv('prelempro_data_hospitality.csv')
