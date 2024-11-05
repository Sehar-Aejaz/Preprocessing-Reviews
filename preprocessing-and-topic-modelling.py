# Importing the pandas library to handle data in DataFrame format
import pandas as pd

# Loading the dataset into a DataFrame called 'papers'
papers = pd.read_csv('hospitality data.csv')

# Displaying the first few rows of the dataset for an initial inspection
papers.head()

# Dropping columns that are unnecessary for the analysis
# This step is for removing metadata and columns unrelated to text processing
papers = papers.drop(columns=[
    'Software name', 'Product category', 'Name', 'Designation', 
    'Industry & Size', 'Industry', 'Company Size', 'Usage Duration', 
    'Heading', 'Overall', 'Review Date', 'Alternatives', 
    'Reason for choosing the software', 'Reason for switching to the software', 
    'Switched from', 'Overall Rating', 'Ease of Use', 'Customer Service', 
    'Features', 'Value for Money', 'Likelihood to Recommend', 'Reviewer Source '],
    axis=1)

# Creating a combined summary column that concatenates 'Pros' and 'Cons' text
summ = []
for i in range(len(papers)):
    s = papers.loc[i].at['Pros'] + " " + papers.loc[i].at['Cons']
    summ.append(s)

# Adding the concatenated summary to a new column 'SUMMARY'
papers["SUMMARY"] = summ

# Dropping the individual 'Pros' and 'Cons' columns as they are now redundant
papers = papers.drop(['Pros'], axis=1)
papers = papers.drop(['Cons'], axis=1)

# Importing regex library to process text
import re

# Removing punctuation from the 'SUMMARY' column
papers['SUMMARY'] = papers['SUMMARY'].map(lambda x: re.sub('[,\.!?]', '', x))

# Converting text in 'SUMMARY' column to lowercase
papers['SUMMARY'] = papers['SUMMARY'].map(lambda x: x.lower())

# Displaying the first rows of the processed 'SUMMARY' column
papers['SUMMARY'].head()

import gensim
from gensim.utils import simple_preprocess

# Function to tokenize and preprocess sentences
# 'deacc=True' removes punctuations
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# Applying the function to tokenize data in 'SUMMARY' column
data1 = papers.SUMMARY.values.tolist()
data_words1 = list(sent_to_words(data1))

# Printing first 30 words of the first document as a sample
print(data_words1[:1][0][:30])

# Building bigram and trigram models for grouped phrases
bigram = gensim.models.Phrases(data_words, min_count=1, threshold=1) # higher threshold = fewer phrases
trigram = gensim.models.Phrases(bigram[data_words], threshold=1)

# Optimized way to get sentence bigrams/trigrams
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Loading and extending the list of stopwords with specific irrelevant terms
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'software', 'product', 'issue'])

# Define functions for removing stopwords, applying bigrams, trigrams, and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN']):
    """Perform lemmatization, keeping only words with specified part-of-speech tags"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

import spacy

# Removing stop words
data_words_nostops = remove_stopwords(data_words)

# Forming bigrams from processed words
data_words_bigrams = make_bigrams(data_words_nostops)

# Loading spacy language model and disabling unnecessary components for efficiency
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Applying lemmatization to keep only specific parts of speech
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN'])

# Further stopwords removal post-lemmatization
data_final = remove_stopwords(data_lemmatized)

# Displaying the first lemmatized document for inspection
print(data_lemmatized[:1][0][:30])

import gensim.corpora as corpora

# Creating a dictionary of unique tokens in the final lemmatized data
id2word = corpora.Dictionary(data_final)

# Filtering out extremely rare and frequent words
id2word.filter_extremes(no_below=15, no_above=0.9)

# Creating a corpus (bag of words format) from lemmatized text
texts = data_final
corpus = [id2word.doc2bow(text) for text in texts]

# Displaying the term-document frequency for the first document as a sample
print(corpus[:1][0][:30])


# This block TAKES A LONG TIME TO RUN ----------------------------------------------
# This block sets up a range of values for the number of topics, alpha, and beta, 
# and computes the coherence score for each combination to find the optimal settings.

# Import necessary library for coherence model calculation
from gensim.models import CoherenceModel

# Define function to compute coherence values for various numbers of topics
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for different numbers of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts (lemmatized)
    limit : Maximum number of topics to test

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence scores corresponding to each model
    """
    coherence_values = []  # List to store coherence scores
    model_list = []        # List to store LDA models

    # Iterate over the specified range of topic numbers
    for num_topics in range(start, limit, step):
        # Build LDA model with the current number of topics
        model = gensim.models.LdaMulticore(
            corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=100, random_state=100, chunksize=2000
        )
        model_list.append(model)  # Add model to the list

        # Calculate coherence score for the current model
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())  # Append coherence score

    return model_list, coherence_values  # Return models and their coherence scores


# Import libraries for parameter grid setup
import numpy as np
import tqdm

# Initialize grid for storing results
grid = {}
grid['Validation_Set'] = {}

# Define range for number of topics
min_topics = 5
max_topics = 20
step_size = 1
topics_range = 1

# Define alpha parameter values (topic density per document)
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Define beta parameter values (word density per topic)
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Define corpus validation sets (75% and 100% of data) for testing
num_of_docs = len(corpus)
corpus_sets = [
    gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),  # 75% of the corpus
    corpus  # 100% of the corpus
]
corpus_title = ['75% Corpus', '100% Corpus']  # Labels for each corpus subset

# Dictionary to store model results for each parameter combination
model_results = {
    'Validation_Set': [],
    'Topics': [],
    'Alpha': [],
    'Beta': [],
    'Coherence': []
}

# Set a fixed number of topics (k=5) for this run
k = 5

# Perform grid search over corpus subsets, alpha, and beta values
if 1 == 1:  # Placeholder for conditional check
    pbar = tqdm.tqdm(total=540)  # Initialize progress bar (540 iterations)

    # Iterate over corpus subsets (75% and 100%)
    for i in range(len(corpus_sets)):
        # Iterate over each alpha value
        for a in alpha:
            # Iterate over each beta value
            for b in beta:
                # Compute coherence score for the current combination of corpus, alpha, and beta
                cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, texts=data_lemmatized,
                                              limit=k, start=k, step=1)[1][0]

                # Save results for this combination
                model_results['Validation_Set'].append(corpus_title[i])
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)

                # Update progress bar
                pbar.update(1)

    # Save all results to a CSV file for analysis
    pd.DataFrame(model_results).to_csv('lda5.csv', index=False)
    pbar.close()  # Close progress bar

#-------------------------------------------------------------------------

# Set the number of topics to 5 for the LDA model
num_topics = 5

# Build the LDA model with the specified parameters
lda_model = gensim.models.LdaMulticore(
    corpus=corpus,           # Input the preprocessed corpus
    id2word=id2word,         # Dictionary mapping word IDs to words
    num_topics=num_topics,   # Set the number of topics to 5
    random_state=100,        # Set a random state for reproducibility
    chunksize=2000,          # Number of documents to process in each training chunk
    passes=100,              # Number of passes through the corpus during training
    alpha=0.01,              # Document-topic density hyperparameter, set low for sparser topics
    eta=0.01                 # Word-topic density hyperparameter, controlling word-topic distributions
)

# Import visualization tools for LDA model
import pyLDAvis.gensim as gensimvis
import pyLDAvis

# Prepare the data for visualization
# `prepare` function takes the LDA model, corpus, and dictionary to create visualization-ready data
vis_data = gensimvis.prepare(lda_model, corpus, id2word)

# Display the interactive topic model visualization
pyLDAvis.display(vis_data)


# Calculate the coherence score to assess topic quality
coherence_model_lda = CoherenceModel(
    model=lda_model,            # The trained LDA model
    texts=data_lemmatized,      # Preprocessed text data for coherence evaluation
    dictionary=id2word,         # Dictionary for mapping words to IDs
    coherence='c_v'             # Type of coherence measure ('c_v' is common for topic models)
)

# Retrieve and print the coherence score, which indicates model quality (higher is better)
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Print the keywords for each of the 8 topics to understand the main terms defining each topic
pprint(lda_model.print_topics())

# Transform the corpus into topic distribution format
# Each document is now represented as a topic probability distribution based on the LDA model
doc_lda = lda_model[corpus]
