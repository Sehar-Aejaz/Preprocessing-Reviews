# Extract document-topic distributions for the first half of the corpus
document_key1 = list(papers.index)  # Get the index of documents for referencing
document_topic1 = {}  # Dictionary to store topic distributions for each document

# Loop through the first half of the corpus
for doc_id in range(int(len(corpus) / 2)):
    docbok = corpus[doc_id]  # Get the document's bag-of-words representation
    doc_topics = lda_model.get_document_topics(docbok, 0)  # Get topic distribution for this document
    tmp = []  # Temporary list to store probabilities of each topic for this document

    # Store topic probabilities
    for topic_id, topic_prob in doc_topics:
        tmp.append(topic_prob)
    document_topic1[document_key1[doc_id]] = tmp  # Add to the main dictionary

# Convert dictionary of document-topic distributions to a DataFrame
dfp = pd.DataFrame.from_dict(document_topic1, orient='index')
topic_column_names = ['topic_' + str(i) for i in range(5)]  # Define column names for each topic
dfp.columns = topic_column_names
dfp['SUMMARY'] = papers['Pros_lem']  # Add a summary column with processed text

# Display the first few rows of the DataFrame as HTML for visual inspection
from IPython.display import display, HTML
display(HTML(dfp.head(5).to_html()))

# Repeat the process for the second half of the corpus
document_key2 = list(papers.index)  # Get index for second half
document_topic2 = {}  # Dictionary to store topic distributions

# Loop through the second half of the corpus
for doc_id in range(int(len(corpus) / 2)):
    docbok = corpus[int(len(corpus) / 2) + doc_id]  # Get document
    doc_topics = lda_model.get_document_topics(docbok, 0)  # Get topic distribution
    tmp = []

    # Store topic probabilities
    for topic_id, topic_prob in doc_topics:
        tmp.append(topic_prob)
    document_topic2[document_key2[doc_id]] = tmp

# Convert to DataFrame
dfc = pd.DataFrame.from_dict(document_topic2, orient='index')
dfc.columns = topic_column_names  # Define columns
dfc['SUMMARY'] = papers['Cons_lem']  # Add summary column with processed text

# Display the first few rows of the DataFrame
display(HTML(dfc.head(5).to_html()))

# Variables for positive and negative sentiment data
x = dfp
y = dfc

# Initialize lists to store filtered topic data for positive sentiments
topic1 = []
prospen1 = []
topics_p = []
prospen_p = []

# Loop through dfp to extract and filter topic probabilities > 0.1
for i in range(len(dfp)):
    a = []
    b = []
    for j in range(5):
        n = "topic_" + str(j)
        if dfp.loc[i].at[n] > 0.1:  # Filter topics with >0.1 probability
            a.append(dfp.loc[i].at[n])
            b.append(n)
    topics_p.append(a)  # Store probabilities for each document
    prospen_p.append(b)  # Store topic names

    # Get the highest probability topic
    a.sort()
    try:
        d = b[a.index(a[-1])]
    except:
        d = ''  # Handle cases where there's no valid topic

    if len(a) == 0:
        a.append('')
        b.append('')

    prospen1.append(a[-1])  # Add highest probability
    topic1.append(d)        # Add corresponding topic

# Repeat for negative sentiments in dfc
topic2 = []
prospen2 = []
topics_c = []
prospen_c = []

# Loop through dfc to extract and filter topic probabilities > 0.1
for i in range(len(dfc)):
    a = []
    b = []
    for j in range(5):
        n = "topic_" + str(j)
        if dfc.loc[i].at[n] > 0.1:  # Filter topics with >0.1 probability
            a.append(dfc.loc[i].at[n])
            b.append(n)
    topics_c.append(a)  # Store probabilities for each document
    prospen_c.append(b)  # Store topic names

    # Get the highest probability topic
    a.sort()
    try:
        d = b[a.index(a[-1])]
    except:
        d = ''  # Handle cases with no valid topic

    if len(a) == 0:
        a.append('')
        b.append('')

    prospen2.append(a[-1])  # Add highest probability
    topic2.append(d)        # Add corresponding topic


# Add the identified highest probability topic for positive sentiments to dfp
dfp['topic'] = topic1
# Add the corresponding propensity values for positive sentiments to dfp
dfp['prospensity'] = prospen1

# Add the identified highest probability topic for negative sentiments to dfc
dfc['topic'] = topic2
# Add the corresponding propensity values for negative sentiments to dfc
dfc['prospensity'] = prospen2

# Display the first few rows of the positive sentiment DataFrame dfp
display(dfp.head())

# Display the first few rows of the negative sentiment DataFrame dfc
display(dfc.head())

# Custom CSS styling for output display in Jupyter Notebook
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: left;
    vertical-align: left;
}
</style>
""")

# Enable inline plotting for Matplotlib
%matplotlib inline  
import matplotlib.pyplot as plt

# Set the default figure size for plots
plt.rcParams["figure.figsize"] = (20, 4)

# Count the frequency of each topic in dfp by grouping the data
topic_frequency = dfp.iloc[:, :2].groupby('topic').count()

# Create a bar plot to visualize the frequency of topics in Pros section
ax = topic_frequency.plot.bar(legend=False)
plt.title("Frequency of Topics", size=12)

# Customize the x-axis ticks for better readability
ax.tick_params(axis='x', which='minor', labelsize='small', labelcolor='m', rotation=30)

# Annotate each bar with its height value (frequency)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

# Show the plot
plt.show()

# Enable inline plotting for Matplotlib, allowing plots to be displayed directly in the notebook
%matplotlib inline  

# Import the pyplot module from Matplotlib for creating visualizations
import matplotlib.pyplot as plt

# Set the default figure size for plots to 20 inches wide and 4 inches tall
plt.rcParams["figure.figsize"] = (20, 4)

# Count the frequency of each topic in the dfc DataFrame by grouping the data by the 'topic' column
topic_frequency = dfc.iloc[:, :2].groupby('topic').count()

# Create a bar plot to visualize the frequency of each topic for the Cons section
ax = topic_frequency.plot.bar(legend=False)

# Set the title of the plot to "Frequency of Topics" and specify the font size
plt.title("Frequency of Topics", size=12)

# Customize the appearance of the x-axis ticks
# - Set minor ticks to be small in size and colored in magenta
# - Rotate the tick labels by 30 degrees for better readability
ax.tick_params(axis='x', which='minor', labelsize='small', labelcolor='m', rotation=30)

# Annotate each bar in the plot with its height (frequency value)
for p in ax.patches:
    ax.annotate(str(p.get_height()), # Convert the height to string for annotation
                (p.get_x() * 1.005, p.get_height() * 1.005)) # Position the annotation slightly above the bar

# Display the plot with the topic frequencies
plt.show()

# Import the os module to handle file paths and directories
import os

# Define BASE_PATH as the directory where the 'efgh-english.csv' file is located
BASE_PATH = os.path.dirname(os.path.abspath('efgh-english.csv'))

# Construct the DATA_PATH variable by joining BASE_PATH with the subdirectory 'Data'
DATA_PATH = os.path.join(BASE_PATH, 'Data')

# FOR PROS---------------

# Create a DataFrame from the document-topic distributions (document_topic1)
# The 'orient' parameter specifies that the keys of the dictionary represent the index (rows) of the DataFrame
topics_all1 = pd.DataFrame.from_dict(document_topic1, orient='index')

# Generate column names for the topics in the DataFrame (e.g., topic_0, topic_1, ..., topic_4)
topic_column_names = ['topic_' + str(i) for i in range(0, 5)]

# Assign the generated topic column names to the DataFrame
topics_all1.columns = topic_column_names

# Save the DataFrame topics_all1 to a CSV file named 'topic_propensities1.csv' in the BASE_PATH directory
topics_all1.to_csv(os.path.join(BASE_PATH, "topic_propensities1.csv"))

# Display the first few rows of the topics_all1 DataFrame to visualize the topic distributions
display(topics_all1.head())

# FOR CONS----------------

# Create a DataFrame from the document-topic distributions (document_topic2)
# The 'orient' parameter specifies that the keys of the dictionary represent the index (rows) of the DataFrame
topics_all2 = pd.DataFrame.from_dict(document_topic2, orient='index')

# Assign the generated topic column names to the DataFrame
topics_all2.columns = topic_column_names

# Save the DataFrame topics_all2 to a CSV file named 'topic_propensities2.csv' in the BASE_PATH directory
topics_all2.to_csv(os.path.join(BASE_PATH, "topic_propensities2.csv"))

# Display the first few rows of the topics_all2 DataFrame to visualize the topic distributions
display(topics_all2.head())





# Import the hierarchy module from the scipy.cluster library for hierarchical clustering for pros
from scipy.cluster import hierarchy

# Set the size of the figure to be displayed (width: 10 inches, height: 7 inches) for pros
plt.figure(figsize=(10, 7))

# Set the title of the plot to "Dendrograms" for pros
plt.title("Dendrograms for Pros")

# Create a dendrogram to visualize the hierarchical clustering for pros
# The linkage method used is 'ward', which minimizes the variance within clusters for pros
dend = hierarchy.dendrogram(hierarchy.linkage(topics_all1, method='ward'))

# Draw a horizontal line at y=9 on the dendrogram for reference for pros,
# indicating a threshold for cutting the dendrogram into clusters
plt.axhline(y=9, color='r', linestyle='--')





# Set the size of the figure to be displayed (width: 10 inches, height: 7 inches) for cons
plt.figure(figsize=(10, 7))

# Set the title of the plot to "Dendrograms for Cons" for cons
plt.title("Dendrograms for Cons")

# Create a dendrogram to visualize the hierarchical clustering for cons
# The linkage method used is 'ward', which minimizes the variance within clusters for cons
dend = hierarchy.dendrogram(hierarchy.linkage(topics_all2, method='ward'))

# Draw a horizontal line at y=9 on the dendrogram for reference for cons,
# indicating a threshold for cutting the dendrogram into clusters
plt.axhline(y=9, color='r', linestyle='--')




# Create a copy of the pros DataFrame for hierarchical visualization
df_for_h_visual1 = dfp

# Add cluster assignments from topics_all1 (related to pros)
df_for_h_visual1["cluster"] = topics_all1["cluster"]

# Remove the 'prospensity' column, not needed for visualization
df_for_h_visual1.drop(['prospensity'], axis=1, inplace=True)

# Replace NaN values in the 'topic' column with "Unknown"
df_for_h_visual1.topic.fillna(value="Unknown", inplace=True)

# Display the first few rows of the modified DataFrame
df_for_h_visual1.head()




# Create a copy of the cons DataFrame for hierarchical visualization
df_for_h_visual2 = dfc

# Add cluster assignments from topics_all2 (related to cons)
df_for_h_visual2["cluster"] = topics_all2["cluster"]

# Remove the 'prospensity' column, not needed for visualization
df_for_h_visual2.drop(['prospensity'], axis=1, inplace=True)

# Replace NaN values in the 'topic' column with "Unknown"
df_for_h_visual2.topic.fillna(value="Unknown", inplace=True)

# Display the first few rows of the modified DataFrame
df_for_h_visual2.head()



#This code essentially creates a stacked bar chart that visualizes the distribution of topics across different clusters for the pros section.
# Group the data by 'topic' and 'cluster', counting occurrences in the 'SUMMARY' column
df_histo1 = df_for_h_visual1.groupby(['topic', 'cluster']).count().reset_index()

# Pivot the DataFrame to create a matrix of topics vs. clusters
df_histo1 = df_histo1.pivot(index='topic', columns='cluster', values='SUMMARY')

# Create a list of new column names for clusters (c0, c1, ..., c7)
col = []
for i in range(8):
    c = 'c' + str(i)
    col.append(c)

# Assign the new column names to the DataFrame
df_histo1.columns = col

# Plot a stacked bar chart of topics against clusters with the 'inferno' colormap
ax = df_histo1.plot.bar(stacked=True, colormap='inferno', edgecolor='black', linewidth=1)

# Adjust the legend position to be on the right side of the plot
ax.legend(loc='center left', bbox_to_anchor=(1.0, .5))

# Hide the top, right, bottom, and left spines of the plot for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Display the plot
plt.show()






# This code creates a stacked bar chart that visualizes the distribution of topics across different clusters for the cons section.
# Group the data by 'topic' and 'cluster', counting occurrences in the 'SUMMARY' column
df_histo2 = df_for_h_visual2.groupby(['topic', 'cluster']).count().reset_index()

# Pivot the DataFrame to create a matrix of topics vs. clusters
df_histo2 = df_histo2.pivot(index='topic', columns='cluster', values='SUMMARY')

# Assign the new column names to the DataFrame
df_histo2.columns = col

# Plot a stacked bar chart of topics against clusters with the 'inferno' colormap
ax = df_histo2.plot.bar(stacked=True, colormap='inferno', edgecolor='black', linewidth=1)

# Adjust the legend position to be on the right side of the plot
ax.legend(loc='center left', bbox_to_anchor=(1.0, .5))

# Hide the top, right, bottom, and left spines of the plot for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Display the plot
plt.show()





"""This code uses the Elbow Method to determine the optimal number of clusters for the KMeans algorithm on the topics_all1 dataset (pros). 
It calculates the sum of squared distances (inertia) for a range of cluster numbers and plots this to help identify the point where adding more 
clusters provides diminishing returns."""
from sklearn.cluster import KMeans
# Initialize an empty list to hold the sum of squared distances for each k
Sum_of_squared_distance = []
# Define the range of k values from 1 to 19
K = range(1, 20)
# Iterate through each k value
for k in K:
    # Create a KMeans model with the current number of clusters k
    km = KMeans(n_clusters=k)
    # Fit the model to the topics_all1 data
    km = km.fit(topics_all1)
    # Append the inertia (sum of squared distances) to the list
    Sum_of_squared_distance.append(km.inertia_)
# Plot the elbow curve
plt.plot(K, Sum_of_squared_distance, 'bx-')
plt.xlabel('k')  # Label for the x-axis
plt.ylabel('Sum of squared distance')  # Label for the y-axis
plt.title('Elbow Method For Optimal clusters')  # Title of the plot
plt.show()  # Display the plot

"""Similarly, this code performs the Elbow Method for the topics_all2 dataset (cons). It follows the same process of calculating and plotting 
the sum of squared distances to determine the optimal number of clusters for the KMeans algorithm. """
# Initialize an empty list to hold the sum of squared distances for each k
Sum_of_squared_distance = []
# Define the range of k values from 1 to 19
K = range(1, 20)
# Iterate through each k value
for k in K:
    # Create a KMeans model with the current number of clusters k
    km = KMeans(n_clusters=k)
    # Fit the model to the topics_all2 data
    km = km.fit(topics_all2)
    # Append the inertia (sum of squared distances) to the list
    Sum_of_squared_distance.append(km.inertia_)
# Plot the elbow curve
plt.plot(K, Sum_of_squared_distance, 'bx-')
plt.xlabel('k')  # Label for the x-axis
plt.ylabel('Sum of squared distance')  # Label for the y-axis
plt.title('Elbow Method For Optimal clusters')  # Title of the plot
plt.show()  # Display the plot





#This code checks each topic score for both pros and cons sections. If a score is below 0.1 or there are no nouns 
#(`Noun_count` is 0), it sets that topic score to 0, filtering out low-relevance topics and rows without nouns.

# Create a copy of `topics_all1` DataFrame for manipulation (PROS)
indi1 = topics_all1

# Loop through each row in `topics_all1`
for i in range(len(topics_all1)):
    # Loop through each topic column (topic_0 to topic_4)
    for j in range(5):
        n = "topic_" + str(j)  # Define the topic column name
        
        # Check if the topic value is below 0.1 or if `Noun_count` is 0
        if(indi1.loc[i].at[n] < 0.1 or indi1.loc[i].at['Noun_count'] == 0):
            # Set the topic value to 0 if either condition is met
            indi1.at[i, n] = 0


# Create a copy of `topics_all2` DataFrame for manipulation (CONS)
indi2 = topics_all2

# Loop through each row in `topics_all2`
for i in range(len(topics_all2)):
    # Loop through each topic column (topic_0 to topic_4)
    for j in range(5):
        n = "topic_" + str(j)  # Define the topic column name
        
        # Check if the topic value is below 0.1 or if `Noun_count` is 0
        if(indi2.loc[i].at[n] < 0.1 or indi2.loc[i].at['Noun_count'] == 0):
            # Set the topic value to 0 if either condition is met
            indi2.at[i, n] = 0






#This code calculates the weighted probabilities for each topic in the pros and cons sections by multiplying each topic’s score by the noun count for each 
#review. These weighted probabilities are then stored in separate columns in the ind_prob DataFrame for pros (_pb_pros) and cons (_pb_cons).


# Initialize an empty DataFrame to store probabilities for pros and cons
ind_prob = pd.DataFrame()

# Loop through each topic (0 to 4) for the pros section
for i in range(5):
    a = []  # Initialize an empty list to store calculated probabilities for each row
    for j in range(len(indi1)):  # Loop through each row in the 'indi1' DataFrame (pros data)
        p = 'topic_' + str(i)  # Generate the topic column name, e.g., 'topic_0'
        # Calculate the weighted probability by multiplying the topic value by the noun count
        n = float(indi1.loc[j].at[p]) * int(indi1.loc[j].at['Noun_count'])
        a.append(n)  # Append the result to the list
    q = p + "_pb_pros"  # Generate a column name for pros probability
    ind_prob[q] = a  # Add the calculated probabilities as a new column in 'ind_prob'

# Repeat the same process for the cons section
for i in range(5):
    a = []  # Initialize an empty list for cons probabilities
    for j in range(len(indi2)):  # Loop through each row in the 'indi2' DataFrame (cons data)
        p = 'topic_' + str(i)  # Generate the topic column name, e.g., 'topic_0'
        # Calculate the weighted probability by multiplying the topic value by the noun count
        n = float(indi2.loc[j].at[p]) * int(indi2.loc[j].at['Noun_count'])
        a.append(n)  # Append the result to the list
    q = p + "_pb_cons"  # Generate a column name for cons probability
    ind_prob[q] = a  # Add the calculated probabilities as a new column in 'ind_prob'

# Display the resulting DataFrame with topic probabilities for pros and cons
ind_prob









#This code performs sentiment analysis by calculating a sentiment score for each topic based on the difference between pros and cons probabilities. 
#The sentiment scores are stored in ind_sent, and additional ratings are appended to this DataFrame before exporting the results to a CSV file.


# Initialize an empty DataFrame to store sentiment analysis results
ind_sent = pd.DataFrame()

# Loop through each topic (0 to 4) to calculate sentiment scores based on pros and cons probabilities
for j in range(5):
    p = 'topic_' + str(j)            # Define the topic column name (e.g., 'topic_0')
    x = p + "_pb_pros"                # Define the pros probability column for the current topic
    y = p + "_pb_cons"                # Define the cons probability column for the current topic
    a = []                            # Initialize a list to store sentiment scores for each row

    # Calculate sentiment score for each row in 'ind_prob'
    for i in range(len(ind_prob)):
        r = ind_prob.loc[i].at[x] - ind_prob.loc[i].at[y]    # Difference between pros and cons probabilities
        s = ind_prob.loc[i].at[x] + ind_prob.loc[i].at[y]    # Sum of pros and cons probabilities
        q = r / s if s != 0 else 0                           # Calculate sentiment ratio, avoiding division by zero

        a.append(q)  # Append the sentiment score to the list

    q = p + "_sent"                 # Define the column name for the sentiment score of the current topic
    ind_sent[q] = a                 # Add the calculated sentiment scores as a new column in 'ind_sent'

# Handle NaN values in 'ind_sent' by replacing them with 0
for i in range(len(ind_sent)):
    for j in range(5):
        n = "topic_" + str(j) + "_sent"   # Define the sentiment column name for each topic
        if str(ind_sent.loc[i].at[n]) == 'nan':
            ind_sent.at[i, n] = 0         # Replace NaN values with 0

# Add additional rating columns from 'store' DataFrame to the 'ind_sent' DataFrame for further analysis
ind_sent['Overall Rating'] = store['Overall Rating']
ind_sent['Ease of Use'] = store['Ease of Use']
ind_sent['Customer Service'] = store['Customer Service']
ind_sent['Features'] = store['Features']
ind_sent['Value for Money'] = store['Value for Money']
ind_sent['Likelihood to Recommend'] = store['Likelihood to Recommend']

# Display the 'ind_sent' DataFrame with calculated sentiment scores and additional ratings
ind_sent

# Save the sentiment analysis results to a CSV file for future reference
ind_sent.to_csv('Rev_sent.csv')







#This code calculates weighted sentiment probabilities for pros and cons based on noun counts for each topic and stores these values in a new DataFrame 
#(sent_prob), which is then saved as a CSV file.

# Initialize an empty list to store noun counts for each document in the corpus
noun_n = []
for i in corpus:
    noun_n.append(len(i))   # Append the length (number of nouns) of each document to noun_n

# Split noun counts into two lists for pros and cons
noun_n1 = noun_n[0:int(len(df)/2)]      # First half of noun counts for pros
noun_n2 = noun_n[int(len(df)/2):]       # Second half of noun counts for cons

# Create copies of the topic data frames for pros and cons
prob_df2 = topics_all2                  # Data for cons
prob_df1 = topics_all1                  # Data for pros

# Add 'Noun_count' column to both pros and cons DataFrames
prob_df2['Noun_count'] = noun_n2
prob_df1['Noun_count'] = noun_n1

# Prepare a DataFrame to store noun counts and sentiment probabilities for pros and cons
data = { 'Noun_count_p' : noun_n1,
         'Noun_count_c' : noun_n2 }
sent_prob = pd.DataFrame(data)

# Calculate weighted probabilities for each topic in pros and add to sent_prob DataFrame
for i in range(5):
    a = []
    for j in range(len(prob_df1)):
        p = 'topic_' + str(i)                                   # Define the topic column name
        n = float(prob_df1.loc[j].at[p]) * int(prob_df1.loc[j].at['Noun_count'])  # Calculate weighted value
        a.append(n)                                             # Append result to list
    q = p + "_pb_pros"                                          # Define the column name for probability of pros
    sent_prob[q] = a                                            # Add pros probabilities to sent_prob

# Calculate weighted probabilities for each topic in cons and add to sent_prob DataFrame
for i in range(5):
    a = []
    for j in range(len(prob_df2)):
        p = 'topic_' + str(i)                                   # Define the topic column name
        n = float(prob_df2.loc[j].at[p]) * int(prob_df2.loc[j].at['Noun_count'])  # Calculate weighted value
        a.append(n)                                             # Append result to list
    q = p + "_pb_cons"                                          # Define the column name for probability of cons
    sent_prob[q] = a                                            # Add cons probabilities to sent_prob

# Save the sentiment probabilities DataFrame to a CSV file
sent_prob.to_csv('sentiment_prob.csv')







# The code calculates overall sentiment scores (positivity/negativity) and importance levels for each topic by analyzing the balance of pros and cons and 
#their cumulative magnitude. This helps identify each topic's sentiment direction and relative weight in the dataset.


# Calculate sentiment scores for each topic by comparing pros and cons
sent = []
for i in range(5):
    a = 0  # Sum of sentiment difference (pros - cons) for the current topic
    b = 0  # Sum of total sentiment (pros + cons) for the current topic
    s = 'topic_' + str(i) + '_pb_cons'  # Cons column name for the current topic
    r = 'topic_' + str(i) + '_pb_pros'  # Pros column name for the current topic
    
    # Loop through each entry in sent_prob to accumulate pros/cons differences and totals
    for j in range(int(len(sent_prob))):
        a += (float(sent_prob.loc[j].at[r]) - float(sent_prob.loc[j].at[s]))  # Add sentiment difference
        b += (float(sent_prob.loc[j].at[r]) + float(sent_prob.loc[j].at[s]))  # Add total sentiment
    
    t = a / b  # Calculate sentiment score (ratio of difference to total sentiment)
    sent.append(t)  # Append the calculated sentiment score for the topic

sent  # Display the calculated sentiment scores

# Calculate importance of each topic based on overall sentiment magnitude
imp = []
fact = []  # Temporary list to store total sentiment for each topic
x = 0  # Sum of all topic sentiment magnitudes

# First loop: Calculate overall sentiment magnitude sum across all topics and instances
for i in range(len(sent_prob)):
    for j in range(5):
        s = 'topic_' + str(j) + '_pb_cons'  # Cons column name for the current topic
        r = 'topic_' + str(j) + '_pb_pros'  # Pros column name for the current topic
        x += ((sent_prob.loc[i].at[r]) + (sent_prob.loc[i].at[s]))  # Add sentiment magnitude

# Second loop: Calculate total sentiment magnitude per topic
for i in range(5):
    b = 0  # Accumulate sentiment for the current topic
    s = 'topic_' + str(i) + '_pb_cons'
    r = 'topic_' + str(i) + '_pb_pros'
    for j in range(int(len(sent_prob))):
        b += ((sent_prob.loc[j].at[r]) + (sent_prob.loc[j].at[s]))  # Add sentiment magnitude for the topic
    fact.append(b)  # Store total sentiment for each topic

# Calculate the relative importance of each topic by dividing topic total by overall total
for i in fact:
    imp.append(i / x)  # Append the importance ratio for each topic

imp  # Display calculated importance values
sent  # Display calculated sentiment scores








# This code calculates sentiment and importance scores for each topic within groups of data based on company size. The results are organized by topic and 
#company size, which can be used to assess topic sentiment and importance differences across various company sizes.

import pandas as pd

# Load a CSV file with company data
store = pd.read_csv('hospitality data.csv')

# Add a new column to 'sent_prob' with company size information from 'store'
sent_prob['columename'] = store['Company Size']

# View the updated 'sent_prob' DataFrame
sent_prob

# Sort 'sent_prob' by company size for further grouping
size_sent = sent_prob.sort_values(['columename'])

# Create a list of unique company sizes
size = []
for i in range(len(store)):
    a = store.loc[i].at['Company Size']
    if a not in size:
        size.append(a)

# Group 'size_sent' by company size and create a list of DataFrames for each group
grouped = size_sent.groupby('columename')
dataframes_s = [grouped.get_group(x) for x in grouped.groups] # List of DataFrames by company size

# Sort the list of unique company sizes
size.sort()

# Merge the last two DataFrames in the list if needed
dataframes_s[-3] = dataframes_s[-3].append(dataframes_s[-2])
dataframes_s.pop(-2)  # Remove the now-merged DataFrame

# Initialize lists to store sentiment and importance values by company size
sent_size = []
imp_size = []

# Calculate sentiment and importance scores for each company size group
for dataframes in dataframes_s:
    # Reset index for dataframes after the first one
    if l != 0:
        dataframes = dataframes.reset_index()
        l = l + 1

    # Initialize a list for sentiment scores for each topic in the current group
    sent_s0 = []
    for i in range(5):  # For each topic (5 topics assumed)
        a = 0
        b = 0
        s = 'topic_' + str(i) + '_pb_cons'
        r = 'topic_' + str(i) + '_pb_pros'
        # Calculate sentiment for the topic by comparing pros and cons
        for j in range(int(len(dataframes))):
            a = a + (float(dataframes.loc[j].at[r]) - float(dataframes.loc[j].at[s]))
            b = b + (float(dataframes.loc[j].at[r]) + float(dataframes.loc[j].at[s]))
        t = a / b  # Sentiment score for the topic
        sent_s0.append(t)
    sent_size.append(sent_s0)  # Append sentiment scores for current company size

    # Initialize lists for importance calculation
    imp_s0 = []
    fact_s0 = []
    x_s0 = 0

    # Calculate total weight (importance) for each topic
    for l in range(len(dataframes)):
        for m in range(5):  # Again, assuming 5 topics
            s = 'topic_' + str(m) + '_pb_cons'
            r = 'topic_' + str(m) + '_pb_pros'
            x_s0 = x_s0 + ((dataframes.loc[l].at[r]) + (dataframes.loc[l].at[s]))

    # Calculate importance for each topic within this company size group
    for k in range(5):
        b = 0
        s = 'topic_' + str(k) + '_pb_cons'
        r = 'topic_' + str(k) + '_pb_pros'
        for n in range(int(len(dataframes))):
            b = b + ((dataframes.loc[n].at[r]) + (dataframes.loc[n].at[s]))
        fact_s0.append(b)

    # Normalize importance scores and add them to the list
    for p in fact_s0:
        imp_s0.append(p / x_s0)
    imp_size.append(imp_s0)  # Append importance scores for current company size






#The code creates a scatter plot comparing sentiment and importance scores across client size groups, using distinct colors and annotations for each group.

# Import the matplotlib library for plotting
import matplotlib.pyplot as plt

# Define color map options for scatter plots
s = ['Blues', 'pink', 'Greens', 'summer', 'Purples', 'Accent', 'Greys', 'bone', 'spring', 'twilight']

# List of labels for annotating points in the plot
n = [0, 1, 2, 3, 4]

# Set up the size of the plot figure
plt.figure(figsize=(20, 10))

# Loop through each of the 10 sets of sentiment and importance data
for i in range(10):
    
    # Set x and y to the sentiment and importance scores for group i
    x = sent_size[i]
    y = imp_size[i]
 
    # Create a scatter plot for each group with varying color maps
    plt.scatter(x, y, s=500, cmap=s[i])  # s=500 sets marker size, cmap sets color map
    
    # Annotate each point in the plot with corresponding labels from list 'n'
    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]))  # Position text annotations at each (x, y) point
    
    # Optional code to limit plot x and y ranges
    # plt.xlim((-1, 8))
    # plt.ylim((0, 8))

# Set labels for x and y axes
plt.xlabel('Sentiment')
plt.ylabel('Importance')

# Set the title for the plot
plt.title('Scatter Plot: Client size group')

# Create a legend using 'size' labels, positioning it in the upper left corner, and scaling marker sizes
plt.legend(size, loc="upper left", markerscale=0.5)

# Display an empty figure for potential additional plots
plt.figure()






#The code creates a scatter plot with each point representing a topic’s sentiment and importance scores. Labels for each topic are displayed next to 
#their points in blue, inside a light blue text box.

# Define color schemes and labels for topics
s = ['Greens', 'summer', 'Purples', 'Accent', 'Greys', 'bone', 'spring', 'twilight']  # Unused in this specific plot
n = ['Topic_0', 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4']  # Labels for each topic

# Set up the plot dimensions
plt.figure(figsize=(20,10))

# Assign sentiment and importance values to x and y
x = sent  # Sentiment scores for each topic
y = imp   # Importance scores for each topic

# Plot a scatter plot with specified marker color and size
plt.scatter(x, y, c='orange', s=1000)  # Scatter plot with orange markers, size 1000

# Annotate each point with its topic label
for i, txt in enumerate(n):
    plt.text(x[i], y[i], txt, fontdict=dict(color='blue', alpha=0.5, size=15),
             bbox=dict(facecolor='lightblue', alpha=0.2))  # Text annotation with styling

# Set x and y labels
plt.xlabel('Sentiment')
plt.ylabel('Importance')

# Add a title to the plot
plt.title('Scatter Plot')

# Create an additional empty plot figure (useful for chaining or setting up further plots)
plt.figure()


