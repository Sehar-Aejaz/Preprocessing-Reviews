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

