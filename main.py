# Import language libraries
import wikipedia, nltk, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import difflib as dl


def filter_pos(raw_text, pos):
    '''
    This function will take in raw text and return a pandas DataFrame containing a unique set of words 
    whose parts of speech match what is defined by pos
    raw_text: raw string from data source
    List pos: a list of strings that define what parts of speech is returned want
    returns: the filtered words
    '''
    columns = ['word', 'tag']
    tagged = nltk.pos_tag(nltk.word_tokenize(raw_text))
    df = pd.DataFrame([x for x in tagged], columns=columns)
    filtered = df[df.tag.isin(pos)].drop_duplicates().reset_index(drop=True)
    return filtered

def makeSampleDataset(list_of_foods, pos):
    '''
    This function takes a list of foods and the parts of speech we want to 
    keep from their wikipedia results. 
    list_of_foods: a list of foods that are valid wikipedia titles
    pos: a list of parts of speech as specified by the NLTK library
    returns: a dataframe with the food as one column and all its descriptors as the next column
    
    '''
    dishes_row = []
    dd_columns = ['title', 'description']
    verified_list = []
    
    for i in range(len(list_of_foods)):
        # checks if the title exists. If it does, append it to the search list
        # if it isn't in there, print the error
        try:
            wikipedia_title = (wikipedia.WikipediaPage(title = list_of_foods[i]).title)
            if wikipedia_title != None:
                verified_list.append(list_of_foods[i])
        except expression as identifier:
            print(identifier)
            pass
        
        
            
    for i in range(len(verified_list)):
        # gets the content (the text on the page) from wikipedia
        wikipedia_content = (wikipedia.WikipediaPage(title = list_of_foods[i]).content)
        filtered_item = filter_pos(wikipedia_content, pos)
        
        description = np.array([])
        for j in range(len(filtered_item)):
            description = np.append(description, filtered_item.word[j])
            
        tup = (list_of_foods[i], ', '.join(description))
        dishes_row.append(tup)
        
    dish_description_df = pd.DataFrame([x for x in dishes_row], columns = dd_columns)
    return dish_description_df

# Function that takes in food title as input and outputs most similar foods
def get_recommendations(title, data):
    '''
    Takes in a name of a food and a food description dataframe and returns the 
    top ten most similar foods to the title food based on the cosine similarity of a 
    tfidf matrix calculated from their descriptions
    title: the food you want a recommendation for
    data: the dataframe containing food names and their descriptions
    returns: the top ten most similar foods
    '''
    
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # TODO: Change this matrix into the form of columns = (item, descriptor, tfidf_score)
    
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(data['description'])
    # print('Size of tfidf_matrix: ', tfidf_matrix.shape)
    #tfidf_df = pd.DataFrame(tfidf_matrix)

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    #Construct a reverse map of indices and food
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    
    # Get the index of the food that matches the description
    idx = indices[title]

    # Get the pairwsie similarity scores of all foods with that food
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the foods based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar foods
    sim_scores = sim_scores[1:11]

    # Get the food indices
    item_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar foods
    return data['title'].iloc[item_indices], tfidf_matrix


pos = ['JJ', 'VB', 'VBG', 'VBP', 'VvBZ']
pos_simple = ['JJ', 'NN']
# a list of foods I compiled that have wikipedia articles
foodList = pd.DataFrame(pd.read_csv('foodlist.csv'))
# reformat the dataframe
foodList = foodList.loc[:, ~foodList.columns.str.contains('^Unnamed')]
if os.path.exists('foodSampleData.csv'):
    foodSampleData = pd.read_csv('foodSampleData.csv')
else:
    foodSampleData = makeSampleDataset(foodList['items'], pos_simple)


    
while True:  
    print("Please type in a food item: ")
    food = input()  
    foodSearchBank = list(foodList['items'])

    if food not in foodSearchBank:
        recFood = dl.get_close_matches(food, foodSearchBank, n=1)
        print("Your search isn't in our food bank, here is the closest match: ", recFood)
        if not recFood:
            print(f"{food} IS NOT IN OUR LIST. PLEASE ENTER ANOTHER FOOD")
            recFood = ['Wonton']
        recommendations, tfidf_csr_matrix = get_recommendations(recFood[0], foodSampleData)
        print(recommendations)
    else: 
        recommendations, tfidf_csr_matrix = get_recommendations(food, foodSampleData)
        print(recommendations)
        
    
  
        
