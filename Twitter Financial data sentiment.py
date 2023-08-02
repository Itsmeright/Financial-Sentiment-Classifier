# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:15:39 2023

@author: HP
"""
#%%importing the necessary libraries
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#set the seed for reproducibility
np.random.seed(42)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
#%%reading the data set
#reading in the dataset
raw_df = pd.read_csv('C:/Users/HP/Desktop/Personal Project/Sentiment Analysis/sent_train.csv')
#Creating an empty Dataframe to hold the clean data
df = pd.DataFrame()
#defining a function to extract the tickers
def extract_tickers(text):
    return re.findall(r'\b[A-Z]{1,5}\b', text)

#defining a function to extract links
def extract_links(text):
    return re.findall(r'http[s]?://\S+', text)

#defining a function to extract the text
def extract_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\$[A-Za-z]+', '', text)  # Remove stock symbols
    text = re.sub(r'\W+', ' ', text.lower())  # Remove non-word characters and convert to lowercase
    return text.strip()
    

#creating a column for tickers
df['tickers'] = raw_df['text'].apply(extract_tickers)
#creating a new column to hold the links
df['links']= raw_df['text'].apply(extract_links)
df['text'] = raw_df['text'].apply(extract_text)
df['label'] = raw_df['label']
#%%Removing stop words and numbers
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove numbers
    no_number = re.sub(r'\d+', '', text)
    #remove single letters
    no_number = re.sub(r'\s+[a-zA-Z]\s+|\s+[a-zA-Z]$|^[a-zA-Z]\s+', ' ', text)
    
    # Remove stopwords
    words = no_number.split()
    words_filtered = [word for word in words if word.lower() not in stop_words]
    
    # Reconstruct the sentence
    return ' '.join(words_filtered)
data = df[['text', 'label']]
data['text'] = data['text'].apply(clean_text)

for i in range(len(data['text'])):
   data ['text'][i]= word_tokenize(data['text'][i])


#%%lemmatizing with NLTk
data['lemma_text'] = None
lemmatizer = WordNetLemmatizer()#initiating the lemmatizer with nltk
for i in range(len(data['text'])):
    data['lemma_text'][i]= [lemmatizer.lemmatize(word) for word in data['text'][i]]
    data['lemma_text'][i] = ' '.join(data['lemma_text'][i])

#%%doing lemmatizing with spacy
lemma = spacy.load('en_core_web_sm')
data['spacy']=None
for i in range(len(data['text'])):
    data['spacy'][i]= [lemmatizer.lemmatize(word) for word in data['text'][i]]
    
#%%Visualizations
plt.pie(data['label'].value_counts(), 
        labels=data['label'].value_counts().index,
        autopct='%.1f')
plt.legend(['Neutral(2)', 'Bullish (1)', 'Bearish (0)'], title='Label')
plt.title('A pie chart showing the label proportions')
plt.show()
#%%Preparing the dataset for the word cloud
#word cloud
neutral= data.query('label==2')
bullish=data.query('label==1')
bearish=data.query('label==0')
#Extracting the neutral words for the word cloud
neutral_list= neutral['lemma_text'].tolist()
neutralwords = ' '.join([' '.join(inner_list) for inner_list in neutral_list])


#extracting the bullish words for a word cloud
bullish_list= bullish['lemma_text'].tolist()
bullishwords = ' '.join([' '.join(inner_list) for inner_list in bullish_list])


#extracting the bearish words for a word cloud
bearish_list= bearish['lemma_text'].tolist()
bearishwords = ' '.join([' '.join(inner_list) for inner_list in bearish_list])




#%%plotting neutral wordclouds
wordcloud_neutral = WordCloud(width=480, height=480, margin=0).generate(neutralwords)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.title('Common words in the neutral sentiments')
plt.show()
#%%plotting bullish word cloud
wordcloud_bullish = WordCloud(width=480, height=480, margin=0).generate(bullishwords)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.title('Common words in the bullish sentiments')
plt.show()

#%%plotting bearish word cloud
wordcloud_bearish = WordCloud(width=480, height=480, margin=0).generate(bearishwords)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis('off')
plt.margins(x=0, y=0)
plt.title('Common words in the bearish sentiments')
plt.show()

#%%Vectorization
#intializing count vectorizer
count_vectorizer = CountVectorizer(max_features=6216)
#Choosing the X and y variables
X = count_vectorizer.fit_transform(data['lemma_text'])
y= data['label']
#spiliting the dataset into training and validation set
x_train, x_val,y_train, y_val = train_test_split(X, y, test_size=0.3, 
                                                 stratify=y)

#%%model selection

models = [(MultinomialNB(), {'alpha':[0.1, 1,10], 'fit_prior': [True, False]}), 
          (SVC(), {'C':[0.1,1,10], 'kernel':['linear', 'rbf']}), 
          (RandomForestClassifier(), {'n_estimators':[50,100,150], 'max_depth':[10,20,50]})]

best_models = []
for model, param_grid in models:
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    best_models.append((best_model, grid.best_params_))
    
    
#%%model prediction

for model, params in best_models:
    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    class_report = classification_report(y_val, y_pred)
    print("Model : ", model, accuracy)
    print("Classification report:",model, class_report)
    
#%%balancing the dataset to increasing precision for 0 label
neutral_b = neutral.sample(n=1442, replace= False, random_state=42)
bullish_b = bullish.sample(n=1442, replace= False, random_state=42)
bearish_b = bearish

balanced_data = pd.concat([neutral_b, bullish_b, bearish_b])

#Splitting balanced data
balanced_data_x = count_vectorizer.fit_transform(balanced_data['lemma_text'])
balanced_data_y = balanced_data['label']
#spiliting the dataset into training and validation set
x_train_b, x_val_b,y_train_b, y_val_b = train_test_split(balanced_data_x, balanced_data_y,
                                                         test_size=0.3, 
                                                         stratify=balanced_data_y)

#%%using the best model on the balanced data
svc_b = SVC(C=10, kernel='rbf')
svc_b.fit(x_train_b, y_train_b)
y_pred_b = svc_b.predict(x_val_b)
accuracy_b = accuracy_score(y_val_b, y_pred_b)
class_report_b = classification_report(y_val_b, y_pred_b)
#%%testing data
test = pd.read_csv('C:/Users/HP/Desktop/Personal Project/Sentiment Analysis/sent_valid.csv')
#Cleaning
test['text']= test['text'].apply(extract_text)
test['text'] = test['text'].apply(clean_text)
for i in range(len(test['text'])):
   test['text'][i]= word_tokenize(test['text'][i])
   
   
for i in range(len(test['text'])):
    test['text'][i]= [lemmatizer.lemmatize(word) for word in test['text'][i]]
    test['text'][i] = ' '.join(test['text'][i])
    
#%%choosing features and targets    
x_test= count_vectorizer.fit_transform(test['text'])
y_test = test['label']
y_pred_test = svc_b.predict(x_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)




