#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import random
import string # to process standard python strings
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # convert a collection of raw documents to a matrix of TF-IDF features
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer


# In[2]:


get_ipython().system('pip install nltk')


# In[8]:


nltk.download('popular',quiet=True)
nltk.download('punkt')
nltk.download('wordnet')


# In[9]:


f=open('input.txt','r',errors='ignore')
raw=f.read()
raw=raw.lower()


# In[5]:


#Tokenization
import nltk
sent_tokens=nltk.sent_tokenize(raw)
word_tokens=nltk.word_tokenize(raw)


# In[12]:


lemmer=nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[18]:


GREETING_INPUTS=("hello","hi","greetings","what's up","hey",\
                  "how are you?")
GREETING_RESPONSES=["hi","hey","hi there","hello",\
                    "I am glad! You are talking to me",\
                    "I am fine! How about you?"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return
        random.choice(GREETING_RESPONSES)


# In[23]:


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')  
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you."
        return robo_response


# In[24]:


flag=True
print("SABot: My name is SABot. How can I assist you ?. \
If you want to exit, type Bye!")
while(flag == True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you' ):
            flag=False
            print("SABot: You are welcome ... ")
        else:
            if(greeting(user_response) != None):
                print("SABot: "+greeting(user_response))
            else:
                print("SABot: ",end=" ")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("SABot: Bye! take care ... ")


# In[ ]:




