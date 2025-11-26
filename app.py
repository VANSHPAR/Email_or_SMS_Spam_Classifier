import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input=st.text_area("Enter the message")

ps=PorterStemmer()

def transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
       y.append(ps.stem(i))

    return " ".join(y)

if st.button('Predict'):
    transformed_sms=transform(input)
    vectorized_sms=tfidf.transform([transformed_sms])
    res=model.predict(vectorized_sms)[0]

    if res==1:
      st.header("Spam")
    else:
      st.header("Not Spam")

