import pickle
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
st.title("Automatic Language Detector")
cv = pickle.load(open("vectorizer.pickle", 'rb'))
pickled_model = pickle.load(open('lang_model.pkl', 'rb'))
user =  st.text_area("Enter Any Text: ")
if len(user) < 1:
        st.write("  ")
else:
        data = cv.transform([user]).toarray()
        output = pickled_model.predict(data)
        st.title(output)
#print(output)     
