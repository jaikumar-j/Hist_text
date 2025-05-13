import streamlit as st
from textblob import TextBlob
import spacy
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import string

nltk.download('stopwords')
from nltk.corpus import stopwords
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  
    text = ''.join([char for char in text if char not in string.punctuation]) 
    tokens = text.split() 
    tokens = [word for word in tokens if word not in stop_words]  
    return tokens

def perform_topic_modeling(texts, num_topics=3):
    dictionary = corpora.Dictionary(texts)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]
    lda_model = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=4)
    return topics

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)

st.title("Historical Text Analysis")

uploaded_file = st.file_uploader("Upload a Historical Document (txt)", type="txt")
if uploaded_file is not None:
    text = uploaded_file.read().decode('utf-8')
    st.write("### Original Text")
    st.write(text[:1000] + '...') 
    tokens = preprocess_text(text)
    
    st.write("### Topic Modeling")
    texts = [tokens]
    topics = perform_topic_modeling(texts)
    for i, topic in enumerate(topics):
        st.write(f"Topic {i+1}: {topic}")
    
    st.write("### Sentiment Analysis")
    sentiment = analyze_sentiment(text)
    st.write(f"The sentiment of the text is: {sentiment}")
    
    st.write("### Named Entity Recognition")
    entities = extract_entities(text)
    for entity, label in entities:
        st.write(f"{entity}: {label}")
    st.write("### Word Cloud")
    generate_wordcloud(text)

