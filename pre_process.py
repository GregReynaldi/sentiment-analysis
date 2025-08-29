import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

def clean_the_content(content) : 
    content = content.lower()
    content = content.split("\n")
    return content

def make_n_grams(content:str) : 
    pattern = re.compile(r"(no|not|none|don\'t|never|dont)+\s([a-z]+)")
    for matches in pattern.findall(content) :
        content = content.replace(" ".join(matches), f"{matches[0]}_{matches[1]}")
    return content

def make_word_token(content) : 
    return word_tokenize(content)

def make_more_clean(contentList) : 
    hasil = []
    stop_words = stopwords.words("english")
    for word in contentList : 
        if word not in stop_words and word.isalnum():
            lemmatize_word =  WordNetLemmatizer().lemmatize(word)
            hasil.append(lemmatize_word)
    return hasil

