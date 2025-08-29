import requests
import os

URL_NEG = "https://pythonprogramming.net/static/downloads/short_reviews/negative.txt"
URL_POS = "https://pythonprogramming.net/static/downloads/short_reviews/positive.txt"

def take_the_content(URL) : 
    return requests.get(URL).text

if __name__ == "__main__" : 
    pos = take_the_content(URL_POS)
    neg = take_the_content(URL_NEG)
    print(pos)