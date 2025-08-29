import pickle
from pre_process import make_more_clean, make_n_grams, make_word_token
import os
from statistics import mode

current_path = os.getcwd()  

# INPUT THE QUESTION : 
question = str(input("What is the reviews that gonna be predict : "))

# PRE-PROCESS TEXT : 
question = question.lower()
raw_word_token = make_n_grams(question)
raw_word_token = make_word_token(raw_word_token)
clean_word_token = make_more_clean(raw_word_token)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH,"models")

txtFile = os.path.join(BASE_PATH, "data.txt")
with open(txtFile,"r") as f : 
    data = f.read().split("\n")

clean_word_dict = {}
for d in data : 
    clean_word_dict[d] = d in clean_word_token

# CHECK WHETHER THE MODELS EXISTS OR NAH : 
PATH  = os.path.exists(MODEL_PATH)

if PATH : 
    model = []
    dataModel = [os.path.join(MODEL_PATH,filePICKLE) for filePICKLE in os.listdir(MODEL_PATH) if filePICKLE.endswith('.pickle')]
    for dm in dataModel:
        with open(f"{dm}","rb") as f: 
            model.append(pickle.load(f))

    voted = [LLM.classify(clean_word_dict) for LLM in model]
    result_of_voted = mode(voted)
    possibility = voted.count(result_of_voted)/6*100
    if result_of_voted == "pos" : 
        result_of_voted = "POSITIVE"
    else : 
        result_of_voted = "NEGATIVE"
    print(f"This Review is {result_of_voted} review with {possibility}% Sure")

else : 
    print("TRAIN THE MODEL FIRST ON train.py - REYNALDI")
