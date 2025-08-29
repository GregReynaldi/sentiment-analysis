from scrape import take_the_content, URL_NEG, URL_POS
from pre_process import clean_the_content, make_word_token, make_more_clean, make_n_grams
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import random
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import store_params

# STEP 1 : Scrape the Content from the URL
pos = take_the_content(URL = URL_POS)
neg = take_the_content(URL = URL_NEG)

# STEP 2 : PreProcess (Lowering + Sent_tokenize + N-Grams + Lemmatize)
dataText = []
all_dataText = []

for categoryText,categoryName in zip([pos,neg],["pos","neg"]) : 
    result = clean_the_content(categoryText)
    for sentence in result : 
        raw_word_token = make_n_grams(sentence)
        raw_word_token = make_word_token(raw_word_token)
        # Clean the stopwords and do lemmatization and n-grams also
        clean_word_token = make_more_clean(raw_word_token)
        dataText.append([clean_word_token,categoryName])
        all_dataText.extend(clean_word_token)
print("STEP 2 Already Done")

# STEP 3 : Process the dataText to become the data that can be trained 
all_dataText = nltk.FreqDist(all_dataText).most_common(3000)
with open("data.txt","a") as f : 
    for i in all_dataText : 
        f.write(i[0]+"\n")

for text_idx in range(len(dataText)) :
    result = dict()
    data_before = set(dataText[text_idx][0])
    for all_key in all_dataText : 
        result[all_key[0]] = all_key[0] in data_before
    dataText[text_idx][0] = result
for _ in range(5) : 
    random.shuffle(dataText)
print("STEP 3 Already Done")

# STEP 4 : Split Data Train and Test
data_train = dataText[:int(len(dataText)*0.8)]
data_test = dataText[int(len(dataText)*0.8):]


# STEP 5 : Build a Model [FIT DATA INTO MODEL]
print("TRAINING MODE STARTED")
model_original = nltk.NaiveBayesClassifier.train(data_train)
model_svc = SklearnClassifier(SVC()).train(data_train)
model_linear_svc = SklearnClassifier(LinearSVC()).train(data_train)
model_mnb = SklearnClassifier(MultinomialNB()).train(data_train)
model_bnb = SklearnClassifier(BernoulliNB()).train(data_train)
model_lr = SklearnClassifier(LogisticRegression()).train(data_train)
model_sgd = SklearnClassifier(SGDClassifier()).train(data_train)

# STEP 6 : Check the Accuracy of Test Data : 
accuracy = nltk.classify.accuracy(model_original, data_test)
print(f"Accuracy Original : {accuracy}")
accuracy = nltk.classify.accuracy(model_svc, data_test)
print(f"Accuracy Linear SVC : {accuracy}")
accuracy = nltk.classify.accuracy(model_mnb, data_test)
print(f"Accuracy MultinomialNB : {accuracy}")
accuracy = nltk.classify.accuracy(model_bnb, data_test)
print(f"Accuracy BernoulliNB : {accuracy}")
accuracy = nltk.classify.accuracy(model_lr, data_test)
print(f"Accuracy Logist Regr : {accuracy}")
accuracy = nltk.classify.accuracy(model_sgd, data_test)
print(f"Accuracy Grad Descent : {accuracy}")

# STEP 6 - Post : Check the Answer of First Data Test 
prediction = model_original.classify(data_test[0][0])
print(str(prediction))

# STEP 7 : Store The Data of Every Models in Pickle File 
method = store_params.SavePickle(original = model_original, lin_svc = model_linear_svc, mnb = model_mnb, bnb = model_bnb,
                                 log_r = model_lr, sgd = model_sgd).save()