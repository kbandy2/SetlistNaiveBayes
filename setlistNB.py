import urllib3
import ast
from time import sleep
import pandas as pd

#Function to get a page of data

def get_data (page_num, mbid, api_key):
    url = "https://api.setlist.fm/rest/1.0/artist/" + mbid + "/setlists?p=" + str(page_num)
    
    headers = {"x-api-key": api_key, "Accept": "application/json"}
    
    http = urllib3.PoolManager()
     
    response = http.request('GET', url, headers=headers)
     
    setlists = response.data.decode("utf-8").replace("null", "0").replace("true", "True").replace("false", "False")
    
    setlists = ast.literal_eval(setlists)
    
    sleep(2)
    
    return setlists


#Get 20 pages of data

results_list = []

for i in range(1,number_of_results_pages+1):
    results_list.append(get_data(i, mbid, api_key))
    
    
#Un-nest the songs and compile into a list of lists

master_setlist_list = []
song_list = []
for index, i in enumerate(results_list):
    setlist = i['setlist']
    for j in setlist:
        sets = j['sets']['set']
        this_setlist = []
        for k in sets:
            songs = k['song']
            for q in songs:
                this_setlist.append(q['name'])
        master_setlist_list.append(this_setlist)
        
#Remove one-off appearances

master_setlist_list = [i for i in master_setlist_list if len(i) > 10]  


#Start model here

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import numpy as np

#Format data such that the response (Y) is the next song and the predictors (X) are the songs preceding.

Y = []
X_songs = []

for i in master_setlist_list:
    set_length = len(i)
    for j in range(1,set_length-1):
        X_songs.append(i[:j])
        Y.append(i[j])
        
X_train, X_test, Y_train, Y_test = train_test_split(X_songs, Y, test_size = .25, shuffle = True, random_state=(54321))
        
cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
m = cv.fit_transform(X_train)
X_headers = cv.get_feature_names()
X_train = m.todense()



#Train Model
model = BernoulliNB(alpha=.5)

model.fit(X_train,Y_train)

def predict_next_song(previous_songs):
    entry = np.zeros(len(X_headers))
    for i in previous_songs:
        ind = X_headers.index(i)
        entry[ind] = 1
    return model.predict(entry.reshape(1,-1))
    

#Test Accuracy
correct = 0

for i, j in zip(X_test, Y_test):
    pred = predict_next_song(i)
    if pred == j:
        correct += 1
        
acc = correct / len(Y_test)
        
baseline_acc = 1 / len(X_headers)

#Hyperparameter tuning

acc_list = []

for i in np.arange(0,1,.1):
    
    model = BernoulliNB(alpha=i)

    model.fit(X_train,Y_train)
        
    correct = 0

    for i, j in zip(X_test, Y_test):
        pred = predict_next_song(i)
        if pred == j:
            correct += 1
    
    acc_list.append(correct / len(Y_test))
    
acc_df = pd.DataFrame({"alpha":np.arange(0,1,.1), "accuracy": acc_list})
display(acc_df.head(10))

#Plot results for model tuning

import matplotlib.pyplot as plt   
import seaborn as sns

plot_df = pd.DataFrame({"Alpha": [round(i,1) for i in np.arange(0,1,.1)], "Accuracy": [round(i,4) for i in acc_list]})

clrs = ['#004B87' if (x < max(acc_list)) else 'red' for x in acc_list]

sns.set(style="darkgrid")
sns.barplot(x = "Alpha", y = "Accuracy",data = plot_df, palette=clrs)


#Re-train model with optimal hyperparameter

model = BernoulliNB(alpha=.2)

model.fit(X_train,Y_train)

predict_next_song(["Cheeseburger in Paradise", "Son of a Son of a Sailor", "Grapefruit—Juicy Fruit", "Lovely Cruise", "It's Five O'Clock Somewhere"])

    
#Start setlist generator here:


#Build distribution to guess length of setlist

from scipy.stats import norm

setlist_lengths = [len(i) for i in master_setlist_list]    
plt.hist(setlist_lengths, bins = len(set(setlist_lengths)), density=True)  
plt.show()     

#Fit the known data to a normal distribution and create function to generate an RV from there               

params = norm.fit(setlist_lengths)

def estimate_setlist_length():
    return round(norm.rvs(loc=params[0], scale=params[1])) 
    
#predict the first song
from random import choice

def predict_first_song():
    first_song_list = [i[0] for i in master_setlist_list]
    return choice(first_song_list)
    
predict_first_song()
    


#Modify prediction function to account for new setlist
def predict_next_song_setlist(previous_songs):
    entry = np.zeros(len(X_headers))
    for i in previous_songs:
        ind = X_headers.index(i)
        entry[ind] = 1
    model_x = entry.reshape(1,-1)
    probabilities = {i:j for i,j in zip(X_headers, model.predict_proba(model_x)[0])}
    probabilities = {i:probabilities[i] for i in probabilities.keys() if i not in previous_songs}
    return max(probabilities, key=probabilities.get)



final_setlist = []

for i in np.arange(estimate_setlist_length()):
    if i == 0:
        final_setlist.append(predict_first_song())
    else:
        next_song = predict_next_song_setlist(final_setlist)
        final_setlist.append(next_song[:])




