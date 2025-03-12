# %%
#import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

# %%
#import data
spam_df=pd.read_csv("spam.csv.txt")
spam_df.sample(5)

# %%
#inspect data
spam_df.groupby('Category').describe()

# %%
#turn spam/ham into numerical data creating a column called 'spam'
spam_df['spam']=spam_df['Category'].apply(lambda x:1 if x=='spam' else 0)

# %%
#create train/test split
x_train,x_test,y_train,y_test=train_test_split(spam_df.Message,spam_df.spam,test_size=0.2)

# %%
x_train.describe()

# %%
#find word count as store data as a matrix
cv=CountVectorizer()
x_train_count= cv.fit_transform(x_train.values)

# %%
x_train_count.toarray()

# %%
#training my model
model=MultinomialNB(alpha=1)
model.fit(x_train_count,y_train)

# %%
email_ham=["Urgent!Please pay 100"]
email_ham_count=cv.transform(email_ham)
model.predict(email_ham_count)

# %%
#pre-test spam
email_spam=["reward money click"]
email_spam_count=cv.transform(email_spam)
model.predict(email_spam_count)

# %%
#test model
x_test_count=cv.transform(x_test)
model.score(x_test_count,y_test)

# %%
#making a pipe and cross-validating

pipe_nb=make_pipeline(CountVectorizer(),MultinomialNB())
scores=cross_validate(pipe_nb,x_train,y_train,return_train_score=True)
pd.DataFrame(scores)

# %%
pd.DataFrame(scores).mean()

# %%
param_grid = {"gamma": [0.1, 1.0, 10, 100]}

# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


grid_search = GridSearchCV(SVC(), param_grid, verbose=2)

# %%
GridSearchCV(estimator=SVC(), param_grid={'gamma': [0.1, 1.0, 10, 100]},
             verbose=2)

# %%

def testWord(word):
    email_ham=[word]
    email_ham_count=cv.transform(email_ham)
    result = model.predict(email_ham_count)
    pred = 'spam' if result[0] == 1 else "ham"
    print("Your sentence was found as", pred)
    
while True:
    word = input("Enter the sentence to test (enter 1 to quit): ")
    if word=="1":
        break
    testWord(word)

    
    


