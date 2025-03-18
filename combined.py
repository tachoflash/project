
# Data Loading, Preprocessing, and Adversarial Augmentation 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

#objective 1: to analyze the nature of spam messages
# Import data (adjust file name and encoding as needed)
spam_df = pd.read_csv("spam.csv")

print("Sample data:")
print(spam_df.sample(5))

# Inspect data
print("Data summary by Category:")
print(spam_df.groupby('Category').describe())

# Convert spam/ham to numerical values: spam=1, ham=0
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x.lower() == 'spam' else 0)

#objective 2: coming up with a working spam detector
# Create train/test split
x_train, x_test, y_train, y_test = train_test_split(
    spam_df.Message, 
    spam_df.spam, 
    test_size=0.2, 
    random_state=34
)

print("Training data description:")
print(x_train.describe())

#objective 3: to improve the robustness of the detector
# Define a simple adversarial attack function that perturbs text
def generate_adversarial_text(text):
    """
    Simulates an adversarial attack by swapping two adjacent characters in words.
    This helps the model learn to be robust to minor modifications.
    """
    words = text.split()
    new_words = []
    for word in words:
        if len(word) > 3:
            idx = np.random.randint(0, len(word) - 1)
            word_list = list(word)
            word_list[idx], word_list[idx + 1] = word_list[idx + 1], word_list[idx]
            new_words.append(''.join(word_list))
        else:
            new_words.append(word)
    return " ".join(new_words)

# Augment training data with adversarial examples (one per original example)
x_train_adv = x_train.apply(generate_adversarial_text)
x_train_aug = pd.concat([x_train, x_train_adv], ignore_index=True)
y_train_aug = pd.concat([y_train, y_train], ignore_index=True)  # Labels remain the same

print("Original training examples:", len(x_train))
print("Augmented training examples:", len(x_train_aug))


#Baseline Model Training (CountVectorizer + MultinomialNB) 

# Convert text to a count matrix using CountVectorizer
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train_aug.values)
print("Shape of augmented training count matrix:", x_train_count.shape)

# Train the baseline MultinomialNB model
model = MultinomialNB(alpha=1)
model.fit(x_train_count, y_train_aug)

# Evaluate on the test set (without adversarial augmentation)
x_test_count = cv.transform(x_test)
baseline_test_score = model.score(x_test_count, y_test)
print("Baseline Test score (CountVectorizer):", baseline_test_score)


#ONJECTIVE 4: to boost the accuracy of the spam detector
#  Boosting Accuracy Using TF-IDF and GridSearchCV 

# Create a pipeline that uses TfidfVectorizer and MultinomialNB
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Define a parameter grid to tune the TF-IDF vectorizer and NB's alpha parameter
param_grid = {
    'tfidfvectorizer__max_df': [0.9, 0.95, 1.0],
    'tfidfvectorizer__min_df': [1, 2, 5],
    'multinomialnb__alpha': [0.1, 0.5, 1, 5, 10]
}

# Use GridSearchCV to search for the best parameters with 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2)
grid_search.fit(x_train_aug, y_train_aug)

print("Best parameters from GridSearchCV:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate the tuned model on the test set
tfidf_test_score = grid_search.score(x_test, y_test)
print("Test score after GridSearchCV (TF-IDF):", tfidf_test_score)

# Define a helper function for spam detection using the tuned model
def detect_spam_message(sentence):
    """
    Returns the classification result for the input sentence.
    If spam is detected, a warning is provided.
    """
    # Normalize the sentence to lowercase
    email = [sentence.lower()]
    result = grid_search.predict(email)
    if result[0] == 1:
        return "Spam detected! Warning: Stop sending spam messages!"
    else:
        return "Your sentence was classified as ham."


#  User Input

print("\n=== Spam Detector Ready ===")
while True:
    user_input = input("Enter a sentence to test (enter '1' to quit): ").strip()
    if user_input == "1":
        break
    print(detect_spam_message(user_input))
