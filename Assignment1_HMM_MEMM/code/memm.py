import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from collections import Counter, defaultdict
import heapq
import random

df_train = pd.read_csv('train.csv')
train_data = [ast.literal_eval(row['tagged_sentence']) for _, row in tqdm(df_train.iterrows())]

df_test = pd.read_csv('test3.csv')
test_data = {row['id']: ast.literal_eval(row['untagged_sentence']) for _, row in tqdm(df_test.iterrows())}


def display_data(sentence_index):
    '''
        Input : 'sentence_index' (int) -> index of a sentence in training data
        Output: None
    '''
    sentence = train_data[sentence_index]
    print("TOKEN -> TAG")
    print('...')
    for token, tag in sentence:
        print(token, '>', tag)
sentence_index = random.choice(range(len(train_data)))
#display_data(sentence_index)
distinct_tags=[]
distinct_words=[]
word_tags = []

def extract_features(sentence, i):
    word = sentence[i][0]
    f = {}
    f[1] = 1 if i == 0 else 0  # Start of sentence
    f[2] = 1 if i == len(sentence) - 1 else 0  # End of sentence
    f[3] = 1 if word[0].isupper() else 0  # Is in capital
    f[4] = 1 if word.endswith('ed') else 0  # End with 'ed'
    f[5] = 1 if any(char.isdigit() for char in word) else 0  # Contains digits
    f[6] = 1 if word.endswith('ing') else 0
    f[7] = 1 if any(char in word for char in ['-']) else 0
    f[8] = 1 if any(char in word for char in [ '_']) else 0
    
    if i != 0:
        f[f'prev_tag_{sentence[i-1][1]}'] = 1  # Tag of the previous word
    else :
        f[f'prev_tag_START'] = 1 
    f[f'word_{sentence[i][0]}'] = 1
        #f[f'prev_word_tag_{sentence[i-1][0]}_{sentence[i-1][1]}'] = 1  # Word and tag of the previous word
    # else:
    #     f[f'prev_tag_START'] = 1  # Tag of the previous word (start of sentence)
    #     f[f'prev_word_tag_{sentence[i][0]}_START'] = 1  # Word and tag of the previous word (start of sentence)
    
    print(f"Values before conversion: {f}")
   
    f = {key: int(value) for key, value in f.items()}
    
    return f


def extract_sentence_features(sentence):
    return [extract_features(sentence, i) for i in range(len(sentence))]

def extract_labels(sentence):
    return [tag for _, tag in sentence]

X_train=[]
Y_train=[]
for sentence in train_data:
    X_train.extend(extract_sentence_features(sentence))
    Y_train.extend(extract_labels(sentence))

#X_train_vectorized = [x.values() for x in X_train]
for i, sample in enumerate(X_train):
    X_train[i] = {str(key): value for key, value in sample.items()}

vectorizer = DictVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

clf = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200, C=1.0)
clf.fit(X_train_vectorized, Y_train)

classifier_classes = clf.classes_
#print('nothing')
#states=list(set(Y_train))

states=list(classifier_classes)
S = len(states)

def seq_algo(sentence):
    T = len(sentence)
    
    viterbi = defaultdict(dict)
    backpointer = defaultdict(dict)


    for state in states:
        dummy_sentence = [(word, '<unk>') for word in sentence]
        #dummy_sentence[t] = (dummy_sentence[t][0], prev_state)
        viterbi[state][0] = clf.predict_proba(vectorizer.transform([extract_features(dummy_sentence, 0)]))[0][states.index(state)]
        backpointer[state][0] = None
    
    top_states = []
    for t in range(1, T):
        top_states = [(-viterbi[state][t-1],state) for state in states]
        while len(top_states) > 5:
            heapq.heappop(top_states)
        for j, s in enumerate(states):
            for neg_prob, prev_state in top_states:
                prev_prob = -neg_prob
                #dummy_sentence = [(word, '<unk>') for word in sentence]
                dummy_sentence[t-1] = (dummy_sentence[t-1][0], prev_state)  # Update the tag for the current word
                transition_prob = clf.predict_proba(vectorizer.transform([extract_features(dummy_sentence, t)]))[0][j]
                prob = prev_prob * transition_prob

            
            viterbi[s][t] = -top_states[0][0] 
            backpointer[s][t] = top_states[0][1] 

            # Clear top_states for the next iteration
        

    best_path = []
    max_prob = 0
    for state in states:
        if viterbi[state][T - 1] > max_prob:
            max_prob = viterbi[state][T - 1]
            best_state = state
    best_path.append(best_state)
    prev_state = best_state
    for t in range(T - 1, 0, -1):
        best_state = backpointer[prev_state][t]
        best_path.insert(0, best_state)
        prev_state = best_state

    return best_path


def memm_tagger_util(sent_id, untagged_sentence):
    tagged_tags = seq_algo(untagged_sentence)
    tagged_sentence = [(word, tag.upper()) for word, tag in zip(sentence, tagged_tags)]
    store_submission(sent_id, tagged_sentence)

def store_submission(sent_id, tagged_sentence):
    
    global submission
    submission['id'].append(sent_id)
    submission['tagged_sentence'].append(tagged_sentence)

def clear_submission():
    global submission
    submission = {'id': [], 'tagged_sentence' : []}

submission = {'id': [], 'tagged_sentence': []}
for sent_id, sentence in tqdm(test_data.items()):
    memm_tagger_util(sent_id,sentence)

pd.DataFrame(submission).to_csv('submission.csv', index=False)
