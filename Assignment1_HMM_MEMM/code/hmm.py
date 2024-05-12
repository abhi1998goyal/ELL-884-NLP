import numpy as np 
from tqdm import tqdm
import pandas as pd
import random
import ast
import os
from collections import Counter
from collections import defaultdict


df = pd.read_csv('train1.csv')
data = []
for index, row in tqdm(df.iterrows()):
    tagged_sentence = ast.literal_eval(row['tagged_sentence'])
    lowercase_tagged_sentence = [(token.lower(), tag.lower()) for token, tag in tagged_sentence]
    data.append(lowercase_tagged_sentence)
print(type(data[0][0]))

df = pd.read_csv('test_small1.csv')
test_data = {} 
for index, row in tqdm(df.iterrows()):
    untagged_sentence = ast.literal_eval(row['untagged_sentence'])
    lowercase_untagged_sentence = [token.lower() for token in untagged_sentence]
    test_data[row['id']] = lowercase_untagged_sentence

# print(test_data[4])
# print(test_data[2])

def display_data(sentence_index):
    sentence = data[sentence_index]
    print('TOKEN --> TAG')
    print('...')
    for token, tag in sentence:
        print(token, '>', tag)

distinct_tags = []
word_tags = []
transitions=[]
emissions=[]

def store_tags():
    global distinct_tags
    global word_tags
    
    for sent in data:
        transition=[]
        word_tags.append(('START','START'))
        transition.append('START')
        for words, tag in sent:
            word_tags.extend([(tag, words)])
            transition.append(tag)
        word_tags.append(('END','END'))
        transition.append('END')
        transitions.append(transition)
    
store_tags()
tags=[]
for tag, words in word_tags:
    tags.append(tag)
distinct_tags=list(set(tags))

words=[]
for tag, word in word_tags:
    words.append(word)
distinct_words=list(set(words))

count_tags = {}
for tag, count in Counter(tags).items():
    count_tags[tag] = count
def get_tag_count(tag):
    return count_tags[tag]

#def get_transition_count(s,d):


submission = {'id': [], 'tagged_sentence' : []}
def store_submission(sent_id, tagged_sentence):
    
    global submission
    submission['id'].append(sent_id)
    submission['tagged_sentence'].append(tagged_sentence)
    
def clear_submission():
    global submission
    submission = {'id': [], 'tagged_sentence' : []}

def get_transition(src,dest):
    count=0
    if dest!='ANYTHING':
        for transition in transitions:
            for tags in range(len(transition)-1):
                if transition[tags]==src and transition[tags+1]==dest :
                    count=count+1
    else :
        for transition in transitions:
            for tags in range(len(transition)-1):
                if transition[tags]==src:
                    count=count+1
    return count


transition_probabilities = defaultdict(dict) 
for source in distinct_tags:
     src_transit=get_transition(source,'ANYTHING')
     for dest in distinct_tags:
            if(src_transit!=0):
               transition_probabilities[source][dest]=get_transition(source,dest)/src_transit
            else:
               transition_probabilities[source][dest]=0

for state in distinct_tags:
    row = transition_probabilities[state]
    non_zero_probs = [prob for prob in row.values() if prob > 0]
    num_non_zero_probs = len(non_zero_probs)
    if num_non_zero_probs == 0 or num_non_zero_probs==len(row.values()) or state=='START' or state=='END':
        continue
    else:
        min_prob = min(non_zero_probs)
        smoothing_value = min_prob / (len(row) - len(non_zero_probs))
        smoothing_value /= 10
        max_prob = max(row.values())
        max_prob -= 10 * smoothing_value
        row[max(row, key=row.get)] = max_prob
        for dest, prob in row.items():
            if prob == 0:
                row[dest] = smoothing_value


def get_emission_count(tag,word):
    count=0
    for ta,wor in word_tags:
        if ta==tag and wor ==word:
           count=count+1
    return count

emission_probabilities = defaultdict(dict) 
for source in distinct_tags:
     tag_cnt=get_tag_count(source)
     for dest in distinct_words:
         emission_probabilities[source][dest]=get_emission_count(source,dest)/tag_cnt

for state in distinct_tags:
    row1 = emission_probabilities[state]
    non_zero_probs = [prob for prob in row1.values() if prob > 0]
    num_non_zero_probs = len(non_zero_probs)
    if num_non_zero_probs == 0 or num_non_zero_probs==len(row.values()) or state=='START' or state=='END':
        continue
    else :
        min_prob = min(non_zero_probs)
        smoothing_value = min_prob / (len(row1) - len(non_zero_probs))
        smoothing_value /= 10
        max_prob = max(row1.values())
        max_prob -= 10 * smoothing_value
        row1[max(row1, key=row1.get)] = max_prob
        for dest, prob in row1.items():
            if prob == 0:
                row1[dest] = smoothing_value

def viterbi_algo(sentence, transition_probabilities, emission_probabilities):
    T = len(sentence)
    states = list(transition_probabilities.keys())
    viterbi_matrix = np.zeros((len(states), T))
    backpointer_matrix = np.zeros((len(states), T), dtype=int)

    for i, state in enumerate(states):
        emission_prob = emission_probabilities[state].get(sentence[0], 0)
        viterbi_matrix[i, 0] = transition_probabilities['START'][state] * emission_prob

    print(viterbi_matrix)

    for t in range(1, T):
        for i, state in enumerate(states):
            scores = []
            emission_prob = emission_probabilities[state].get(sentence[t], 0)
            for j in range(len(states)):
                prev_score = viterbi_matrix[j, t - 1]
                transition_prob = transition_probabilities[states[j]][state]
                scores.append(prev_score * transition_prob * emission_prob)
            max_score_index = np.argmax(scores)
            viterbi_matrix[i, t] = np.max(scores)
            backpointer_matrix[i, t] = max_score_index

    # Backtrack to find the best path
    best_path_indices = [np.argmax(viterbi_matrix[:, -1])]
    for t in range(T - 2, -1, -1):
        best_path_indices.append(backpointer_matrix[best_path_indices[-1], t])
    best_path_indices.reverse()

    best_path = [states[i] for i in best_path_indices]
    return best_path

def hmm_tagger_util(sent_id, untagged_sentence):
     tagged_sentence=viterbi_algo(untagged_sentence,transition_probabilities,emission_probabilities)
     store_submission(sent_id, tagged_sentence)

for sent_id in tqdm(list(test_data.keys())):
    sent = test_data[sent_id]
    hmm_tagger_util(sent_id, sent)

print(submission)



