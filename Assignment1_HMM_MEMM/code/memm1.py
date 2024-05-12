import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from collections import Counter, defaultdict
import heapq
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
data = [ast.literal_eval(row['tagged_sentence']) for _, row in tqdm(df_train.iterrows())]

df_test = pd.read_csv('test_small.csv')
test_data = {row['id']: ast.literal_eval(row['untagged_sentence']) for _, row in tqdm(df_test.iterrows())}


distinct_tags=[]
distinct_words=[]
word_tags = []
def store_tags():
    global distinct_tags
    global distinct_words
    global word_tags
    #data = [[(token, tag) for token, tag in sentence] for sentence in data]
    distinct_tags = list(set(tag for sentence in data for _, tag in sentence))
    distinct_words = list(set(token for sentence in data for token, _ in sentence))
    for sent in data:
        word_tags.append(('START','START'))
        for words, tag in sent:
            word_tags.extend([(tag, words)])
        word_tags.append(('END','END'))

store_tags()
states=distinct_tags
S = len(states)

tags=[]
for tag, words in word_tags:
    tags.append(tag)
distinct_tags=list(set(tags))
count_tags = {}
for tag, count in Counter(tags).items():
    count_tags[tag] = count

plt.figure(figsize=(12, 6))
plt.xticks(rotation='vertical')
plt.bar(range(len(count_tags)), list(count_tags.values()), align='center')
plt.xticks(range(len(count_tags)), list(count_tags.keys()))
plt.xlabel('Tag')
plt.ylabel('Count')
plt.show()

word_prev_tag_counts = defaultdict(Counter)
transition_counts = defaultdict(Counter)

tag_prev_word_probabilities = defaultdict(dict)

tag_prev_word_counts = defaultdict(dict)

for sentence in data:
    for i in range(len(sentence)):
        
        if i == 0:
            
            if f'{sentence[i][0]}_START' not in word_prev_tag_counts:
                word_prev_tag_counts[f'{sentence[i][0]}_START'] = Counter()
            if f'{sentence[i][1]}_START' not in tag_prev_word_counts:
                tag_prev_word_counts[f'{sentence[i][1]}_START'] = Counter()

            word_prev_tag_counts[f'{sentence[i][0]}_START'][sentence[i][1]] += 1
            tag_prev_word_counts[f'{sentence[i][1]}_START'][sentence[i][0]] += 1
        else:
            if f'{sentence[i][0]}_{sentence[i-1][1]}' not in word_prev_tag_counts:
                word_prev_tag_counts[f'{sentence[i][0]}_{sentence[i-1][1]}'] = Counter()
            if f'{sentence[i][1]}_{sentence[i-1][1]}' not in tag_prev_word_counts:
                tag_prev_word_counts[f'{sentence[i][1]}_{sentence[i-1][1]}'] = Counter()

           
            word_prev_tag_counts[f'{sentence[i][0]}_{sentence[i-1][1]}'][sentence[i][1]] += 1
            tag_prev_word_counts[f'{sentence[i][1]}_{sentence[i-1][1]}'][sentence[i][0]] += 1


    if f'END_{sentence[-1][1]}' not in word_prev_tag_counts:
        word_prev_tag_counts[f'END_{sentence[-1][1]}'] = Counter()
    if f'{sentence[-1][1]}_END' not in tag_prev_word_counts:
        tag_prev_word_counts[f'{sentence[-1][1]}_END'] = Counter()

    word_prev_tag_counts[f'END_{sentence[-1][1]}']['END'] += 1
    tag_prev_word_counts[f'{sentence[-1][1]}_END'][sentence[-1][0]] += 1


# for sentence in data:
#     for i in range(len(sentence)):
#         if(i==0):
#            transition_counts[f'START_{sentence[i][0]}'][sentence[i][1]]+=1
#         else:
#            transition_counts[f'{sentence[i-1][1]}_{sentence[i][0]}'][sentence[i][1]]+=1

word_prev_tag_probabilities = defaultdict(dict)


for tag, word_counts in word_prev_tag_counts.items():
    total_count = sum(word_counts.values())
    for word_prev_tag, count in word_counts.items():
        word_prev_tag_probabilities[tag][word_prev_tag] = count / total_count
        tag_prev_word_probabilities[f'{word_prev_tag}_{tag.split("_")[1]}'][f'{tag.split("_")[0]}'] = count / total_count


# transition_probabilities = defaultdict(dict)

# for tag, transition_counts_tag in transition_counts.items():
#     total_count = sum(transition_counts_tag.values())
#     for transition, count in transition_counts_tag.items():
#         transition_probabilities[tag][transition] = count / total_count


for prev_tag_curr_tag, counts in tag_prev_word_counts.items():
    if any(count == 1 for count in counts.values()):
        prev_tag, curr_tag = prev_tag_curr_tag.split('_')
        # word_probabilities = word_prev_tag_probabilities.get(prev_tag_curr_tag, {})
        # # Find the word with count 1
        # word_with_count_one = next(word for word, count in word_probabilities.items() if count == 1)
        # # Get the probability of the word with count 1
        # probability_of_word_one = word_probabilities[word_with_count_one]
        # Assign the same probability to the UNK value in tag_prev_word_probabilities for the same previous tag-current tag combination
        tag_prev_word_probabilities[f'{prev_tag}_{curr_tag}']['UNK'] = 1


def viterbi_algo(sentence):
    T = len(sentence)
    #S = len(states)
    
    viterbi = defaultdict(dict) 
    backpointer = defaultdict(dict) 
    
    for state in states:
            viterbi[state][0] = word_prev_tag_probabilities[f'{sentence[0]}_START'].get(state,
                                    tag_prev_word_probabilities[f'START_{state}'].get('UNK',0)                 
                                                                                        )

            backpointer[state][0] = 'START'
    
    for t in range(1, T):
        for state in states:
                scores = {
                    prev_state: viterbi[prev_state][t - 1] * word_prev_tag_probabilities[f'{sentence[t]}_{prev_state}'].get(state,
                                                                          tag_prev_word_probabilities[f'{state}_{prev_state}'].get('UNK',0)                                                   
                                                                                                                            )
                    for prev_state in states
                }
                backpointer[state][t] = max(scores, key=scores.get) 
                viterbi[state][t] = scores[backpointer[state][t]]
    
    scores = {state: viterbi[state][T - 1] * word_prev_tag_probabilities[f'END_{state}'].get('END',
                                                      tag_prev_word_probabilities[f'END_{state}'].get('UNK',0)                                          
                                                                                             ) for state in states}
    best_final_state = max(scores, key=scores.get)
    best_path = [best_final_state]
    for t in range(T - 1, 0, -1):
        best_final_state = backpointer[best_final_state][t]
        best_path.insert(0, best_final_state)
    
    return best_path


def memm_tagger_util(sent_id, untagged_sentence):
    tagged_tags = viterbi_algo(untagged_sentence)
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