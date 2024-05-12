import numpy as np
from tqdm import tqdm
import pandas as pd
import ast
from collections import Counter, defaultdict
import random
import matplotlib.pyplot as plt

epsilon = 1


def good_turing_smoothing(emission_counts, total_count):
    # Calculate the count of counts
    count_of_counts = Counter(emission_counts.values())
    
    # Compute the expected count for unseen words using Good-Turing formula
    max_count = max(emission_counts.values())
    N1 = count_of_counts[1]  # Count of unigrams that appear once
    N = total_count          # Total count of emissions
    r0 = N1 / N              # Probability of unigrams that appear once
    r_star = (1 * (N1 + 1) / N) if N1 > 0 else 0  # Smoothed probability for unseen words
    
    # Apply Good-Turing smoothing to emission probabilities
    emission_probabilities_smoothed = {}
    for word, count in emission_counts.items():
        # Apply Good-Turing smoothing if count is 0
        if count == 0:
            emission_probabilities_smoothed[word] = r_star
        else:
            # Use the original count for words with non-zero count
            emission_probabilities_smoothed[word] = count / N
            
    return emission_probabilities_smoothed

df_train = pd.read_csv('train.csv')
data = [ast.literal_eval(row['tagged_sentence']) for _, row in tqdm(df_train.iterrows())]

df_test = pd.read_csv('test_small.csv')
test_data = {row['id']: ast.literal_eval(row['untagged_sentence']) for _, row in tqdm(df_test.iterrows())}


def display_data(sentence_index):
    '''
        Input : 'sentence_index' (int) -> index of a sentence in training data
        Output: None
    '''
    sentence = data[sentence_index]
    print("TOKEN -> TAG")
    print('...')
    for token, tag in sentence:
        print(token, '>', tag)
sentence_index = random.choice(range(len(data)))
#display_data(sentence_index)
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

transition_counts = defaultdict(lambda: defaultdict(int))
for sentence in data:
    transition_counts['START'][sentence[0][1]] += 1
    transition_counts[sentence[-1][1]]['END'] += 1
    
    for i in range(len(sentence) - 1):
        transition_counts[sentence[i][1]][sentence[i + 1][1]] += 1

transition_probabilities = defaultdict(dict)
for source, targets in transition_counts.items():
    total_count = sum(targets.values()) + epsilon * len(distinct_tags)
    for target, count in targets.items():
        transition_probabilities[source][target] = (count + epsilon )/ (total_count)

emission_counts = defaultdict(lambda: defaultdict(int))
for sentence in data:
    for token, tag in sentence:
        emission_counts[tag][token] += 1

total_count = sum(sum(emissions.values()) for emissions in emission_counts.values())

# Apply Good-Turing smoothing to emission probabilities
emission_probabilities_smoothed = {}
for tag, emissions in emission_counts.items():
    emission_probabilities_smoothed[tag] = good_turing_smoothing(emissions, total_count)

emission_probabilities = defaultdict(dict)
for tag, emissions in emission_counts.items():
    total_count = sum(emissions.values()) + epsilon * len(distinct_tags)
    for emission, count in emissions.items():
        emission_probabilities[tag][emission] =  (count + epsilon )/ (total_count)

#for tag in emission_counts:
 #       emission_probabilities[tag]['<UNK>'] = min(emission_probabilities[tag].values())/10

distinct_words_set = set(distinct_words)

def viterbi_algo(sentence, transition_probs, emission_probs):
    T = len(sentence)
    states = list(transition_probs.keys())
    S = len(states)
    
    viterbi = defaultdict(dict) 
    backpointer = defaultdict(dict) 
    
    for state in states:
        if sentence[0] not in distinct_words_set:
            viterbi[state][0] = transition_probs['START'].get(state, 0) * emission_probabilities_smoothed[state].get(sentence[0],emission_probabilities_smoothed[state].get('<UNK>',0))
            backpointer[state][0] = 'START'
        else :
            viterbi[state][0] = transition_probs['START'].get(state, 0) * emission_probabilities_smoothed[state].get(sentence[0], 0)
            backpointer[state][0] = 'START'
    
    for t in range(1, T):
        for state in states:
            if sentence[t] not in distinct_words_set:
                scores = {
                    prev_state: viterbi[prev_state][t - 1] * transition_probs[prev_state].get(state, 0) * emission_probabilities_smoothed[state].get(sentence[t], emission_probabilities_smoothed[state].get('<UNK>',0))
                    for prev_state in states
                }
            else :
                scores = {
                    prev_state: viterbi[prev_state][t - 1] * transition_probs[prev_state].get(state, 0) * emission_probabilities_smoothed[state].get(sentence[t], 0)
                    for prev_state in states
                }
            backpointer[state][t] = max(scores, key=scores.get) 
            viterbi[state][t] = scores[backpointer[state][t]]
    
    scores = {state: viterbi[state][T - 1] * transition_probs[state].get('END', 0) for state in states}
    best_final_state = max(scores, key=scores.get)
    best_path = [best_final_state]
    for t in range(T - 1, 0, -1):
        best_final_state = backpointer[best_final_state][t]
        best_path.insert(0, best_final_state)
    
    return best_path

def hmm_tagger_util(sent_id, untagged_sentence):
    tagged_tags = viterbi_algo(untagged_sentence, transition_probabilities, emission_probabilities)
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
    hmm_tagger_util(sent_id,sentence)

pd.DataFrame(submission).to_csv('submission.csv', index=False)
