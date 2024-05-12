import numpy as np
from tqdm import tqdm
import pandas as pd
import ast
from collections import Counter, defaultdict


def good_turing_smoothing(counts):
    """
    Apply Good-Turing smoothing to the given counts.
    """
    N = sum(counts.values())
    smoothed_counts = defaultdict(int)
    
    # Compute counts of counts
    counts_of_counts = Counter(counts.values())
    
    # Calculate the discounting factor
    r_star = {}
    for r in range(1, max(counts_of_counts) + 1):
        if r in counts_of_counts:
            r_star[r] = (r + 1) * counts_of_counts[r + 1] / counts_of_counts[r]
        else:
            r_star[r] = r
    
    # Apply smoothing
    for key, count in counts.items():
        if count in r_star:
            smoothed_counts[key] = r_star[count] / N
        else:
            smoothed_counts[key] = count / N
    
    return smoothed_counts

epsilon = 0.01

df_train = pd.read_csv('train.csv')
data = [ast.literal_eval(row['tagged_sentence']) for _, row in tqdm(df_train.iterrows())]

df_test = pd.read_csv('test_small.csv')
test_data = {row['id']: ast.literal_eval(row['untagged_sentence']) for _, row in tqdm(df_test.iterrows())}


data = [[(token, tag) for token, tag in sentence] for sentence in data]
distinct_tags = list(set(tag for sentence in data for _, tag in sentence))
distinct_words = list(set(token for sentence in data for token, _ in sentence))

transition_counts = defaultdict(lambda: defaultdict(int))
for sentence in data:
    transition_counts['START'][sentence[0][1]] += 1
    transition_counts[sentence[-1][1]]['END'] += 1
    
    for i in range(len(sentence) - 1):
        transition_counts[sentence[i][1]][sentence[i + 1][1]] += 1

# transition_probabilities = defaultdict(dict)
# for source, targets in transition_counts.items():
#     total_count = sum(targets.values()) + epsilon * len(distinct_tags)
#     for target, count in targets.items():
#         transition_probabilities[source][target] = (count + epsilon )/ (total_count)

emission_counts = defaultdict(lambda: defaultdict(int))
for sentence in data:
    for token, tag in sentence:
        emission_counts[tag][token] += 1

#min_word_frequency = 100

# emission_probabilities = defaultdict(dict)
# for tag, emissions in emission_counts.items():
#     total_count = sum(emissions.values()) + epsilon * len(distinct_tags)
#     for emission, count in emissions.items():
#         emission_probabilities[tag][emission] =  (count + epsilon )/ (total_count)
        
smoothed_transition_counts = {source: good_turing_smoothing(targets) for source, targets in transition_counts.items()}
smoothed_emission_counts = {tag: good_turing_smoothing(emissions) for tag, emissions in emission_counts.items()}

transition_probabilities = defaultdict(dict)
for source, targets in smoothed_transition_counts.items():
    total_count = sum(targets.values())
    for target, count in targets.items():
        transition_probabilities[source][target] = count / total_count

emission_probabilities = defaultdict(dict)
for tag, emissions in smoothed_emission_counts.items():
    total_count = sum(emissions.values())
    for emission, count in emissions.items():
        emission_probabilities[tag][emission] = count / total_count

for tag in emission_counts:
        emission_probabilities[tag]['<UNK>'] = min(emission_probabilities[tag].values())/10

distinct_words_set = set(distinct_words)

def viterbi_algo(sentence, transition_probs, emission_probs):
    T = len(sentence)
    states = list(transition_probs.keys())
    S = len(states)
    
    viterbi = defaultdict(dict) 
    backpointer = defaultdict(dict) 
    
    for state in states:
        if sentence[0] not in distinct_words_set:
            viterbi[state][0] = transition_probs['START'].get(state, 0) * emission_probs[state].get(sentence[0],emission_probs[state].get('<UNK>',0))
            backpointer[state][0] = 'START'
        else :
            viterbi[state][0] = transition_probs['START'].get(state, 0) * emission_probs[state].get(sentence[0], 0)
            backpointer[state][0] = 'START'
    
    for t in range(1, T):
        for state in states:
            if sentence[t] not in distinct_words_set:
                scores = {
                    prev_state: viterbi[prev_state][t - 1] * transition_probs[prev_state].get(state, 0) * emission_probs[state].get(sentence[t], emission_probs[state].get('<UNK>',0))
                    for prev_state in states
                }
            else :
                scores = {
                    prev_state: viterbi[prev_state][t - 1] * transition_probs[prev_state].get(state, 0) * emission_probs[state].get(sentence[t], 0)
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


submission = {'id': [], 'tagged_sentence': []}
for sent_id, sentence in tqdm(test_data.items()):
    tagged_tags = viterbi_algo(sentence, transition_probabilities, emission_probabilities)
    tagged_sentence = [(word, tag.upper()) for word, tag in zip(sentence, tagged_tags)]
    submission['id'].append(sent_id)
    submission['tagged_sentence'].append(tagged_sentence)

pd.DataFrame(submission).to_csv('submission.csv', index=False)
