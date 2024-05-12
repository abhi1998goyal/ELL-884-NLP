import numpy as np
from tqdm import tqdm
import pandas as pd
import ast
from collections import Counter, defaultdict

# Load training data
df_train = pd.read_csv('train.csv')
data = [ast.literal_eval(row['tagged_sentence']) for _, row in tqdm(df_train.iterrows())]

# Load test data
df_test = pd.read_csv('test2.csv')
test_data = {row['id']: ast.literal_eval(row['untagged_sentence']) for _, row in tqdm(df_test.iterrows())}

# Preprocess data
data = [[(token.lower(), tag.lower()) for token, tag in sentence] for sentence in data]
distinct_tags = list(set(tag for sentence in data for _, tag in sentence))
distinct_words = list(set(token for sentence in data for token, _ in sentence))

# Compute transition probabilities
transition_counts = defaultdict(lambda: defaultdict(int))
for sentence in data:
    for i in range(len(sentence) - 1):
        transition_counts[sentence[i][1]][sentence[i + 1][1]] += 1

transition_probabilities = defaultdict(dict)
for source, targets in transition_counts.items():
    total_count = sum(targets.values())
    for target, count in targets.items():
        transition_probabilities[source][target] = count / total_count

# Compute emission probabilities
emission_counts = defaultdict(lambda: defaultdict(int))
for sentence in data:
    for token, tag in sentence:
        emission_counts[tag][token] += 1

emission_probabilities = defaultdict(dict)
for tag, emissions in emission_counts.items():
    total_count = sum(emissions.values())
    for emission, count in emissions.items():
        emission_probabilities[tag][emission] = count / total_count

# Define Viterbi algorithm
def viterbi_algo(sentence, transition_probs, emission_probs):
    T = len(sentence)
    states = list(transition_probs.keys())
    S = len(states)
    
    viterbi = np.zeros((S, T))
    backpointer = np.zeros((S, T), dtype=int)
    
    # Initialization
    for i, state in enumerate(states):
        viterbi[i, 0] = transition_probs['START'].get(state, 0) * emission_probs[state].get(sentence[0], 0)
    
    # Recursion
    for t in range(1, T):
        for i, state in enumerate(states):
            scores = [
                viterbi[j, t - 1] * transition_probs[prev_state].get(state, 0) * emission_probs[state].get(sentence[t], 0)
                for j, prev_state in enumerate(states)
            ]
            backpointer[i, t] = np.argmax(scores)
            viterbi[i, t] = np.max(scores)
    
    print(viterbi)
    # Termination
    best_path_idx = np.argmax(viterbi[:, -1])
    best_path = [states[best_path_idx]]
    for t in range(T - 1, 0, -1):
        best_path_idx = backpointer[best_path_idx, t]
        best_path.insert(0, states[best_path_idx])
    
    return best_path

# Tag untagged sentences using Viterbi algorithm
submission = {'id': [], 'tagged_sentence': []}
for sent_id, sentence in tqdm(test_data.items()):
    tagged_sentence = viterbi_algo(sentence, transition_probabilities, emission_probabilities)
    submission['id'].append(sent_id)
    submission['tagged_sentence'].append(tagged_sentence)

# Save submission to CSV
pd.DataFrame(submission).to_csv('submission.csv', index=False)
