import pandas as pd

# Training data
data = pd.DataFrame({
    'Free':    [1, 1, 0, 0],
    'Win':     [1, 1, 0, 0],
    'Hello':   [0, 0, 0, 1],
    'iPhone':   [0, 0, 1, 1,],
    'Label':   ['Spam', 'Spam', 'Not Spam', 'Not Spam']
})
# Get prior probabilities
priors = data['Label'].value_counts(normalize=True)
print("Priors:\n", priors)
# Calculate conditional probabilities for each feature given the label
def conditional_prob(feature, value, label):
    subset = data[data['Label'] == label]
    return (subset[feature] == value).sum() / len(subset)

features = ['Free', 'Win', 'Hello', 'iPhone']
labels = data['Label'].unique()

# Calculate and store probabilities
probs = {}
for label in labels:
    probs[label] = {
        feat: {
            1: conditional_prob(feat, 1, label),
            0: conditional_prob(feat, 0, label)
        }
        for feat in features
    }
# New email
new_email = {'Free': 1, 'Win': 0, 'Hello': 0, 'iPhone': 1}

# Calculate posterior for each label
posterior = {}
for label in labels:
    prob = priors[label]
    for feat in features:
        prob *= probs[label][feat][new_email[feat]]
    posterior[label] = prob

# Choose the label with the highest posterior probability
prediction = max(posterior, key=posterior.get)

print("Posterior probabilities:", posterior)
print("Prediction:", prediction)

