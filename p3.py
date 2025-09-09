import math

# ---------------------------
# Dataset (embedded)
# ---------------------------
data = [
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','No'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes'],
    ['Overcast','Hot','High','Weak','Cool','Same','Yes'],
    ['Rainy','Warm','Normal','Weak','Warm','Change','No'],
    ['Sunny','Hot','Normal','Strong','Warm','Same','Yes'],
    ['Rainy','Warm','High','Strong','Cool','Change','No'],
    ['Overcast','Warm','Normal','Weak','Warm','Same','Yes'],
    ['Sunny','Cold','Normal','Weak','Cool','Same','Yes'],
    ['Rainy','Hot','High','Strong','Warm','Change','No'],
    ['Sunny','Hot','High','Weak','Warm','Same','No'],
    ['Overcast','Cold','Normal','Strong','Cool','Same','Yes'],
    ['Rainy','Warm','High','Weak','Cool','Change','No'],
    ['Sunny','Hot','Normal','Strong','Cool','Same','Yes']
]

# Feature names and target
features = ['Outlook','Temperature','Humidity','Wind','Water','Forecast']
target = 'EnjoySport'

# ---------------------------
# Helper functions
# ---------------------------
def entropy(rows):
    labels = [row[-1] for row in rows]
    total = len(labels)
    return sum([-labels.count(c)/total * math.log2(labels.count(c)/total) for c in set(labels)])

def info_gain(rows, col):
    total_entropy = entropy(rows)
    values = set([row[col] for row in rows])
    subset_entropy = 0
    for v in values:
        subset = [row for row in rows if row[col] == v]
        subset_entropy += len(subset)/len(rows) * entropy(subset)
    return total_entropy - subset_entropy

def id3(rows, feats):
    labels = [row[-1] for row in rows]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if not feats:
        return max(set(labels), key=labels.count)

    gains = [info_gain(rows, i) for i in range(len(feats))]
    best = gains.index(max(gains))
    tree = {feats[best]: {}}

    values = set([row[best] for row in rows])
    for v in values:
        subset = [row for row in rows if row[best] == v]
        remaining_feats = feats[:best] + feats[best+1:]
        tree[feats[best]][v] = id3(subset, remaining_feats)
    return tree

def classify(tree, feats, sample):
    if isinstance(tree, str):
        return tree
    root = next(iter(tree))
    value = sample[feats.index(root)]
    branch = tree[root].get(value)
    if not branch:
        return "Unknown"
    return classify(branch, [f for f in feats if f != root], sample)

# ---------------------------
# Build tree and classify
# ---------------------------
tree = id3(data, features)
print("Decision Tree:", tree)

new_sample = ['Sunny','Hot','Normal','Strong','Cool','Same']
print("New Sample:", new_sample)
print("Prediction:", classify(tree, features, new_sample))
