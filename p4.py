import random, math

class SimpleNN:
    def __init__(self, ni, nh, no):
        self.net = [
            [{'w': [random.uniform(-1, 1) for _ in range(ni + 1)]} for _ in range(nh)],
            [{'w': [random.uniform(-1, 1) for _ in range(nh + 1)]} for _ in range(no)]
        ]

    def sigmoid(self, x): return 1 / (1 + math.exp(-x))

    def forward(self, inp):
        for layer in self.net:
            inp = [self.sigmoid(sum(w * i for w, i in zip(n['w'][:-1], inp)) + n['w'][-1]) for n in layer]
            for n, o in zip(layer, inp): n['o'] = o
        return inp

    def backward(self, exp):
        for i in reversed(range(len(self.net))):
            for j, n in enumerate(self.net[i]):
                out = n['o']
                err = (exp[j] - out) if i == 1 else sum(n2['w'][j] * n2['d'] for n2 in self.net[i + 1])
                n['d'] = err * out * (1 - out)

    def update(self, inp, lr):
        for i, layer in enumerate(self.net):
            inp = [n['o'] for n in self.net[i - 1]] if i > 0 else inp
            for n in layer:
                for j in range(len(inp)): n['w'][j] += lr * n['d'] * inp[j]
                n['w'][-1] += lr * n['d']

    def train(self, data, lr, epochs):
        for _ in range(epochs):
            for row in data:
                o = self.forward(row[:-1])
                self.backward(row[-1])
                self.update(row[:-1], lr)

    def predict(self, x): return [round(o) for o in self.forward(x)]

# === Training ===
data = [[0,0,[0]],[0,1,[1]],[1,0,[1]],[1,1,[0]],[0.1,0.9,[1]],[0.9,0.1,[1]],
        [0.2,0.2,[0]],[0.5,0.5,[0]],[0.7,0.3,[1]],[0.3,0.7,[1]]]

model = SimpleNN(2, 2, 1)
model.train(data[:4], 0.1, 10000)

# === Testing ===
print("\nTest Results:")
acc = 0
for d in data:
    p = model.predict(d[:-1])
    acc += p == d[-1]
    print(f"In={d[:-1]} Exp={d[-1]} Got={p} {'✅' if p==d[-1] else '❌'}")
print(f"Accuracy: {acc/len(data)*100:.1f}%")