import pandas as pd

def findS(data):
    h = ["0"] * (len(data[0]) - 1)
    step = 1
    print(f"initial hypothesis: {h}")

    for row in data:
        if row[-1].strip().lower() == "yes":
            print(f"\nStep {step}: +ve example: {row[-1]}")
            if step==1:
                h = row[:-1]
                print(f"\nFirst +ve example, setting hypothesis to: {h}")
            else:
                for i in range(len(h)):
                    if h[i] != row[i]:
                        h[i] = "?"
                print(f"\nUpdated Hypothesis: {h}")
        else:
            print(f"\nStep {step}: Skipping -ve example: {row[-1]}")   
        step += 1
    return h

df = pd.read_csv("train.csv")

data = df.values.tolist()

findS(data)
