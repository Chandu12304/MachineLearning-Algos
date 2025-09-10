#finds
import pandas as pd

def findS(data):
    h=["0"]*(len(data[0])-1)
    step=1
    print(f"initial hypothesis:\n {h}\n")
    
    for row in data:
        example=row[:-1]
        label=row[-1]
        if label=="yes":
            if step==1:
                h=example
            else:
                for i in range(len(h)):
                    if h[i]!=example[i]:
                        h[i]="?"
            print(f"step {step}: positive exxample , hence updated hypothesis:\n {h}\n")
        else:
            print(f"step {step}: negative example , hence skipped\n")
        step+=1

data=pd.read_csv("train.csv").values.tolist()
findS(data)
