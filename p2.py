import pandas as pd

def candidate_el(data):
    S=["0"]*(len(data[0])-1)
    G=[["?"]*(len(data[0])-1)]
    step=1
    for row in data:
        example=row[:-1]
        label=row[-1]
        if label=='yes':
            if step==1:
                S=example
            else:
                for i in range(len(S)):
                    if S[i]!=example[i]:
                        S[i]="?"
                new_G=[]
                for g in G:
                    ok=True
                    for i in range(len(g)):
                        if not (g[i]=='?' or g[i]==example[i]):
                                ok=False
                                break
                    if ok:
                        new_G.append(g)
                G=new_G
                
        else:
            new_G=[]
            for g in G:
                for i in range(len(g)):
                    if g[i]=='?':
                        for value in domain[i]:
                            if example[i]!=value:
                                new_g=g.copy()
                                new_g[i]=value
                                if new_g not in new_G:
                                    new_G.append(new_g)
            G=new_G
        step+=1
    return S,G


df=pd.read_csv("train.csv")
data=df.values.tolist()

domain=[list(df[col].unique()) for col in df.columns[:-1]]

S,G=candidate_el(data)
print("\n==== Result ====\n")
print(f"Final S :\n{S}\n")
print(f"Final G:\n{G}\n")
