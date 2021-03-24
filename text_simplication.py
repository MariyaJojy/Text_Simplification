import stanfordnlp
import pandas as pd

from pandas import DataFrame

#importing text data
df = pd.read_csv('dat.csv')
text=df.values.tolist()
sen_type=[]
final=[]


nlp = stanfordnlp.Pipeline()
for sent in text:
    doc=nlp(sent[0])
    a  = doc.sentences[0]
    d=a.dependencies
    cnt=0   #for counting nsubj or nsubjpass types
    sd=[]
    v=[]    #stores nsubj/nsubjpass type words

    for i in d:
        #removing unwanted relations
        if i[1] not in ["acl", "advcl", "appos", "ccomp", "conj", "dep",  "mark", "parataxis", "ref"]:
            sd.append([i[1], i[0].text, i[0].index, i[2].text, i[2].index])
        if(i[1]=="nsubj" or i[1]=="nsubjpass"):
            v.append( [i[0].text, i[2].text])

            cnt=cnt+1

    if cnt>1:
        sen_type.append("complex/compound")
    else:
        sen_type.append("simple")

    #function for sorting
    def ord(elem):

        return int(elem[1])

    #identifying splits
    s=""
    for k in v:
        arr = []
        for j in sd:
            if j[1] == k[0] or j[1] == k[1] or j[3] == k[0] or j[3] == k[1]:
                arr.append([j[1], j[2]])
                arr.append([j[3], j[4]])

        arr.sort(key=ord)
        for i in range(1, int(arr[-1][1]) + 1):
            c = 0
            for j in arr:

                if int(j[1]) == i and c < 1:
                    c = c + 1
                    if j[0] != None and j[0] != ",":
                        s = s + " " + j[0]
        s=s+"/"         #separating splits by "/"
    final.append(s)

#converting to dataframe and then csv
d = { 'Sentence': text, 'Type':sen_type, 'splits': final }
frame = pd.DataFrame(d, columns=['Sentence', 'Type','splits'] )
frame.to_csv('result.csv')
