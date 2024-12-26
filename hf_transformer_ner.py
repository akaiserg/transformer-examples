from transformers import pipeline
import pandas as pd

nlp = pipeline('ner')
result = nlp("I am UCSC instructor and  my name is Oswald")
print("Result:",result)

df = pd.DataFrame(result)
print("df",df)
print()

nlp = pipeline('ner')
result = nlp("I Love Chicago Pizza and I Really Like New York Bagels")
print("Result:",result)

df = pd.DataFrame(result)
print("df",df)

