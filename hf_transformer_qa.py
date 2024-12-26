from transformers import pipeline
import pandas as pd

nlp = pipeline('question-answering')
result = nlp({
 'question':"Do you know my name?",
 'context':"My name is Andres"   
})
print("Result:",result)


print()
