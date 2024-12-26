from transformers import pipeline

sentence = "I Love Chicago Deep Dished Pizza"

token_classifier = pipeline("token-classification")

tokens = token_classifier(sentence)

print("Sentence:", sentence)

print("Tokens:", tokens)