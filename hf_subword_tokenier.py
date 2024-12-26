from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "I love Chicago always deeply dished pizzas!"

tokens = tokenizer.tokenize(sequence)

print("sequence:", sequence)
print("tokens:", tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token ids:", ids)

decoded_ids = tokenizer.decode(ids)
print("decoded:",decoded_ids)