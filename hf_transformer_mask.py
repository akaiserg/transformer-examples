from transformers import AutoTokenizer, XLNetTokenizer

sentence = "I love Chicago deep dish           pizza don't you?"

# instance BERT-based tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#invoke pre-tokenizer
result1 = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)

print("Tokenizer BERT")
print(f"Sentence {sentence}")
print(f"tokenized: {result1}")

# instance GPT2-based tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

result2 = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)

print("Tokenizer GPT2")
print(f"Sentence {sentence}")
print(f"tokenized: {result2}")

# instance T4-based tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

result3 = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)

print("Tokenizer T5-based")
print(f"Sentence {sentence}")
print(f"tokenized: {result3}")

# instance XLNetTokenizer-based
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
result4 = tokenizer.tokenize(sentence)

print("Tokenizer XLNetTokenizer-based")
print(f"Sentence {sentence}")
print(f"tokenized: {result4}")