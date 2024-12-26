from transformers import pipeline

translator = pipeline("translation_en_to_fr")

input = "Today the weather was really nice and sunny"

result = translator(input, clean_up_tokenization_spaces=True, min_length=50)
print(result)