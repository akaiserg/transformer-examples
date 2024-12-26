from transformers import pipeline

pipe = pipeline("fill-mask", model="bert-base-uncased")


#PART 1
movie_desc = "The main characters of the movie matrix are Neo, Morpheus and Trinity"
prompt = "The movie is about [MASK]"
input_text = movie_desc + prompt
output = pipe(input_text)
print(output)
for element in output:
    print(f"Token {element['token_str']}: {element['score']:.3f}%")