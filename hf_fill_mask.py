from transformers import pipeline

mlm = pipeline("fill-mask")

prompt = "The dog <mask> over the fence"

result = mlm(prompt)
print(result)

prompt = "The dog <mask> over the pool"

result = mlm(prompt)
print(result)


prompt = "The horse <mask> over the wall"

result = mlm(prompt)
print(result)
