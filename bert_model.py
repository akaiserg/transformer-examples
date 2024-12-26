from transformers import BertConfig, BertModel

config = BertConfig()

model = BertModel(config=config)
print(config)