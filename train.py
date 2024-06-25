import os
import time
from model.basic import BasicTokenizer
from model.regex import RegexTokenizer

data_path = 'tok/data/taylorswift.txt'
if not os.path.exists(data_path):
    print(f'Error! Incorrect path to data')

text = open(data_path, "r", encoding="utf-8").read()

models_dir = 'tok/models'
if not os.path.exists(models_dir):
    print(f'Error! Incorrect path to model')

# create a directiory for models
os.makedirs(models_dir, exist_ok = True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ['basic', 'regex']):

    # construct Tokenizer object
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)

    # writes two files to the models directory: name.model and name.vocab
    prefix = os.path.join(models_dir, name)
    tokenizer.save(prefix)
t1 = time.time()

print(f'training took {t1-t0:.2f} seconds')





