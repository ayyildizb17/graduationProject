import pandas as pd
from gensim.models import Word2Vec

# Reading the data
data = pd.read_csv('studentInfo.csv')

# Selected columns of the data
selected_data = data[['code_module', 'code_presentation', 'id_student','gender','region',
                      'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts',
                      'studied_credits', 'disability', 'final_result']]

# Creating sentences for Word2Vec
# sentences = selected_data.groupby('id_student').apply(lambda x: x.values.tolist()).tolist()
sentences = None

# Creating the Word2Vec model
# Window is the number of words that algorithm looks before or after the word
# Min_count is the minimum number of words that should be in a sentence
# Workers are the number of CPU cores that is used for training
model = Word2Vec(window=5, min_count=2, workers=4)

# Building a vocabulary from a sequence of sentences
model.build_vocab(sentences, progress_per=1000)

# Training the model, epochs number is 5 as default
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

# Saving the model
model.save('model')