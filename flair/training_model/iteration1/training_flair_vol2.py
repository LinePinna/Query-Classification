"""
created on Mon Apr  1 10:26:47 2019-O
@author: lpinna
"""

from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher#, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List
import os
import setGPU


columns = {0: 'text', 1: 'ner'}

os.chdir("/home/lpinna")

# this is the folder in which train, test and dev files reside
data_folder = '/home/lpinna/classification1/training/vol3'

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                             train_file='train_20190426_v1.csv',
                                                             test_file='test_20190426_v1.csv')

print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
#print(tag_dictionary.idx2item)


# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [


    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    #FlairEmbeddings('news-forward'),
    #FlairEmbeddings('news-backward'),

]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('/home/lpinna/classification1/training/vol3',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
	      embeddings_in_memory=False)
