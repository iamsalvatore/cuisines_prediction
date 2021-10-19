import nltk


hypothesis = ['olive oil','onion','parnship']
# hypothesis_1 = ['chicken','cinnamon']


reference = ['carrot','chickpea','cumin','egg','garlic','honey','lemon','lemon_juice','olive oil','olive oil','onion','parnship']

references = [reference]
list_of_references = [references] # list of references for all sentences in corpus.
list_of_hypotheses = [hypothesis]
#there may be several references
BLEUscore_sentence= nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)
print(BLEUscore_sentence)

BLEUscore_corpus = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore_corpus)

