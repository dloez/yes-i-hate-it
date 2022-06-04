from gensim.models import Word2Vec


sentences = [['me', 'gusta', 'el', 'futbol'], ['vaya', 'dia', 'de', 'mierda', 'que', 'llevo']]

model =  Word2Vec(min_count=1, epochs=1, vector_size=100)
# gensim.models.Word2Vec(sentences, iter=100, size=200, workers = 4)
model.build_vocab(sentences)
# model.train(data)

# v1 = model.wv['futbol']

sim = model.wv.most_similar('futbol')

print(sim)


# print(f"futbol-pie: {model1.wv.similarity('futbol', 'pie')}")
# print(f"balon-pie: {model.accuracy('balon pie')}")
