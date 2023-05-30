#%%
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


#%%
# 训练的语料
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
            ['this', 'is', 'the', 'second', 'sentence'],
            ['yet', 'another', 'sentence'],
            ['one', 'more', 'sentence'],
            ['and', 'the', 'final', 'sentence']]


#%%
# 利用语料训练模型
model = Word2Vec(sentences, window=5,vector_size=100, min_count=1)
#  https://radimrehurek.com/gensim/models/word2vec.html
#  vector_size=100
#  window=5:窗口设置一般是5，而且是左右随机1-5（小于窗口大小）的大小,表示当前词与预测词在一个句子中的最大距离是多少
#  min_count：频率小于min-count的单词则会被忽视

#%%
# 基于2d PCA拟合数据
words = list(model.wv.key_to_index)
X=model.wv[words]
print(X.shape)
pca = PCA(n_components=2)
result = pca.fit_transform(X)



#%%
# 可视化展示
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()


# %%
