import streamlit as st
import wordcloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# TODO: DBに移行
# TODO: 再生回数も調べたい

st.title("YouTube外国語学習動画のコメント分析")

langs = ["Japanese", "English", "Chinese", "Russian", "French", "Korean", "Hindi", "German", "Spanish", "Turkish"]

corpus = []
for lang in langs:
    with open(f'comments_by_lang/{lang.lower()}.txt', mode="rb") as f:
        text = f.read().decode("utf-8")
    corpus.append(text)

tfidf = TfidfVectorizer(stop_words=wordcloud.STOPWORDS)
X = tfidf.fit_transform(corpus)

lang = st.selectbox("言語", langs)

with open(f'comments_by_lang/{lang.lower()}.txt', mode="rb") as f:
    text = f.read().decode("utf-8")

wc = wordcloud.WordCloud(background_color='white').generate(text)

fig = plt.figure()
plt.imshow(wc)
st.pyplot(fig)

tfidf_index = langs.index(lang)
top_word_indices = np.argsort(X.toarray())[:, -50:-1][tfidf_index]

st.table(pd.DataFrame({"特徴語": [tfidf.get_feature_names()[i] for i in top_word_indices[::-1]]}))