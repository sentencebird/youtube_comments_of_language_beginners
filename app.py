import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# TODO: DBに移行
# TODO: 再生回数も調べたい

st.title("Comments of YouTube videos for beginners of language")

langs = ["Japanese", "English", "Chinese", "Russian", "French", "Korean", "Hindi", "German", "Spanish", "Turkish"]
lang = st.selectbox("Language", langs)

with open(f'comments_by_lang/{lang.lower()}.txt', mode="rb") as f:
    text = f.read().decode("utf-8")

wc = WordCloud(background_color='white').generate(text)

fig = plt.figure()
plt.imshow(wc)
st.pyplot(fig)