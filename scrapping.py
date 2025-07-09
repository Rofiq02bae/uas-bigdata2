# !pip install google-play-scraper streamlit wordcloud nltk seaborn matplotlib pandas numpy

from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

nltk.download('stopwords')

# ===================== SCRAPING FUNGSI (optional) =====================
def scrape_reviews():
    result, _ = reviews(
        'com.shopee.id',
        lang='id',
        country='id',
        sort=Sort.MOST_RELEVANT,
        count=1000
    )
    df = pd.DataFrame(np.array(result), columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    df = df[['userName', 'score', 'at', 'content']]
    return df

# ===================== LOAD CSV STATIS =====================
@st.cache_data
def load_data():
    return pd.read_csv("ulasan_shopee_static.csv")

df = load_data()

# ===================== OPTIONAL FLEXIBILITAS (dikomentari) =====================
# if st.button("🔄 Ambil Data Terbaru dan Simpan CSV"):
#     df = scrape_reviews()
#     df.to_csv("ulasan_shopee_static.csv", index=False)
#     st.experimental_rerun()

# ===================== PREPROCESS =====================
df['content'] = df['content'].astype(str)
df['CaseFolding'] = df['content'].str.lower()

stop_words = set(stopwords.words('indonesian') + ['yg', 'dg', 'rt'])
def clean_text(text):
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

df['CleanedText'] = df['CaseFolding'].apply(clean_text)

def label_sentiment(score):
    if score in [1, 2]:
        return 'Negatif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Positif'

df['Sentimen'] = df['score'].apply(label_sentiment)

# ===================== TAB LAYOUT =====================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🗂️ Data Ulasan", "📝 Kesimpulan"])

# ===================== TAB 1: DASHBOARD =====================
with tab1:
    st.title("📱 Dashboard Analisis Ulasan Shopee")

    st.subheader("Distribusi Sentimen")
    st.bar_chart(df['Sentimen'].value_counts())

    st.subheader("Wordcloud dari Ulasan (Tanpa Stopwords)")
    all_text = " ".join(df['CleanedText'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader("20 Kata Paling Umum dalam Ulasan")
    words = " ".join(df['CleanedText']).split()
    most_common = Counter(words).most_common(20)
    common_df = pd.DataFrame(most_common, columns=['Kata', 'Frekuensi'])
    fig2, ax2 = plt.subplots()
    sns.barplot(data=common_df, x='Frekuensi', y='Kata', ax=ax2)
    st.pyplot(fig2)

# ===================== TAB 2: DATA ULASAN =====================
with tab2:
    st.title("🗂️ Data Ulasan Asli Shopee")
    st.dataframe(df[['userName', 'content', 'score', 'Sentimen']].head(50))

    selected = st.selectbox("Filter berdasarkan sentimen", df['Sentimen'].unique())
    filtered = df[df['Sentimen'] == selected]
    st.write(f"Menampilkan {len(filtered)} ulasan dengan sentimen '{selected}'")
    st.dataframe(filtered[['userName', 'content', 'score', 'Sentimen']].head(20))

# ===================== TAB 3: KESIMPULAN =====================
with tab3:
    st.title("📝 Kesimpulan dan Penjelasan Proyek")

    st.markdown("""
    ### 📌 Tujuan Proyek
    Proyek ini bertujuan untuk **mengambil dan menganalisis ulasan aplikasi Shopee** dari Google Play Store secara otomatis, lalu menyajikan visualisasi interaktif dalam bentuk dashboard.

    ### 🔧 Langkah-langkah yang Dilakukan:
    1. **Scraping ulasan Shopee** secara langsung menggunakan `google-play-scraper`.
    2. **Preprocessing ringan** dilakukan dengan:
       - Case folding (mengubah ke huruf kecil)
       - Stopword removal (menghapus kata umum seperti "dan", "yang", dsb.)
    3. **Label sentimen otomatis** berdasarkan skor:
       - Skor 1–2: Negatif
       - Skor 3: Netral
       - Skor 4–5: Positif
    4. Menyajikan hasil berupa:
       - Distribusi sentimen
       - Wordcloud
       - 20 kata terbanyak
       - Tabel ulasan

    ### 🎯 Kesimpulan
    - Sebagian besar ulasan pengguna Shopee menunjukkan **sentimen positif**, meskipun masih ada kritik dari pengguna.
    - Kata-kata yang sering muncul mencerminkan fokus pada **pengiriman, harga, dan kualitas produk**.
    - Dashboard ini bisa digunakan untuk **monitoring opini pengguna secara berkala**, atau pengambilan keputusan berbasis data customer feedback.

    ### 🧠 Saran Pengembangan Lanjutan
    - Gunakan **model sentimen machine learning** (BERT, Naive Bayes, dll) untuk klasifikasi otomatis.
    - Tambahkan filter **berdasarkan tanggal** atau fitur pencarian kata kunci.
    - Simpan histori scraping ke database untuk analisis tren jangka panjang.
    """)

