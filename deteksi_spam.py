# ----------- [1. IMPORT LIBRARY] -----------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# ----------- [2. MUAT DATASET] -----------
def load_dataset():
    try:
        if not os.path.exists("data_spam.csv"):
            print("❌ ERROR: File 'data_spam.csv' tidak ditemukan!")
            print("Pastikan file dataset ada di folder yang sama.")
            exit()
        
        df = pd.read_csv("data_spam.csv")
        df["panjang"] = df["email"].apply(len)  # Hitung panjang teks
        
        print("\n✅ Dataset berhasil dimuat!")
        print(f"📊 Total Data: {len(df)} email")
        print(f"🔴 Spam: {sum(df['label'] == 'spam')} | 🟢 Bukan: {sum(df['label'] == 'bukan')}")
        
        return df
    
    except Exception as e:
        print(f"❌ Gagal memuat dataset: {e}")
        exit()

# ----------- [3. VISUALISASI DATA] -----------
def show_visualization(df):
    print("\n📊 Menampilkan Visualisasi Data...")
    
    plt.figure(figsize=(15, 6))
    
    # [1] WordCloud Kata-kata Spam
    plt.subplot(1, 2, 1)
    spam_words = " ".join(df[df["label"] == "spam"]["email"])
    wordcloud = WordCloud(width=600, height=400, background_color="white").generate(spam_words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Kata-kata Kunci dalam Email Spam", pad=20)
    plt.axis("off")
    
    # [2] Histogram Panjang Email (Perbaikan Error)
    plt.subplot(1, 2, 2)
    
    # Pisahkan data
    spam_len = df[df["label"] == "spam"]["panjang"]
    bukan_len = df[df["label"] == "bukan"]["panjang"]
    
    # Plot histogram terpisah
    plt.hist(spam_len, bins=10, alpha=0.7, color="red", label="Spam")
    plt.hist(bukan_len, bins=10, alpha=0.7, color="green", label="Bukan Spam")
    
    plt.title("Perbandingan Panjang Email")
    plt.xlabel("Jumlah Karakter")
    plt.ylabel("Frekuensi")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ----------- [4. TRAINING MODEL] -----------
def train_model(df):
    print("\n🤖 Melatih Model Naive Bayes...")
    
    # Ekstraksi Fitur (TF-IDF)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["email"])
    y = df["label"]
    
    # Model Naive Bayes
    model = MultinomialNB()
    model.fit(X, y)
    
    print("✅ Model berhasil dilatih!")
    return model, vectorizer

# ----------- [5. DETEKSI SPAM INTERAKTIF] -----------
def detect_spam(model, vectorizer):
    print("\n" + "=" * 50)
    print("🔍 MODE DETEKSI SPAM EMAIL")
    print("=" * 50)
    
    while True:
        email = input("\nMasukkan email (atau ketik 'keluar'):\n>>> ")
        
        if email.lower() == "keluar":
            break
        
        # Prediksi
        X_test = vectorizer.transform([email])
        pred = model.predict(X_test)[0]
        proba = model.predict_proba(X_test)[0]
        
        # Tampilkan Hasil
        print("\n" + "=" * 30)
        print(f"📧 Email: {email}")
        
        if pred == "spam":
            print(f"🔴 HASIL: SPAM ({proba[1]*100:.1f}% keyakinan)")
        else:
            print(f"🟢 HASIL: BUKAN SPAM ({proba[0]*100:.1f}% keyakinan)")
        
        print("\n📊 Probabilitas:")
        print(f"• Spam: {proba[1]*100:.2f}%")
        print(f"• Bukan: {proba[0]*100:.2f%}")
        print("=" * 30)

# ----------- [PROGRAM UTAMA] -----------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("📧 APLIKASI DETEKSI EMAIL SPAM")
    print("=" * 50)
    
    # 1. Muat Dataset
    df = load_dataset()
    
    # 2. Tampilkan Visualisasi
    show_visualization(df)
    
    # 3. Training Model
    model, vectorizer = train_model(df)
    
    # 4. Deteksi Interaktif
    detect_spam(model, vectorizer)