import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os

print("1. Veri seti yükleniyor...")
try:
    df = pd.read_csv('enronSpamSubset.csv')
   
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna(subset=['Body']) 
    
except FileNotFoundError:
    print("HATA: 'enronSpamSubset.csv' dosyası bulunamadı. Lütfen dosya adını kontrol et.")
    df = pd.DataFrame() 

if not df.empty:
    
    print("2. Temizlik (Regex ve Stopwords) yapılıyor...")

   
    stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", 
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", 
        "subject", "re", "fw", "http", "com", "www", "html", "image", "file", "message"
    ])

    def clean_text(text):
        text = str(text).lower()                            
        text = re.sub(r'<.*?>', '', text)                   
        text = re.sub(r'http\S+|www\.\S+', '', text)        
        text = re.sub(r'\S+@\S+', '', text)                 
        text = re.sub(r'[^a-z\s]', '', text)               
        
        words = text.split()
        
        clean_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        return " ".join(clean_words)

    
    df['Cleaned_Body'] = df['Body'].apply(clean_text)

    df['Cleaned_Body'].replace('', np.nan, inplace=True)
    df = df.dropna(subset=['Cleaned_Body'])

    
    print("3. Veri Analizi Grafikleri hazırlanıyor...")
    
    
    final_df = df[['Cleaned_Body', 'Label']]

    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.countplot(x='Label', data=final_df, ax=axes[0], palette="viridis")
    axes[0].set_title('Spam ve Normal Mail Sayıları')
    axes[0].set_xlabel('0: Normal | 1: Spam')
    axes[0].set_ylabel('Adet')

    all_words = " ".join(final_df['Cleaned_Body']).split()
    word_counts = Counter(all_words)
    common_words_df = pd.DataFrame(word_counts.most_common(15), columns=['Kelime', 'Frekans'])

    sns.barplot(x='Frekans', y='Kelime', data=common_words_df, ax=axes[1], palette="magma")
    axes[1].set_title('En Çok Geçen Kelimeler')

    plt.tight_layout()
    plt.show()

    output_filename = 'cleaned_emails.csv'
    final_df.to_csv(output_filename, index=False)

    
    current_directory = os.getcwd()
    full_path = os.path.join(current_directory, output_filename)

    print("\n" + "="*50)
    print("GÖREV TAMAMLANDI (RUBRİK MADDE E)")
    print("="*50)
    print(f"✅ Dosya Başarıyla Oluşturuldu: {output_filename}")
    print(f"📂 Dosyanın Kaydedildiği Tam Konum:\n👉 {full_path}")
    print("="*50)
    print(f"İçerik: {len(final_df)} satır veri temizlendi ve kaydedildi.")