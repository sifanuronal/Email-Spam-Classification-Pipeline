import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

INPUT_FILE = "data/cleaned_emails.csv"  
OUTPUT_FOLDER = "processed_data/" 
MAX_FEATURES = 3000  


def run_process():
    print("🚀 2. Kişi (Feature Engineering) İşlemi Başladı...")

 
    if not os.path.exists(INPUT_FILE):
        print(f"HATA: {INPUT_FILE} bulunamadı! Dosyayı 'data' klasörüne attın mı?")
        return

    print("📥 Veri yükleniyor...")
    df = pd.read_csv(INPUT_FILE)

    
    df = df.dropna(subset=['Cleaned_Body', 'Label'])

    X_text = df['Cleaned_Body'].astype(str)  
    y = df['Label'].astype(int)  

    print(f"📊 İşlenecek Toplam Mail Sayısı: {len(df)}")


    print("✂️ Veri %80 Eğitim - %20 Test olarak ayrılıyor...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

   
    print("🔢 TF-IDF Matrisi oluşturuluyor...")
    tfidf = TfidfVectorizer(ngram_range=(1, 1))  

    
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    
    X_test_tfidf = tfidf.transform(X_test_raw)

   
    print(f"✨ Chi-Square testi ile en iyi {MAX_FEATURES} özellik seçiliyor...")
    selector = SelectKBest(chi2, k=MAX_FEATURES)

    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = selector.transform(X_test_tfidf)

    
    feature_names = tfidf.get_feature_names_out()
    selected_indices = selector.get_support(indices=True)
    selected_words = feature_names[selected_indices]
    print(f"✅ Seçilen örnek kelimeler: {selected_words[:10]}")

    
    print("💾 Dosyalar 'processed_data' klasörüne kaydediliyor...")

    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    
    np.save(f"{OUTPUT_FOLDER}X_train.npy", X_train_selected.toarray())
    np.save(f"{OUTPUT_FOLDER}X_test.npy", X_test_selected.toarray())
    np.save(f"{OUTPUT_FOLDER}y_train.npy", y_train.values)
    np.save(f"{OUTPUT_FOLDER}y_test.npy", y_test.values)

    
    pd.DataFrame(selected_words, columns=['Selected_Features']).to_csv(f"{OUTPUT_FOLDER}features_list.csv", index=False)

    print("\n🎉 İŞLEM BAŞARIYLA TAMAMLANDI!")
    print("👉 'processed_data' klasöründeki dosyaları 3. kişiye gönderebilirsin.")


if __name__ == "__main__":
    run_process()