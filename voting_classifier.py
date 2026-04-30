import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

try:
   
    required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        raise FileNotFoundError(f"Eksik dosyalar: {missing_files}")

    X_train = np.load('X_train.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True)
    y_test = np.load('y_test.npy', allow_pickle=True)
    
    print(f"✅ Veriler yüklendi.")
    print(f"Eğitim Seti: {X_train.shape}")
    print(f"Test Seti: {X_test.shape}")

except FileNotFoundError as e:
    print(f"❌ HATA: {e}")
    print("Lütfen dosyaları çalışma dizinine yüklediğinizden emin olun.")
    
    exit()

clf1 = RandomForestClassifier(n_estimators=300, min_samples_split=5, random_state=42, n_jobs=-1)
clf2 = SVC(kernel='linear', probability=True, random_state=42)
clf3 = LogisticRegression(random_state=42, n_jobs=-1)


voting_model = VotingClassifier(
    estimators=[
        ('rf', clf1), 
        ('svm', clf2), 
        ('lr', clf3)
    ],
    voting='soft', 
    n_jobs=-1
)

models = {
    "Random Forest (Tekil)": clf1,
    "Voting Classifier (Ortak Akıl)": voting_model
}

results = []
plt.figure(figsize=(15, 6)) 


print("\nModeller eğitiliyor, lütfen bekleyin...\n")

for i, (name, model) in enumerate(models.items()):
    print(f"➡️ {name} eğitiliyor...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0, average='weighted'),
        "Recall": recall_score(y_test, y_pred, zero_division=0, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0, average='weighted')
    })

    
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(1, len(models), i+1) # <-- OTOMATİK AYARLANAN GRAFİK SAYISI
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name}")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")

plt.tight_layout()
plt.show()


results_df = pd.DataFrame(results)
print("\n--- MODEL SONUÇLARI ---")
print(results_df)


best_model_row = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]
print(f"\n🏆 En iyi model seçildi: {best_model_name} (F1: {best_model_row['F1-Score']:.4f})")


param_grids = {
    "Naive Bayes": {'alpha': [0.1, 0.5, 1.0, 2.0]},
    
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear']},
    
    "Random Forest": { 
        'n_estimators': [200, 300, 500],
        'max_depth': [None, 40, 60],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    },
    
    "Logistic Regression": {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
}


final_model = models[best_model_name] 

if "Voting" not in best_model_name: 
    if "Random Forest" in best_model_name: 
        search_key = "Random Forest"
    elif "SVM" in best_model_name:
        search_key = "SVM"
    else:
        search_key = best_model_name

    if search_key in param_grids:
        print(f"\n⚙️ {best_model_name} için en iyi parametreler aranıyor (Bu işlem biraz sürebilir)...")
        base_model = models[best_model_name]
        
        
        grid_search = GridSearchCV(base_model, param_grids[search_key], cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"✅ En iyi parametreler: {grid_search.best_params_}")
        final_model = grid_search.best_estimator_
    else:
        print(f"\nℹ️ {best_model_name} için grid search tanımlı değil, varsayılan model kullanılıyor.")
else:
    print(f"\nℹ️ Voting Classifier seçildiği için hiperparametre araması atlanıyor (Zaten optimize edilmiş modeller kullanılıyor).")



joblib.dump(final_model, 'best_model.joblib')
results_df.to_csv('results_table.csv', index=False)
print("💾 Model ve sonuçlar kaydedildi.")


try:
    features_df = pd.read_csv('features_list.csv')

    feature_names = features_df.iloc[:, 0].values 
    
    importances = None
    
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
    elif hasattr(final_model, 'coef_'):
        
        importances = np.abs(final_model.coef_[0])
    elif isinstance(final_model, VotingClassifier):
        
        print("ℹ️ Voting Classifier için doğrudan özellik önemi çizilemiyor.")
        
        if hasattr(clf1, 'feature_importances_'):
             print("ℹ️ Referans olarak Random Forest bileşeninin özellik önemleri gösteriliyor.")
             importances = clf1.feature_importances_

    if importances is not None:
        
        if len(importances) != len(feature_names):
            print(f"⚠️ Uyarı: Özellik sayısı uyuşmazlığı (Model: {len(importances)} vs Liste: {len(feature_names)}).")
            
        else:
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })

            top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
            plt.title(f'En Önemli 20 Kelime (Model: {best_model_name} veya Bileşeni)')
            plt.tight_layout()
            plt.show()

except FileNotFoundError:
    print("⚠️ 'features_list.csv' bulunamadığı için özellik analizi yapılamadı.")
except Exception as e:
    print(f"Özellik analizi sırasında bir hata oluştu: {e}")