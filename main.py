import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from src.utils import get_stratified_folds, evaluate_predictions, SEED
from src.features import engineer_all_features 

DATA_PATH = "data/raw/YARISMA_TRAIN_MASTER.csv"
TARGET_COL = "Label"

def main():
    df = pd.read_csv(DATA_PATH)

    print("🚀 GeneSight Pipeline Başlatılıyor...")
    print("Dataset shape:", df.shape) # Beklenen: (2931, 353)

    if TARGET_COL in df.columns:
        print("\nLabel distribution:")
        print(df[TARGET_COL].value_counts()) # Beklenen: 2149 P / 782 B
        
        # --- ÖZELLİK MÜHENDİSLİĞİ KATMANI -
        # df = engineer_all_features(df)
        
        # --- VERİ HAZIRLIĞI ---
        # Modele girdi olmayacak ve ID sızıntısı yaratacak Variant_ID ile hedef değişkeni ayırıyoruz[cite: 1]
        features = [col for col in df.columns if col not in ["Variant_ID", TARGET_COL, "CAT_6"]] # CAT_6 %98 boş olduğu için drop[cite: 1]
        
        X = df[features].copy()
        y = df[TARGET_COL].copy()

        # Kategorik sütunları (CAT_ ve AA_) XGBoost'un tanıması için category tipine alıyoruz
        cat_cols = [col for col in X.columns if col.startswith('CAT_') or col.startswith('AA_')]
        for col in cat_cols:
            X[col] = X[col].astype('category')

        # --- CROSS-VALIDATION DÖNGÜSÜ ---
        folds = get_stratified_folds(df, target_col=TARGET_COL, n_splits=5)
        fold_f1_scores = []

        print(f"\nTraining started with 5-Fold Stratified CV (Seed: {SEED})...")

        for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # Sınıf dengesizliğini yönetmek için ağırlık hesabı[cite: 1]
            pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            # Raporumuzda bahsettiğimiz Monotonik Kısıtları tanımlıyoruz[cite: 3]
            # (Özellik mühendisliği sütunları eklendiğinde aktif edilecek yön haritası)
            monotone_constraints = {}
            if 'grantham_score' in X_train.columns:
                monotone_constraints['grantham_score'] = 1 # Yapısal hasar risk artırır (+)[cite: 3]
                monotone_constraints['blosum62_score'] = -1 # Evrimsel benzerlik risk düşürür (-)[cite: 3]

            # Model Tanımı
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=pos_weight,
                enable_categorical=True, # CAT_ ve AA_ native categorical desteği[cite: 1]
                random_state=SEED,
                eval_metric='logloss',
                # monotone_constraints=monotone_constraints # Yarın aktif edeceğiz
            )

            # Modeli Eğit (XGBoost eksik verileri yerel olarak yönetir, imputation gerekmez)
            model.fit(X_train, y_train)

            # Olasılık Tahmini ve Dinamik Karar Eşiği Uygulaması[cite: 1, 3]
            # Kanser paneli için 0.36 bulmuştuk, Master için baseline aşamasında 0.50 ile başlayabiliriz[cite: 1, 3]
            proba_preds = model.predict_proba(X_valid)[:, 1]
            
            # Şimdilik standart 0.5 eşiği, optimizasyon aşamasında sweep edeceğiz[cite: 1, 3]
            threshold = 0.5 
            binary_preds = (proba_preds >= threshold).astype(int)

            # Fold Skorunu Hesapla
            fold_f1 = evaluate_predictions(y_valid, binary_preds)
            fold_f1_scores.append(fold_f1)
            print(f"  -> Fold {fold_id} F1 Score: {fold_f1:.4f}")

        print("\n🏁 PIPELINE BASELINE SONUCU:")
        print(f"  • 5-Fold Ortalama F1 Skoru: {np.mean(fold_f1_scores):.4f}")

    else:
        print(f"Target column '{TARGET_COL}' not found.")


if __name__ == "__main__":
    main()
