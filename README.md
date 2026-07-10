# Kredi-Temerr-t-Tahmini-ve-Analizi
Projemin amacı, bir bankanın kredi başvurularını kullanarak hem iş zekası (BI) düzeyinde görselleştirme hem de makine öğrenmesi ile temerrüt (ödememe) riskini tahmin eden bir model geliştirmektir.
📊 Proje İki Bölümden Oluşuyor

<img width="1287" height="717" alt="PowerBıAnaliz" src="https://github.com/user-attachments/assets/9c3074ae-9554-4d90-9309-4df597df061b" />


1) Power BI Dashboard

149K kredi başvurusunu analiz eden interaktif bir dashboard hazırlandı:
- Toplam başvuru, ödeme/ödememe oranı gibi temel KPI'lar
- Kredi türü ve kredi sebeplerine göre dağılım
- Yaş gruplarına göre ödeme durumu karşılaştırması
- Gelir seviyesine göre ödememe oranı trendi (risk segmentasyonu)
- Kredi tutarı aralıklarına göre yoğunluk analizi

<img width="1000" height="500" alt="f1score" src="https://github.com/user-attachments/assets/3509c3cd-f87b-4e12-bf3c-3b90b4ba7138" />


2) Makine Öğrenmesi Modeli 


clean_data.csv üzerinde çalışan bir sınıflandırma pipeline'ı kuruldu:
Veri Hazırlığı
- Eksik değerler kategorik sütunlarda mod, sayısal sütunlarda medyan ile dolduruldu
- Data leakage önlendi: rate_of_interest, Upfront_charges, Interest_rate_spread gibi
  kredi onaylandıktan SONRA belirlenen sütunlar modelden çıkarıldı
- Sınıf dengesizliği SMOTE (Synthetic Minority Oversampling) ile giderildi
Modeller ve Karşılaştırma
Logistic Regression, Decision Tree ve Random Forest modelleri eğitildi ve
Accuracy, Precision, Recall, F1 Score, ROC-AUC metrikleriyle karşılaştırıldı.

<img width="1000" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/1762374d-3a30-451b-b762-0a6e7dc1d4b0" />

En iyi performansı gösteren Random Forest modeli seçildi.
Karışıklık Matrisi (Random Forest)
- Doğru tahmin edilen "ödemeyen" (temerrüt): 4.991
- Doğru tahmin edilen "ödeyen": 20.104
- Model, temerrüde düşecek müşterilerin büyük kısmını başarıyla yakalıyor
Özellik Önem Sıralaması
Modele göre temerrüdü en çok etkileyen faktörler: credit_type, LTV (kredi/değer oranı),
dtir1 (borç/gelir oranı), property_value. Bu sonuçlar, Power BI dashboard'daki
gelir-risk ilişkisi bulgusuyla da tutarlı.

🛠️ Kullanılan Teknolojiler

Python (pandas, scikit-learn, imbalanced-learn, matplotlib), Power BI

<img width="640" height="480" alt="karışıklık matrisi" src="https://github.com/user-attachments/assets/7048876f-489f-4b12-824d-0d08c03edcf3" />


