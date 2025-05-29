

### 📄 `README.md`

```markdown
# 🧠 Ürün Adı Benzerlik Analizi

Bu proje, büyük ölçekli ürün adı verisi üzerinde **doğal dil işleme (NLP)** tekniklerini kullanarak benzerlik analizi yapmayı amaçlamaktadır. Projede, ürün adları arasında **TF-IDF** ve **Word2Vec** gibi vektörleştirme yöntemleriyle benzerlik ölçümleri yapılmış ve farklı model konfigürasyonlarının başarıları karşılaştırılmıştır.

---

## 📌 Proje Özellikleri

- TF-IDF tabanlı benzerlik hesaplama
- Word2Vec (CBOW & Skip-gram) modelleriyle semantik karşılaştırma
- Lemmatization ve Stemming ile ön işleme karşılaştırması
- Cosine benzerliği ile puanlama
- Jaccard benzerlik matrisi ile model sıralama tutarlılığı
- Görsel analizler (matplotlib & seaborn)

---

## 📁 Proje Yapısı

```

text\_similarity\_project/
│
├── data/                            # Veri dosyaları
│   ├── urun\_adlari\_large\_dataset.csv
│   ├── lemmatized.csv
│   └── stemmed.csv
│
├── evaluation/                      # Değerlendirme modülleri
│   ├── jaccard\_eval.py
│   └── subjective\_eval.py
│
├── models/                          # Eğitilmiş Word2Vec modelleri (.model)
│
├── main.py                          # Ana çalışma dosyası
├── run\_preprocessing.py             # Veri ön işleme scripti
├── README.md                        # Proje açıklaması
├── requirements.txt                 # Bağımlılıklar
└── .gitignore                       # Git izleme dışı dosyalar

````

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
````

### 2. Ön işleme işlemlerini çalıştırın:

```bash
python run_preprocessing.py
```

### 3. Benzerlik analizini başlatın:

```bash
python main.py
```

---

## 🧪 Kullanılan Yöntemler

### ✅ TF-IDF

* Tokenization, Lemmatization ve Stemming ile ön işleme
* TF-IDF vektör matrisi oluşturma
* Cosine similarity ile puanlama

### ✅ Word2Vec

* CBOW ve Skip-gram mimarileri
* Farklı pencere boyutları (`window=2`, `window=4`)
* Vektör boyutları (`dim=100`, `dim=300`)
* Ortalama vektör (mean pooling)
* Cosine benzerliği

---

## 📊 Örnek Sonuçlar

| Model                          | Ortalama Skor |
| ------------------------------ | ------------- |
| TF-IDF Lemmatized              | 0.65          |
| TF-IDF Stemmed                 | 0.72          |
| Word2Vec Skip-gram (stem, w=2) | **0.90**      |
| Word2Vec CBOW (lemma, w=2)     | 0.78          |

📈 `grafik1_benzerlik_skorlari.png` ve
📈 `jaccard_benzerlik_matrisi_urun_adlari.png` gibi görsellerle detaylı analiz sağlanmıştır.

---

## 📚 Bağımlılıklar

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
gensim
```

---

## 🔬 Hedef Uygulamalar

* E-ticaret ürün adı eşleştirme
* Ürün katalog standardizasyonu
* Otomatik etiketleme ve öneri sistemleri
* Marka adları varyant tespiti

---

## 🧑‍💻 Geliştirici

**Melih Ulusoy**
[GitHub](https://github.com/MelihUlusoy)

---

## ⚖️ Lisans

Bu proje MIT lisansı ile lisanslanmıştır.

```

---

İstersen bu dosyayı doğrudan projenin içine `.md` formatında kaydedebilirim. Ayrıca `.gitignore` ve `requirements.txt` dosyalarını da hazır hale getirmemi ister misin?
```
