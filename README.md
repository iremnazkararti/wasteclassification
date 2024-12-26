# wasteclassification
### Plastik Atık Sınıflandırma ve Karbon Ayak İzi Tahmini

Bu proje, plastik atıkları sınıflandırmak ve karbon ayak izi tahmini yapmak için geliştirilmiştir. **MobileNet tabanlı derin öğrenme modeli** ile birlikte **Random Forest, Decision Tree ve SVM** gibi klasik makine öğrenimi modelleri kullanılmıştır.

---

### Özellikler
- Plastik türlerini sınıflandırma (**Plastik, Metal, Kağıt, Karton, Cam, Çöp**).
- Eğitim ve test için veri artırma ve ön işleme.
- MobileNet'ten özellik çıkarımı ve klasik makine öğrenimi modelleri ile sınıflandırma.
- Tahmin edilen tür için karbon ayak izi hesaplama.

---

### Karbon Ayak İzi Tahmini
Her plastik türü için karbon ayak izi değerleri:
| **Tür**       | **Karbon Ayak İzi (kg CO2e)** |
|---------------|-------------------------------|
| Plastik       | 6.0                           |
| Metal         | 5.0                           |
| Kağıt         | 4.0                           |
| Karton        | 3.0                           |
| Cam           | 2.0                           |
| Çöp           | 1.0                           |

---

### Kullanım
1. **Veri Seti**: `train` ve `test` dizinlerinde organize edilmiş plastik görüntüleri.
2. **Model Eğitimi**: Kod, hem derin öğrenme hem de klasik modeller için eğitim sağlar.
3. **Tahmin**: Test görüntüleri üzerinde sınıflandırma ve karbon ayak izi tahmini yapılabilir.

---

### İletişim
Sorularınız için bir issue açabilir veya bana e-posta gönderebilirsiniz:  
**iremnazkararti@gmail.com**
