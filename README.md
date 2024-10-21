# COVID-19 Görüntü Sınıflandırma Uygulaması

Bu uygulama, COVID-19 tespiti için görüntü sınıflandırma yapan bir makine öğrenimi modelidir. Kullanıcıların yüklediği görüntüler üzerinde farklı sınıflandırma modelleri kullanarak tahminler yapmaktadır. Destek Vektör Makineleri (SVM), Karar Ağaçları, Lojistik Regresyon ve Random Forest gibi modeller arasından seçim yapma imkanı sunar.

## Özellikler

- **Birden fazla model desteği**: Random Forest, Lojistik Regresyon, SVM ve Karar Ağaçları modelleri ile tahmin yapma.
- **Görüntü yükleme**: Kullanıcıların kendi COVID-19 görüntülerini yükleyerek tahmin yapabilmeleri.
- **Model performansı gösterimi**: Her modelin geçmiş performansı ve doğruluk oranları grafikte gösterilmektedir.

## Kullanım

1. Uygulamayı çalıştırmak için [Streamlit](https://streamlit.io/) kütüphanesini kurun.
2. Model dosyalarını (`rf.pkl`, `lr.pkl`, `svm.pkl`, `dt.pkl`) uygulama ile aynı dizine yerleştirin.
3. Uygulamayı çalıştırmak için terminalde aşağıdaki komutu kullanın:

   ```bash
   streamlit run app.py
##NOTLAR
- Uygulamayı test etmek için covid.png ve noncovid.png adında iki adet örnek görüntü mevcuttur. Bu görüntüleri yükleyerek modeli test edebilirsiniz.
