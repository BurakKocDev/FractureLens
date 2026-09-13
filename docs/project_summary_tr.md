# FractureLens — Nihai Proje Özeti

## Projenin amacı

FractureLens, kas-iskelet röntgenlerinde kırık varlığını görüntü düzeyinde
sınıflandıran, şüpheli bölgeleri kutularla lokalize eden ve iki model arasındaki
uyuşmazlıkları görünür kılan uçtan uca bir araştırma prototipidir. Proje klinik
tanı sistemi değil; veri bilimi, bilgisayarlı görü, model güvenilirliği ve ürün
geliştirme adımlarını birlikte gösteren bir portföy çalışmasıdır.

## Veri ve deney tasarımı

- Veri kaynağı: CC BY 4.0 lisanslı FracAtlas v7
- Denetlenen kaynak kayıt: 4.083
- Temizlenmiş ve modele alınan görüntü: 3.978
- Eğitim: 2.784 görüntü, 485 pozitif
- Validasyon: 395 görüntü, 68 pozitif
- Test: 799 görüntü, 141 pozitif
- Exact-pixel ve yakın kopyalar grup bazında aynı bölmede tutuldu.
- Etiket çelişkisi içeren kopya bileşenleri model geliştirmeden çıkarıldı.
- Hasta kimlikleri bulunmadığından hasta bazlı ayrım garanti edilemedi.

Model seçimi, karar eşikleri, sıcaklık kalibrasyonu ve NMS ayarı yalnızca
validasyon verisiyle yapıldı. Test bölmesi nihai ölçüm için sabit tutuldu.

## Geliştirilen bileşenler

1. **Veri denetimi:** arşiv bütünlüğü, EXIF yönü, bozuk/truncated JPEG,
   anotasyon biçimleri ve kopya görüntü analizi.
2. **Track A:** pozitif görüntülerde YOLOv8s detection ve YOLOv8s-seg
   segmentasyon yeniden üretimi.
3. **Track B sınıflandırma:** MobileNetV3-Small başlangıç modeli ve son
   DenseNet121 karşılaştırıcısı.
4. **Track B lokalizasyon:** negatif görüntüleri de içeren YOLOv8s detector.
5. **Güvenilirlik:** sıcaklık ölçekleme, ECE/Brier, sabit validasyon eşikleri,
   alt grup ve hata analizi.
6. **Füzyon:** OR/AND kararları, global-lokal uyum ve açık uyuşmazlık durumu.
7. **Uygulama:** FastAPI, dosya yükleme, tarayıcı arayüzü, işaretli görüntü ve
   indirilebilir PNG raporu.

## Temel sonuçlar

### DenseNet121 sınıflandırıcı

| Metrik | Test sonucu |
| --- | ---: |
| AUROC | 0,912 |
| AUPRC | 0,781 |
| Duyarlılık | 0,773 |
| Özgüllük | 0,881 |
| Dengeli doğruluk | 0,827 |
| ECE | 0,057 |
| Brier | 0,084 |

### YOLOv8s lokalizasyon

| Metrik | Test sonucu |
| --- | ---: |
| mAP50 | 0,479 |
| mAP50-95 | 0,195 |
| Sabit eşikte lezyon precision | 0,590 |
| Sabit eşikte lezyon duyarlılığı | 0,449 |
| Negatif görüntü yanlış alarm oranı | %0,76 |

Validasyonda seçilen NMS IoU 0,50, yerel inference çıktısında örtüşen kutuları
azalttı. Testte precision 0,738 ve F1 0,545 olurken lezyon duyarlılığı 0,432'ye
geriledi. Kanonik mAP sonuçları bu sunum sonrası işleminden ayrı tutuldu.

### Füzyon

| Yaklaşım | Duyarlılık | Özgüllük | Dengeli doğruluk |
| --- | ---: | ---: | ---: |
| DenseNet121 | 0,773 | 0,881 | 0,827 |
| Detector görüntü sinyali | 0,532 | 0,992 | 0,762 |
| OR füzyon | **0,865** | 0,880 | **0,873** |
| AND füzyon | 0,440 | **0,994** | 0,717 |

Testte iki model 664/799 görüntüde aynı yönde karar verdi. Sınıflandırıcının
kaçırdığı 13 pozitif görüntü detector tarafından, detector'ın lokalize
edemediği 47 pozitif görüntü ise sınıflandırıcı tarafından yakalandı. Her iki
model 19 pozitif görüntüyü birlikte kaçırdı.

## Projenin güçlü tarafları

- Rastgele yeniden bölme yerine SHA-256 ile sabitlenmiş duplicate-aware manifest
- Pozitiflerle sınırlı detector yerine negatif görüntüleri içeren gerçekçi deney
- Teste bakmadan validasyonda seçilen eşik ve model kararları
- Yalnız AUROC değil AUPRC, kalibrasyon, alt grup ve hata ölçümleri
- Model uyuşmazlığını saklamayan açıklanabilir füzyon tasarımı
- Araştırma kodundan çalışan API ve kullanıcı arayüzüne kadar uçtan uca sistem
- Model ağırlığı bütünlük kontrolü ve yeniden üretilebilir deney kayıtları

## Sınırlılıklar

- Tek veri setinde retrospektif değerlendirme yapıldı; dış doğrulama yok.
- Hasta kimliği ve demografik bilgi bulunmuyor.
- JPG görüntüler DICOM piksel derinliği ve çekim bilgisini taşımıyor.
- Detector sabit eşikte anotasyonlu lezyonların yarısından fazlasını kaçırıyor.
- Donanım, kenarlık, yazı ve çoklu çekim panelleri shortcut riski oluşturabilir.
- Model skorları klinik olasılık veya hastalık şiddeti değildir.

## Portföy anlatımı

Kısa anlatım:

> FracAtlas üzerinde kopya farkındalıklı veri ayrımı kurdum; kalibre edilmiş
> DenseNet121 sınıflandırıcı ile negatif görüntüleri de gören YOLOv8s detector'ı
> eğitip test ettim. İki modelin tamamlayıcı hatalarını OR/AND füzyon ve açık
> uyuşmazlık durumlarıyla analiz ettim. Sistemi FastAPI ve görsel rapor üreten
> yerel web arayüzüyle uçtan uca çalışır hale getirdim.

Bu çalışma araştırma/portföy kapsamı bakımından tamamlanmıştır. Klinik kullanım
için dış veri doğrulaması, hasta bazlı ayrım, görüntü kalite kontrolü,
prospektif çalışma ve mevzuat süreçleri gerekir; bunlar mevcut projenin kapsamı
dışındadır.
