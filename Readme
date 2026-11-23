# SafeDrive-AI

> Yapay zekâ destekli motor koruma ve termal kontrol sistemi

SafeDrive-AI, bir motorun sıcaklık, titreşim, akım, voltaj ve hız verilerini izleyerek
gaz kesme kararı üreten ve motoru kademeli olarak koruyan bir sistemdir. Sert bir acil durdurma
yerine, sensör verisine göre sürekli ve yumuşak bir güç ayarı yapar.

## Temel Özellikler

- **Kademeli güç kesme:** Sabit eşikler yerine sinir ağının ürettiği sürekli bir karar.
- **Mekanik arıza riski tahmini:** Çok başlı (multi-head) mimaride ikinci bir çıktı.
- **PID ile yumuşatma:** Ani güç değişimlerini azaltan anti-windup korumalı PID denetleyici.
- **Fail-safe sınırları:** Aşırı sıcaklık, aşırı titreşim, düşük voltaj ve sensör hatası için sert durdurma.
- **CAN Bus simülasyonu:** Sensör verileri OBD-II / SAE J1939 benzeri 8 baytlık çerçevelerle kodlanıp çözülür.
- **Stres testleri:** Normal sürüş, ani yük, titreşim artışı ve voltaj düşüşü senaryoları.
- **Karşılaştırma:** SafeDrive-AI, eşik tabanlı kontrol, yalnızca PID ve yalnızca sinir ağı yöntemleriyle karşılaştırılır.
- **Canlı dashboard:** `FuncAnimation` ile 300 ms aralıklı gerçek zamanlı grafik.

## Mimari

```
Sensör girişi (7 özellik)
        │
        ├─ state_net (7→64→32→16, LayerNorm + Tanh)
        │         │
        │   skip_proj (7→16)
        │         │
        └──── state (16)
                  │
        ┌─────────┴──────────┐
        │                    │
  control_head           fault_head
 (16→8→1, Sigmoid)    (16→8→1, Sigmoid)
        │                    │
  Gaz kesme [0-1]      Arıza riski [0-1]
        │
   PID (anti-windup)
        │
    Fail-safe
        │
  Motor gücü [0-1]
```

## Eğitim

- 3.000 örnek, sensör gürültüsüyle birlikte üretilmiştir.
- Gaz kesme etiketi sıcaklığa göre kademeli, titreşim ve akıma göre ek kesme içerir.
- Arıza etiketi, yüksek titreşim ve yüksek akım birlikte olduğunda 1 olur.
- Kayıp fonksiyonu: gaz kesme için SmoothL1, arıza için BCE; toplam kayıp `loss_gaz + 0.6 * loss_ariza`.
- Optimizasyon: AdamW, CosineAnnealingLR, gradyan kırpma (`max_norm=1.0`).

## Fail-Safe Eşikleri

| Koşul | Eşik |
|---|---|
| Aşırı sıcaklık | T ≥ 98 °C |
| Aşırı titreşim | titreşim > 0.90 |
| Düşük voltaj | voltaj < 0.35 |
| Mekanik arıza | arıza riski > 0.92 |
| Sensör hatası | NaN veya sonsuz değer |

## Kurulum

```bash
pip install torch numpy matplotlib
python safedrive_ai.py
```

İlk çalıştırmada model eğitilip `motor_koruyucu.pth` olarak kaydedilir. Sonraki çalıştırmalarda
kayıtlı model yüklenir. Sıfırdan eğitmek için `safedrive_ai.py` içinde `ZORLA_EGIT = True` yapın.

## Sınırlılıklar

- Eğitim verisi kurallarla sentetik olarak üretilmiştir; model gerçek motor sensör verisiyle
  doğrulanmamıştır.
- Fiziksel model (ısınma ve soğuma katsayıları) basitleştirilmiş bir termal modeldir.
- CAN Bus katmanı bir simülasyondur; gerçek bir araç ağına bağlanmamıştır.
- Sonraki adım: gerçek sensör verisiyle test ve model doğrulaması.

## Sürüm Geçmişi

### v2.4 (güncel)
- Model mimarisi: 7→64→32→16, LayerNorm, ayrı kontrol ve arıza başlıkları.
- Eğitim verisi 54'ten 3.000 örneğe çıkarıldı.
- Eğitim döngüsüne kosinüs zamanlayıcı, gradyan kırpma ve model kaydetme eklendi.
- PID'ye anti-windup sınırı eklendi.
- Fail-safe eşikleri güncellendi, düşük voltaj kontrolü eklendi.
- CAN Bus simülasyonu ve stres testleri eklendi.
- Canlı dashboard ile önceki sürümdeki animasyon hataları giderildi.

### v2.3
- Çok başlı mimari, PID + NN + fail-safe entegrasyonu.
- Gürültü altında kararlılık testi ve karar eğrisi görselleştirmesi.

### v1 (eski)
- Eşik tabanlı başlangıç sürümü.
