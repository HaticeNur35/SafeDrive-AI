# SafeDrive-AI

> Yapay zekâ destekli motor koruma ve termal kontrol sistemi

SafeDrive-AI, bir motorun sıcaklık, titreşim, akım, voltaj ve hız verilerini izleyerek
gaz kesme kararı üreten ve motoru kademeli olarak koruyan bir sistemdir. Sert bir acil durdurma
yerine sensör verisine göre sürekli ve yumuşak bir güç ayarı yapar. Kritik durumlarda ise
sert sınırlar (fail-safe) devreye girer ve isteğe bağlı olarak Telegram üzerinden uyarı gönderir.

## Temel Özellikler

- **Kademeli güç kesme:** Sabit eşikler yerine sinir ağının ürettiği sürekli bir gaz kesme kararı.
- **Mekanik arıza riski tahmini:** Çok başlı (multi-head) mimaride ikinci bir çıktı.
- **PID ile yumuşatma:** Ani güç değişimlerini azaltan, integral sınırlı (anti-windup) PID denetleyici.
- **Fail-safe sınırları:** Aşırı sıcaklık, aşırı titreşim, düşük voltaj, arıza riski ve sensör hatası için sert durdurma.
- **Telegram uyarıları (isteğe bağlı):** Acil durdurma, mekanik arıza ve yüksek sıcaklık durumlarında mesaj gönderir.
- **CAN Bus simülasyonu:** Sensör verileri OBD-II / SAE J1939 benzeri 8 baytlık çerçevelerle kodlanıp çözülür.
- **Stres testleri:** Normal sürüş, ani yük, titreşim artışı ve voltaj düşüşü senaryoları.
- **Karşılaştırma:** SafeDrive-AI, eşik tabanlı kontrol, yalnızca PID ve yalnızca sinir ağı yöntemleriyle karşılaştırılır.
- **Canlı dashboard:** Matplotlib `FuncAnimation` ile gerçek zamanlı sıcaklık, motor gücü ve arıza riski grafikleri.

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

- 3.000 örnek üretilir; her örneğe sensör gürültüsü eklenir.
- Gaz kesme etiketi sıcaklığa göre kademelidir. Yüksek titreşim ve yüksek akım ek kesme uygular.
- Arıza etiketi, yüksek titreşim ve yüksek akım birlikte olduğunda 1 olur.
- Kayıp fonksiyonu: gaz kesme için SmoothL1, arıza için BCE. Toplam kayıp `loss_gaz + 0.6 * loss_ariza`.
- Optimizasyon: AdamW, CosineAnnealingLR, gradyan kırpma (`max_norm=1.0`), 800 epoch.

## Fail-Safe Eşikleri

| Koşul | Eşik |
|---|---|
| Aşırı sıcaklık | T ≥ 98 °C |
| Aşırı titreşim | titreşim > 0.90 |
| Düşük voltaj | voltaj < 0.35 |
| Mekanik arıza | arıza riski > 0.92 |
| Sensör hatası | NaN veya sonsuz değer |

## Kurulum

Gerekli kütüphaneler:

```bash
pip install torch numpy matplotlib requests
```

Çalıştırma:

```bash
python safedrive_ai.py
```

İlk çalıştırmada model eğitilip `motor_koruyucu.pth` olarak kaydedilir. Sonraki çalıştırmalarda
kayıtlı model yüklenir ve eğitim atlanır. Sıfırdan eğitmek için `safedrive_ai.py` içinde
`ZORLA_EGIT = True` yapın.

## Telegram Bildirimleri (İsteğe Bağlı)

Telegram uyarıları için bir bot token'ı ve sohbet kimliği gerekir. Bu bilgiler koda yazılmaz,
ortam değişkeni olarak tanımlanır:

Windows (PowerShell):
```powershell
$env:SAFEDRIVE_BOT_TOKEN="BOT_TOKEN_BURAYA"
$env:SAFEDRIVE_CHAT_ID="CHAT_ID_BURAYA"
```

Linux / macOS:
```bash
export SAFEDRIVE_BOT_TOKEN="BOT_TOKEN_BURAYA"
export SAFEDRIVE_CHAT_ID="CHAT_ID_BURAYA"
```

Bu değişkenler tanımlanmazsa program çalışmaya devam eder, yalnızca bildirimler atlanır.
Aynı tür uyarı en fazla 30 saniyede bir gönderilir.

## Çıktılar

Program çalıştığında betiğin bulunduğu klasöre şu dosyaları kaydeder:

- `stres_testi.png`: Dört senaryonun sıcaklık, güç ve arıza riski karşılaştırması.
- `baseline_karsilastirma.png`: SafeDrive-AI ile üç temel yöntemin ani yük senaryosundaki karşılaştırması.
- `motor_koruyucu.pth`: Eğitilmiş model ağırlıkları.

Canlı dashboard, `python safedrive_ai.py` çalıştığında ayrı bir pencerede açılır.

## Sınırlılıklar

- Eğitim verisi kurallarla sentetik olarak üretilmiştir. Model gerçek motor sensör verisiyle doğrulanmamıştır.
- Fiziksel model (ısınma ve soğuma katsayıları) basitleştirilmiş bir termal modeldir.
- CAN Bus katmanı bir simülasyondur; gerçek bir araç ağına bağlanmamıştır.
- Sonraki adım: gerçek sensör verisiyle test ve model doğrulaması.

## Sürüm Geçmişi

### v2.6 (güncel)
- Canlı dashboard, PyQtGraph yerine matplotlib `FuncAnimation` ile çalışır.
- Telegram uyarı sistemi eklendi. Token ve sohbet kimliği ortam değişkeninden okunur.
- Telegram bildirimleri için 30 saniyelik bekleme süresi (cooldown) eklendi.

### v2.5 (deneysel)
- PyQtGraph tabanlı dashboard denendi. Eksen ve ölçek sorunları nedeniyle kaldırıldı.

### v2.4
- Model mimarisi: 7→64→32→16, LayerNorm, ayrı kontrol ve arıza başlıkları.
- Eğitim verisi 54'ten 3.000 örneğe çıkarıldı.
- Eğitim döngüsüne kosinüs zamanlayıcı, gradyan kırpma ve model kaydetme eklendi.
- PID'ye anti-windup sınırı eklendi.
- Fail-safe eşikleri güncellendi, düşük voltaj kontrolü eklendi.
- CAN Bus simülasyonu ve stres testleri eklendi.

### v2.3
- Çok başlı mimari, PID + NN + fail-safe entegrasyonu.
- Gürültü altında kararlılık testi ve karar eğrisi görselleştirmesi.

### v1 (eski)
- Eşik tabanlı başlangıç sürümü. Dosya: `legacy/thermal_ai_control_v1_legacy.py`.

## Eski Sürümler

Eski dosyalar yalnızca karşılaştırma amacıyla `legacy/` klasöründe tutulmaktadır.
