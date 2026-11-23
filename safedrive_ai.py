# ============================================================
#   SafeDrive-AI  —  Son Sürüm (Tüm Hatalar Düzeltildi)
#   Yazar  : SafeDrive-AI Projesi
#   Versiyon: 2.4
# ============================================================

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# Tekrarlanabilirlik için seed sabitle
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Grafik kayıt dizini: betiğin yanına kaydeder (Windows uyumlu)
try:
    CIKTI_DIZIN = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CIKTI_DIZIN = os.getcwd()
os.makedirs(CIKTI_DIZIN, exist_ok=True)
print(f"Grafikler şuraya kaydedilecek: {CIKTI_DIZIN}")

# ============================================================
# 1.  MODEL — DÜZELTILMIŞ MULTİ-HEAD MİMARİ
# ============================================================
#   Eski hatalar:
#     • state + x[:,1:2]  → boyut uyumsuzluğu, fiziksel anlamsızlık
#     • Çok küçük ağ (7→32→16) → öğrenme kapasitesi yetersiz
#     • Tek lineer head → karar sınırı çok kaba
# ============================================================
class MotorKoruyucuAI(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Ortak Gövde (State Encoder) ----
        self.state_net = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        # ---- Skip Connection Projeksiyonu (7 → 16) ----
        # ESKİ KOD: state + x[:,1:2]  → 16 boyuta 1 boyut ekleniyordu (YANLIŞ)
        # YENİ KOD: tam boyut eşleştirme ile anlamlı artık bağlantı
        self.skip_proj = nn.Linear(7, 16, bias=False)

        # ---- Head 1: Gaz Kesme (Sürekli 0–1) ----
        self.control_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        # ---- Head 2: Mekanik Arıza Riski (0–1) ----
        self.fault_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        state = self.state_net(x) + self.skip_proj(x)   # artık boyutlar eşit
        gaz_kesme  = torch.sigmoid(self.control_head(state))
        ariza_risk = torch.sigmoid(self.fault_head(state))
        return gaz_kesme, ariza_risk


# ============================================================
# 2.  EĞİTİM VERİSİ — ZENGİN & GERÇEKÇİ
# ============================================================
#   Eski hatalar:
#     • Sadece 54 örnek → aşırı öğrenme (overfitting) kaçınılmaz
#     • Titreşim ve akım sabit → model bu özellikleri öğrenemedi
#     • Gürültü yok → gerçek sensör verisiyle uyumsuzluk
# ============================================================
def veri_uret(n_ornek: int = 3000):
    X, Y_gaz, Y_ariza = [], [], []

    for _ in range(n_ornek):
        t        = random.uniform(42.0, 100.0)
        T0       = t / 100.0
        T1       = max(0.0, (t - random.uniform(0.5, 2.5)) / 100.0)
        T2       = max(0.0, (t - random.uniform(1.5, 5.0)) / 100.0)
        hiz      = random.uniform(0.2, 1.0)
        akim     = random.uniform(0.15, 0.95)
        titresim = random.uniform(0.05, 0.95)
        voltaj   = random.uniform(0.55, 1.05)

        # ---- Gaz kesme etiketi (sıcaklık + titreşim etkili) ----
        if   t < 68:   kesme = 0.0
        elif t < 75:   kesme = (t - 68) / 70.0
        elif t < 85:   kesme = 0.10 + (t - 75) * 0.028
        elif t < 95:   kesme = 0.38 + (t - 85) * 0.062
        else:          kesme = 1.0

        # Yüksek titreşim → ek kesme
        if titresim > 0.60:
            kesme = min(1.0, kesme + (titresim - 0.60) * 0.45)

        # Yüksek akım → hafif kesme
        if akim > 0.75:
            kesme = min(1.0, kesme + (akim - 0.75) * 0.20)

        # ---- Arıza etiketi (titreşim & akım birlikte yüksekse) ----
        ariza = 1.0 if (titresim > 0.62 and akim > 0.62) else 0.0

        # ---- Sensor gürültüsü ----
        def n(std=0.008): return random.gauss(0, std)

        X.append([
            hiz      + n(0.015),
            T0       + n(0.006),
            akim     + n(0.012),
            titresim + n(0.012),
            voltaj   + n(0.008),
            T1       + n(0.006),
            T2       + n(0.006),
        ])
        Y_gaz.append([kesme])
        Y_ariza.append([ariza])

    return (
        torch.tensor(X,       dtype=torch.float32),
        torch.tensor(Y_gaz,   dtype=torch.float32),
        torch.tensor(Y_ariza, dtype=torch.float32),
    )


# ============================================================
# 3.  EĞİTİM DÖNGÜSÜ + MODEL KAYDET / YÜKLE
# ============================================================
#   İlk çalıştırma : eğitir → motor_koruyucu.pth kaydeder
#   Sonraki çalıştırmalar: .pth varsa yükler, eğitimi atlar (~30s kazanç)
#   Zorla yeniden eğitmek için: MODEL_DOSYASI'nı sil veya ZORLA_EGIT=True yap
# ============================================================
MODEL_DOSYASI = os.path.join(CIKTI_DIZIN, "motor_koruyucu.pth")
ZORLA_EGIT    = False   # True yapınca kayıtlı model olsa bile yeniden eğitir

print("=" * 55)
print("  SafeDrive-AI  —  Model Yönetimi")
print("=" * 55)

model = MotorKoruyucuAI()

if os.path.exists(MODEL_DOSYASI) and not ZORLA_EGIT:
    # ---- Model Yükleme ----
    model.load_state_dict(torch.load(MODEL_DOSYASI, weights_only=True))
    model.eval()
    print(f"  ✓ Kayıtlı model yüklendi  →  {MODEL_DOSYASI}")
    print("    (Yeniden eğitmek için ZORLA_EGIT = True yapın)\n")

else:
    # ---- Model Eğitimi ----
    if ZORLA_EGIT:
        print("  ⚡ ZORLA_EGIT=True — model sıfırdan eğitiliyor...")
    else:
        print("  ℹ Kayıtlı model bulunamadı — ilk eğitim başlıyor...")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=800, eta_min=1e-5
    )
    loss_fn_gaz   = nn.SmoothL1Loss()
    loss_fn_ariza = nn.BCELoss()

    X_train, Y_gaz_train, Y_ariza_train = veri_uret(3000)

    EPOCH = 800
    for epoch in range(1, EPOCH + 1):
        model.train()
        optimizer.zero_grad()

        gaz_pred, ariza_pred = model(X_train)

        loss_gaz   = loss_fn_gaz(gaz_pred, Y_gaz_train)
        loss_ariza = loss_fn_ariza(ariza_pred, Y_ariza_train)
        loss_total = loss_gaz + 0.6 * loss_ariza

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 200 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:>4} | GazLoss: {loss_gaz:.4f} "
                  f"| ArizaLoss: {loss_ariza:.4f} | LR: {lr:.6f}")

    # ---- Kaydet ----
    torch.save(model.state_dict(), MODEL_DOSYASI)
    print(f"\n  ✓ Model kaydedildi  →  {MODEL_DOSYASI}")
    print("  Eğitim tamamlandı.\n")


# ============================================================
# 4.  MODEL TESTİ — 3 KRİTİK SENARYO
# ============================================================
print("-" * 55)
print("  Model Kararları — Senaryo Testi")
print("-" * 55)

def model_test(ad, veri_liste):
    model.eval()
    inp = torch.tensor([veri_liste], dtype=torch.float32)
    with torch.no_grad():
        gaz, ariza = model(inp)
    g, a = gaz.item(), ariza.item()
    durum = ("⚠ MEKANİK ARIZA"  if a > 0.7  else
             "↓ KADEMELİ SOĞUTMA" if g > 0.15 else
             "✓ SİSTEM STABİL")
    print(f"  {ad:<28} | GazKesme: %{g*100:5.1f} | "
          f"Ariza: %{a*100:5.1f} | {durum}")

model_test("Normal Çalışma",
           [0.65, 0.72, 0.30, 0.20, 0.85, 0.71, 0.70])
model_test("Titreşim Var, Soğuk",
           [0.65, 0.62, 0.72, 0.75, 0.82, 0.61, 0.60])
model_test("Sıcak + Titreşimli",
           [0.65, 0.90, 0.72, 0.75, 0.78, 0.89, 0.88])
print()


# ============================================================
# 5.  PID DENETLEYİCİ — ANTİ-WİNDUP'LI
# ============================================================
#   Eski hata: integral sınırsız büyüyebiliyordu (windup riski)
# ============================================================
class PID:
    def __init__(self, kp=0.8, ki=0.1, kd=0.05, dt=1.0,
                 integral_limit=8.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_limit = integral_limit
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def step(self, setpoint, measurement):
        error         = setpoint - measurement
        self.integral = float(np.clip(
            self.integral + error * self.dt,
            -self.integral_limit, self.integral_limit
        ))
        derivative    = (error - self.prev_error) / self.dt
        self.prev_error = error
        return (self.kp * error +
                self.ki * self.integral +
                self.kd * derivative)


# ============================================================
# 6.  FAIL-SAFE — ÇOKLU SERT SINIR
# ============================================================
def fail_safe(sensor: torch.Tensor, ariza_risk: float):
    """
    Sensör değerlerine ve model çıktısına göre
    acil durdurma kararı verir.
    Döndürür: (durdur: bool, sebep: str)
    """
    if torch.isnan(sensor).any() or torch.isinf(sensor).any():
        return True, "SENSOR_HATASI"

    T0       = sensor[0, 1].item() * 100.0
    titresim = sensor[0, 3].item()
    voltaj   = sensor[0, 4].item()

    if T0 >= 98.0:
        return True, "ASIRI_SICAKLIK"
    if titresim > 0.90:
        return True, "ASIRI_TITRESIM"
    if voltaj < 0.35:
        return True, "DUSUK_VOLTAJ"
    if ariza_risk > 0.92:
        return True, "MEKANIK_ARIZA"
    return False, "OK"


# ============================================================
# 7.  ANA KONTROL SİSTEMİ (NN + PID + FAIL-SAFE)
# ============================================================
class MotorKontrolSistemi:
    def __init__(self, model: nn.Module):
        self.model = model
        self.pid   = PID(kp=0.8, ki=0.1, kd=0.05)
        self.guc   = 1.0          # anlık motor gücü [0-1]

    def reset(self):
        self.guc = 1.0
        self.pid.reset()

    def adim(self, sensor: torch.Tensor):
        """
        Bir zaman adımı çalıştır.
        Döndürür: (guc, gaz_kesme, ariza, durum_str)
        """
        self.model.eval()
        with torch.no_grad():
            gaz_kesme, ariza = self.model(sensor)
        gaz_val   = float(gaz_kesme)
        ariza_val = float(ariza)

        # Fail-safe kontrolü
        dur, sebep = fail_safe(sensor, ariza_val)
        if dur:
            self.guc = 0.0
            return 0.0, gaz_val, ariza_val, f"FAIL-SAFE:{sebep}"

        # NN → hedef güç
        hedef = 1.0 - gaz_val

        # PID → yumuşat
        pid_cikti = self.pid.step(setpoint=hedef,
                                  measurement=self.guc)
        self.guc = float(np.clip(self.guc + pid_cikti, 0.0, 1.0))

        # Uyarı eşiği
        T0 = sensor[0, 1].item() * 100.0
        if T0 >= 80:
            durum = "UYARI:SICAK"
        elif gaz_val > 0.15:
            durum = "SOGUTMA"
        else:
            durum = "NORMAL"

        return self.guc, gaz_val, ariza_val, durum


# ============================================================
# 8.  MOTOR FİZİĞİ — EULER YÖNTEMI
# ============================================================
#   dT/dt = α·P + γ·Vib - β·(T - T_ortam) + ekstra_yuk
#   Katsayılar gerçek bir taşıt motoruna yakın kalibre edildi.
#   Eski kod: alpha=0.6 (çok yüksekti; motor asla soğumuyordu)
# ============================================================
def motor_fizigi(T: float,
                 guc: float,
                 titresim: float  = 0.0,
                 ekstra_yuk: float = 0.0,
                 ortam_T: float   = 25.0,
                 dt: float        = 1.0) -> float:
    alpha  = 0.55   # ısınma katsayısı (artırıldı: motor gerçekçi ısınsın)
    beta   = 0.028  # soğuma katsayısı (azaltıldı: gerçek motor yavaş soğur)
    gamma  = 1.40   # titreşim ısı katkısı (artırıldı: titreşim ciddi ısı üretir)

    isinma = alpha * guc + gamma * max(0.0, titresim - 0.3)
    soguma = beta  * (T - ortam_T)
    return T + (isinma - soguma + ekstra_yuk) * dt


# ============================================================
# 9.  BASELINE KONTROL YÖNTEMLERİ (Karşılaştırma)
# ============================================================
def baseline_threshold(T_derece: float) -> float:
    """
    Baseline-1: Sabit eşik tabanlı kontrol.
    Gerçek dünya uygulamalarında hâlâ yaygın kullanılan
    basit kural tabanlı yöntem.
    """
    if   T_derece >= 95: return 0.00
    elif T_derece >= 80: return 0.55
    else:                return 1.00


def baseline_pid_only(pid_b: PID,
                      T_derece: float,
                      mevcut_guc: float) -> float:
    """
    Baseline-2: Saf PID kontrolü.
    Hedef sıcaklık: 72°C. Arıza bilgisi yok.
    """
    hata    = 72.0 - T_derece
    cikti   = pid_b.step(setpoint=0.0, measurement=-hata)
    return float(np.clip(mevcut_guc + cikti * 0.05, 0.0, 1.0))


def baseline_nn_only(model: nn.Module,
                     sensor: torch.Tensor) -> float:
    """
    Baseline-3: PID yumuşatması olmadan saf NN kararı.
    Ani gaz değişimlerine karşı savunmasızdır.
    """
    model.eval()
    with torch.no_grad():
        gaz_kesme, _ = model(sensor)
    return float(1.0 - gaz_kesme)


# ============================================================
# 10. STRES TEST SENARYOLARI
# ============================================================
def stres_senaryosu(t: int, senaryo: str) -> dict:
    """
    Her adımda dinamik sensör değerleri üretir.
    t_mod ile döngüsel çalışır → dashboard'da sonsuza kadar tekrarlar.
    """
    def n(std=0.02): return random.gauss(0, std)

    t_mod = t % 85   # 85 saniyelik döngü — dashboard için kritik

    if senaryo == "normal":
        return dict(hiz=0.65+n(0.015), akim=0.48+n(0.025),
                    titresim=0.28+n(0.018), voltaj=0.88+n(0.010),
                    ekstra_yuk=0.05)

    elif senaryo == "ani_yuk":
        ani  = 0.45 if 18 <= t_mod <= 32 else 0.0
        ani2 = 0.35 if 55 <= t_mod <= 68 else 0.0
        return dict(hiz=0.90+n(0.020), akim=0.78+ani+ani2+n(0.030),
                    titresim=0.62+ani*0.7+ani2*0.5+n(0.025),
                    voltaj=0.74+n(0.015),
                    ekstra_yuk=(ani+ani2)*1.80 + 0.12)

    elif senaryo == "titresim_artisi":
        vib = 0.32 + 0.62 * abs(math.sin(t_mod * 0.18))
        return dict(hiz=0.70+n(0.018), akim=0.52+vib*0.45+n(0.025),
                    titresim=min(0.95, vib)+n(0.018), voltaj=0.83+n(0.010),
                    ekstra_yuk=max(0, vib-0.4)*0.80 + 0.10)

    elif senaryo == "voltaj_dususu":
        volt = max(0.45, 0.88 - t_mod * 0.006)
        akim_artis = max(0.0, (0.88 - volt) * 1.6)
        return dict(hiz=0.65+n(0.015), akim=0.52+akim_artis+n(0.025),
                    titresim=0.32+akim_artis*0.40+n(0.018),
                    voltaj=volt+n(0.010),
                    ekstra_yuk=akim_artis*0.60 + 0.08)

    elif senaryo == "soguk_baslangic":
        isi_faz = min(1.0, t_mod / 40.0)
        return dict(hiz=0.38+isi_faz*0.45+n(0.018),
                    akim=0.82-isi_faz*0.32+n(0.025),
                    titresim=0.62-isi_faz*0.35+n(0.022),
                    voltaj=0.70+isi_faz*0.15+n(0.010),
                    ekstra_yuk=max(0.0, 0.35-t_mod*0.006))

    else:
        raise ValueError(f"Bilinmeyen senaryo: {senaryo}")


# ============================================================
# 11. STRES TESTİ RUNNER
# ============================================================
def stres_testi_calistir(senaryo_adi: str,
                          sure: int = 120) -> dict:
    """
    Belirli bir senaryoyu simüle eder.
    Başlangıç sıcaklığı 55°C (motor zaten ısınmış halde başlar).
    """
    sistem = MotorKontrolSistemi(model)
    sistem.reset()

    T     = 55.0   # Gerçekçi: motor çalışıyor, zaten ısınmış
    T_his = [54.0, 53.0]

    t_seri, T_seri, P_seri, R_seri, D_seri = [], [], [], [], []

    for t in range(sure):
        sv = stres_senaryosu(t, senaryo_adi)

        sensor = torch.tensor([[
            float(np.clip(sv["hiz"],      0.0, 1.0)),
            float(np.clip(T / 100.0,      0.0, 1.0)),
            float(np.clip(sv["akim"],     0.0, 1.0)),
            float(np.clip(sv["titresim"], 0.0, 1.0)),
            float(np.clip(sv["voltaj"],   0.0, 1.0)),
            float(np.clip(T_his[0]/100.0, 0.0, 1.0)),
            float(np.clip(T_his[1]/100.0, 0.0, 1.0)),
        ]], dtype=torch.float32)

        guc, gaz, ariza, durum = sistem.adim(sensor)

        T_yeni = motor_fizigi(
            T, guc,
            titresim=sv["titresim"],
            ekstra_yuk=sv["ekstra_yuk"],
        )

        t_seri.append(t)
        T_seri.append(T)
        P_seri.append(guc * 100.0)
        R_seri.append(ariza * 100.0)
        D_seri.append(durum)

        T_his = [T, T_his[0]]
        T     = T_yeni

    return dict(t=t_seri, T=T_seri, P=P_seri, R=R_seri, D=D_seri)


# ============================================================
# 12. BASELINE KARŞILAŞTIRMA RUNNER
# ============================================================
def baseline_karsilastir(sure: int = 120) -> dict:
    """
    SafeDrive-AI vs Threshold vs PID-Only vs NN-Only
    Hepsini aynı 'ani_yuk' senaryosunda karşılaştırır.
    """
    senaryo = "ani_yuk"

    # SafeDrive-AI
    ai_sistem = MotorKontrolSistemi(model)
    ai_sistem.reset()

    # Baseline-2 PID
    pid_b = PID(kp=0.5, ki=0.08, kd=0.03)
    pid_b_guc = 1.0

    T_ai   = T_thr  = T_pid  = T_nn  = 55.0
    Th_ai  = [54.0, 53.0]
    Th_thr = [54.0, 53.0]
    Th_pid = [54.0, 53.0]
    Th_nn  = [54.0, 53.0]

    res = {k: [] for k in ["t","T_ai","T_thr","T_pid","T_nn",
                             "P_ai","P_thr","P_pid","P_nn"]}

    for t in range(sure):
        sv = stres_senaryosu(t, senaryo)

        def make_sensor(T, T_h):
            return torch.tensor([[
                float(np.clip(sv["hiz"],      0.0, 1.0)),
                float(np.clip(T/100.0,        0.0, 1.0)),
                float(np.clip(sv["akim"],     0.0, 1.0)),
                float(np.clip(sv["titresim"], 0.0, 1.0)),
                float(np.clip(sv["voltaj"],   0.0, 1.0)),
                float(np.clip(T_h[0]/100.0,   0.0, 1.0)),
                float(np.clip(T_h[1]/100.0,   0.0, 1.0)),
            ]], dtype=torch.float32)

        # SafeDrive-AI
        s_ai  = make_sensor(T_ai,  Th_ai)
        P_ai, _, _, _ = ai_sistem.adim(s_ai)

        # Baseline-1: Threshold
        P_thr = baseline_threshold(T_thr)

        # Baseline-2: PID Only
        P_pid = baseline_pid_only(pid_b, T_pid, pid_b_guc)
        pid_b_guc = P_pid

        # Baseline-3: NN Only
        s_nn  = make_sensor(T_nn, Th_nn)
        P_nn  = baseline_nn_only(model, s_nn)

        # Fizik güncelle
        kw = dict(titresim=sv["titresim"], ekstra_yuk=sv["ekstra_yuk"])
        T_ai_y  = motor_fizigi(T_ai,  P_ai,  **kw)
        T_thr_y = motor_fizigi(T_thr, P_thr, **kw)
        T_pid_y = motor_fizigi(T_pid, P_pid, **kw)
        T_nn_y  = motor_fizigi(T_nn,  P_nn,  **kw)

        res["t"].append(t)
        res["T_ai"].append(T_ai);   res["P_ai"].append(P_ai*100)
        res["T_thr"].append(T_thr); res["P_thr"].append(P_thr*100)
        res["T_pid"].append(T_pid); res["P_pid"].append(P_pid*100)
        res["T_nn"].append(T_nn);   res["P_nn"].append(P_nn*100)

        Th_ai  = [T_ai,  Th_ai[0]];  T_ai  = T_ai_y
        Th_thr = [T_thr, Th_thr[0]]; T_thr = T_thr_y
        Th_pid = [T_pid, Th_pid[0]]; T_pid = T_pid_y
        Th_nn  = [T_nn,  Th_nn[0]];  T_nn  = T_nn_y

    return res


# ============================================================
# 13. STRES TESTİ GRAFİĞİ — 4 SENARYO
# ============================================================
print("Stres testi senaryoları çalıştırılıyor...")

SENARYOLAR = {
    "Normal Sürüş":    "normal",
    "Ani Yük Atışı":   "ani_yuk",
    "Titreşim Artışı": "titresim_artisi",
    "Voltaj Düşüşü":   "voltaj_dususu",
}

C_SICAK = "#e74c3c"
C_GUC   = "#2980b9"
C_RISK  = "#f39c12"

fig_stres, axes = plt.subplots(3, 4, figsize=(20, 9))
fig_stres.suptitle(
    "SafeDrive-AI  —  Stres Testi Karşılaştırması",
    fontsize=14, fontweight="bold", y=1.01
)

ROW_LABELS = ["Sıcaklık (°C)", "Motor Gücü (%)", "Arıza Riski (%)"]

for col, (isim, kod) in enumerate(SENARYOLAR.items()):
    sonuc = stres_testi_calistir(kod)

    ax_T = axes[0][col]
    ax_P = axes[1][col]
    ax_R = axes[2][col]

    # Sıcaklık
    ax_T.plot(sonuc["t"], sonuc["T"], color=C_SICAK, linewidth=2)
    ax_T.axhline(95, color="red",    linestyle="--", lw=1.2, label="95°C Krit.")
    ax_T.axhline(80, color="orange", linestyle=":",  lw=1.0, label="80°C Uyarı")
    ax_T.set_title(isim, fontsize=10, fontweight="bold", pad=6)
    ax_T.set_ylim(20, 115)
    ax_T.set_ylabel(ROW_LABELS[0] if col == 0 else "")
    ax_T.legend(fontsize=7, loc="lower right")
    ax_T.grid(True, alpha=0.3)

    # Motor gücü
    ax_P.plot(sonuc["t"], sonuc["P"], color=C_GUC, linewidth=2)
    ax_P.set_ylim(0, 108)
    ax_P.set_ylabel(ROW_LABELS[1] if col == 0 else "")
    ax_P.grid(True, alpha=0.3)

    # Arıza riski
    ax_R.fill_between(sonuc["t"], sonuc["R"], color=C_RISK, alpha=0.40)
    ax_R.plot(sonuc["t"], sonuc["R"], color=C_RISK, linewidth=1.8)
    ax_R.axhline(70, color="red", linestyle="--", lw=1, label="Risk Eşiği %70")
    ax_R.set_ylim(0, 102)
    ax_R.set_ylabel(ROW_LABELS[2] if col == 0 else "")
    ax_R.set_xlabel("Saniye")
    ax_R.legend(fontsize=7, loc="upper right")
    ax_R.grid(True, alpha=0.3)

plt.tight_layout()
dosya_stres = os.path.join(CIKTI_DIZIN, "stres_testi.png")
plt.savefig(dosya_stres, dpi=150, bbox_inches="tight")
print(f"  ✓ Stres testi grafiği kaydedildi: {dosya_stres}")


# ============================================================
# 14. BASELINE KARŞILAŞTIRMA GRAFİĞİ
# ============================================================
print("Baseline karşılaştırması çalıştırılıyor...")

bl = baseline_karsilastir()

fig_bl, (bax1, bax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig_bl.suptitle(
    "SafeDrive-AI  vs  Baseline Yöntemleri  (Ani Yük Senaryosu)",
    fontsize=13, fontweight="bold"
)

for (etiket, T_key, stil, renk) in [
    ("SafeDrive-AI", "T_ai",  "-",  "#e74c3c"),
    ("Threshold",    "T_thr", "--", "#8e44ad"),
    ("PID Only",     "T_pid", ":",  "#27ae60"),
    ("NN Only",      "T_nn",  "-.", "#f39c12"),
]:
    bax1.plot(bl["t"], bl[T_key], linestyle=stil, color=renk,
              linewidth=2, label=etiket)

bax1.axhline(95, color="red", linestyle="--", lw=1.2, alpha=0.7, label="95°C Krit.")
bax1.axhline(80, color="orange", linestyle=":", lw=1.0, alpha=0.7, label="80°C Uyarı")
bax1.set_ylabel("Sıcaklık (°C)")
bax1.set_ylim(20, 115)
bax1.legend(fontsize=9)
bax1.grid(True, alpha=0.3)

for (etiket, P_key, stil, renk) in [
    ("SafeDrive-AI", "P_ai",  "-",  "#e74c3c"),
    ("Threshold",    "P_thr", "--", "#8e44ad"),
    ("PID Only",     "P_pid", ":",  "#27ae60"),
    ("NN Only",      "P_nn",  "-.", "#f39c12"),
]:
    bax2.plot(bl["t"], bl[P_key], linestyle=stil, color=renk,
              linewidth=2, label=etiket)

bax2.set_ylabel("Motor Gücü (%)")
bax2.set_xlabel("Saniye")
bax2.set_ylim(0, 108)
bax2.legend(fontsize=9)
bax2.grid(True, alpha=0.3)

plt.tight_layout()
dosya_bl = os.path.join(CIKTI_DIZIN, "baseline_karsilastirma.png")
plt.savefig(dosya_bl, dpi=150, bbox_inches="tight")
print(f"  ✓ Baseline grafiği kaydedildi: {dosya_bl}")


# ============================================================
# 15. CAN BUS SİMÜLASYONU
# ============================================================
#   Gerçek araçlarda sensörler CAN Bus protokolü üzerinden
#   8 byte'lık mesajlar halinde iletilir. Her mesajın bir
#   ID'si, ölçek faktörü ve offset değeri vardır.
#
#   Bu modül:
#     1. Araç durumunu ham CAN frame'lerine kodlar  (encode)
#     2. Frame'leri çözerek normalize sensör verisine dönüştürür (decode)
#     3. Decode edilen veriyi doğrudan AI modeline gönderir
#
#   Kullanılan ID'ler (OBD-II / SAE J1939 standardına yakın):
#     0x0C8 → Motor RPM          (0–8000 rpm,  2 byte)
#     0x130 → Soğutucu Sıcaklığı (0–150 °C,   1 byte)
#     0x1A0 → Gaz Kelebeği %     (0–100 %,     1 byte)
#     0x3B0 → Batarya Voltajı    (0–16 V,      1 byte, x0.1)
#     0x4A0 → Titreşim           (0–100 g,     1 byte, x0.01)
#     0x5C0 → Motor Akımı        (0–500 A,     2 byte, x0.1)
#     0x6D0 → Araç Hızı          (0–300 km/h,  1 byte)
# ============================================================

# CAN ID tanımları: {id: (isim, max_değer, byte_sayısı, ölçek)}
CAN_TANIM = {
    0x0C8: ("RPM",      8000.0, 2, 1.0  ),
    0x130: ("SICAKLIK", 150.0,  1, 1.0  ),
    0x1A0: ("GAZ",      100.0,  1, 1.0  ),
    0x3B0: ("VOLTAJ",   16.0,   1, 0.1  ),
    0x4A0: ("TITRESIM", 100.0,  1, 0.01 ),
    0x5C0: ("AKIM",     500.0,  2, 0.1  ),
    0x6D0: ("HIZ",      300.0,  1, 1.0  ),
}


def can_encode(rpm, sicaklik, gaz, voltaj, titresim, akim, hiz):
    """
    Gerçek değerleri CAN frame byte dizilerine kodlar.
    Her frame 8 byte; kullanılmayan byte'lar 0x00 ile doldurulur.
    """
    def enc(val, max_val, n_byte, scale):
        raw = int(round(float(np.clip(val, 0, max_val)) / scale))
        raw = min(raw, (256 ** n_byte) - 1)
        return raw.to_bytes(n_byte, byteorder="big") + bytes(8 - n_byte)

    return {
        0x0C8: enc(rpm,            8000.0, 2, 1.0),
        0x130: enc(sicaklik,        150.0, 1, 1.0),
        0x1A0: enc(gaz,             100.0, 1, 1.0),
        0x3B0: enc(voltaj * 10,     160.0, 1, 1.0),   # 0.1V çözünürlük
        0x4A0: enc(titresim * 100, 10000,  1, 1.0),   # 0.01g çözünürlük
        0x5C0: enc(akim * 10,      5000.0, 2, 1.0),   # 0.1A çözünürlük
        0x6D0: enc(hiz,             300.0, 1, 1.0),
    }


def can_decode(frames):
    """
    CAN frame byte dizilerini gerçek fiziksel değerlere çözer.
    Döndürür: {isim: gerçek_değer} sözlüğü
    """
    sonuc = {}
    ozel_scale = {
        0x3B0: 0.1,    # voltaj: raw * 0.1 → Volt
        0x4A0: 0.01,   # titreşim: raw * 0.01 → g
        0x5C0: 0.1,    # akım: raw * 0.1 → Amper
    }
    for can_id, (isim, max_val, n_byte, _) in CAN_TANIM.items():
        raw  = int.from_bytes(frames[can_id][:n_byte], byteorder="big")
        sc   = ozel_scale.get(can_id, 1.0)
        sonuc[isim] = float(np.clip(raw * sc, 0.0, max_val))
    return sonuc


def can_to_sensor(decoded, T1_derece, T2_derece):
    """
    Decode edilmiş CAN verisini AI modeline girecek
    normalize (0-1) tensöre dönüştürür.
    """
    return torch.tensor([[
        decoded["HIZ"]      / 300.0,    # hız
        decoded["SICAKLIK"] / 150.0,    # T(t)
        decoded["AKIM"]     / 500.0,    # akım
        decoded["TITRESIM"] / 100.0,    # titreşim
        decoded["VOLTAJ"]   / 16.0,     # voltaj
        T1_derece           / 150.0,    # T(t-1)
        T2_derece           / 150.0,    # T(t-2)
    ]], dtype=torch.float32)


def can_bus_adim(t, senaryo, sicaklik, T1, T2, sistem):
    """
    Tek bir CAN Bus döngüsü:
      encode → simüle CAN hattı → decode → tensor → AI+PID karar
    Döndürür: (guc, ariza, durum, decoded_dict, ekstra_yuk)
    """
    sv = stres_senaryosu(t, senaryo)

    # Normalize sensör → gerçek birime çevir
    rpm        = sv["hiz"]      * 8000.0
    gaz_pct    = max(0.0, (1.0 - sv["akim"] * 0.3)) * 100.0
    voltaj_v   = sv["voltaj"]   * 16.0
    titresim_g = sv["titresim"] * 100.0
    akim_a     = sv["akim"]     * 500.0
    hiz_kmh    = sv["hiz"]      * 300.0

    frames  = can_encode(rpm, sicaklik, gaz_pct,
                         voltaj_v, titresim_g, akim_a, hiz_kmh)
    decoded = can_decode(frames)
    sensor  = can_to_sensor(decoded, T1, T2)

    guc, gaz, ariza, durum = sistem.adim(sensor)
    return guc, ariza, durum, decoded, sv["ekstra_yuk"]


# ---- CAN Bus Demo: 5 adım terminale yazdır ----
print("-" * 55)
print("  CAN Bus Simülasyon Demo  (5 adım, ani_yuk senaryosu)")
print("-" * 55)
_demo_sys = MotorKontrolSistemi(model)
_demo_sys.reset()
_T, _T1, _T2 = 55.0, 54.0, 53.0
for _i in range(5):
    _g, _a, _d, _dec, _ = can_bus_adim(
        _i, "ani_yuk", _T, _T1, _T2, _demo_sys
    )
    print(f"  [CAN t={_i:>2}s] "
          f"RPM:{_dec['RPM']:>5.0f} | "
          f"T:{_dec['SICAKLIK']:>5.1f}°C | "
          f"V:{_dec['VOLTAJ']:>4.1f}V | "
          f"Tit:{_dec['TITRESIM']:>5.1f}g | "
          f"Akm:{_dec['AKIM']:>5.1f}A | "
          f"Guc:%{_g*100:>5.1f} | "
          f"Risk:%{_a*100:>4.1f} | {_d}")
    _T2 = _T1
    _T1 = _T
    _T  = motor_fizigi(_T, _g, titresim=_dec["TITRESIM"] / 100.0)
print()


# ============================================================
# 16. CANLI DASHBOARD — TAM DÜZELTILMIŞ
# ============================================================
#   Eski hatalar:
#     • animate() global değişkenlere bağlıydı → animasyon donuyordu
#     • ariza_risk = random.uniform(0,10) → sahte veri
#     • 'power' scope hatası → NameError
#   Düzeltmeler:
#     • Tüm state dash_state sözlüğünde → her frame kendi adımını hesaplıyor
#     • ariza gerçek model çıktısından geliyor
#     • Hiçbir global değişkene bağımlılık yok
# ============================================================

AKTIF_SENARYO = "ani_yuk"   # değiştirilebilir: normal / ani_yuk /
                             # titresim_artisi / voltaj_dususu / soguk_baslangic
PENCERE_BOYUTU = 60          # grafik penceresinde görünecek adım sayısı

# Dashboard state: her şey burada, global değişken yok
dash = {
    "sistem": MotorKontrolSistemi(model),
    "T":      55.0,
    "T_his":  [54.0, 53.0],
    "t":      0,
    "x":      [], "T_y": [], "P_y": [], "R_y": [],
}
dash["sistem"].reset()

fig_dash = plt.figure(figsize=(12, 8))
fig_dash.suptitle(
    f"SafeDrive-AI  —  Canlı Dashboard (CAN Bus)  |  Senaryo: {AKTIF_SENARYO.upper()}",
    fontsize=12, fontweight="bold"
)
gs  = gridspec.GridSpec(3, 1, hspace=0.50)
da1 = fig_dash.add_subplot(gs[0])
da2 = fig_dash.add_subplot(gs[1])
da3 = fig_dash.add_subplot(gs[2])

def animate(frame):
    d = dash
    t = d["t"]
    T = d["T"]

    # CAN Bus pipeline: encode → decode → tensor → AI+PID
    guc, ariza, durum, decoded, ekstra_yuk = can_bus_adim(
        t, AKTIF_SENARYO,
        sicaklik = T,
        T1       = d["T_his"][0],
        T2       = d["T_his"][1],
        sistem   = d["sistem"],
    )

    T_yeni = motor_fizigi(
        T, guc,
        titresim  = decoded["TITRESIM"] / 100.0,
        ekstra_yuk= ekstra_yuk,
    )

    # State güncelle
    d["T_his"] = [T, d["T_his"][0]]
    d["T"]     = T_yeni
    d["t"]    += 1

    d["x"].append(t);          d["T_y"].append(T)
    d["P_y"].append(guc*100);  d["R_y"].append(ariza*100)

    # Kayan pencere
    if len(d["x"]) > PENCERE_BOYUTU:
        for lst in (d["x"], d["T_y"], d["P_y"], d["R_y"]):
            lst.pop(0)

    xv = d["x"]

    # ---- Sıcaklık ----
    da1.cla()
    da1.plot(xv, d["T_y"], color=C_SICAK, linewidth=2, label="Sıcaklık (°C)")
    da1.axhline(95, color="red",    linestyle="--", lw=1.2, label="95°C Krit.")
    da1.axhline(80, color="orange", linestyle=":",  lw=1.0, label="80°C Uyarı")
    da1.set_ylabel("Sıcaklık")
    da1.set_ylim(18, 115)
    da1.legend(loc="upper left", fontsize=8)
    da1.grid(True, alpha=0.3)
    da1.set_title(
        f"t={t}s  |  Isı: {T:.1f}°C  |  RPM: {decoded['RPM']:.0f}"
        f"  |  Güç: %{guc*100:.1f}  |  Durum: {durum}",
        fontsize=9
    )

    # ---- Motor gücü ----
    da2.cla()
    da2.plot(xv, d["P_y"], color=C_GUC, linewidth=2, label="Motor Gücü (%)")
    da2.set_ylabel("Güç %")
    da2.set_ylim(0, 108)
    da2.legend(loc="upper left", fontsize=8)
    da2.grid(True, alpha=0.3)

    # ---- Arıza riski (modelden) ----
    da3.cla()
    da3.fill_between(xv, d["R_y"], color=C_RISK, alpha=0.40, label="Arıza Riski %")
    da3.plot(xv, d["R_y"], color=C_RISK, linewidth=1.8)
    da3.axhline(70, color="red", linestyle="--", lw=1, label="Risk Eşiği %70")
    da3.set_ylabel("Risk %")
    da3.set_ylim(0, 102)
    da3.set_xlabel("Saniye")
    da3.legend(loc="upper left", fontsize=8)
    da3.grid(True, alpha=0.3)


ani = FuncAnimation(
    fig_dash, animate,
    interval=300,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()
