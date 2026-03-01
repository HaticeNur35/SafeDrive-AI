# ============================================================
# 1.MODEL MULTİ-HEAD MİMARİ
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

class MotorKoruyucuAI(nn.Module):
    def __init__(self):
        super().__init__()

        # =========================
        # STATE (ORTAK ZİHİN)
        # =========================
        self.state_net = nn.Sequential(
            nn.Linear(7, 32),   # 5 sensör + T(t-1) + T(t-2)
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh()
        )

        # =========================
        # HEAD 1: GAZ KESME (CONTROL)
        # =========================
        self.control_head = nn.Linear(16, 1)

        # =========================
        # HEAD 2: MEKANIK ARIZA RİSKİ
        # =========================
        self.fault_head = nn.Linear(16, 1)

    def forward(self, x):
        state = self.state_net(x)

        # Fiziksel öncelik: sıcaklık state'e skip
        state = state + x[:, 1:2]

        gaz_kesme = torch.sigmoid(self.control_head(state))
        ariza_risk = torch.sigmoid(self.fault_head(state))

        return gaz_kesme, ariza_risk
# ============================================================
# 2. EĞİTİM VERİSİ – ARIZA ETİKETİ EKLEME
# ============================================================
def veri_uret_multihead():
    X, Y_gaz, Y_ariza = [], [], []

    for t in range(42, 96):
        T0 = t / 100.0
        T1 = (t - 1) / 100.0
        T2 = (t - 2) / 100.0

        # -------- Gaz kesme etiketi (aynı mantık) --------
        if t < 70:
            kesme = 0.0
        elif t < 75:
            kesme = (t - 70) * 0.04
        elif t < 85:
            kesme = 0.2 + (t - 75) * 0.02
        elif t < 95:
            kesme = 0.4 + (t - 85) * 0.06
        else:
            kesme = 1.0

        # -------- Mekanik arıza etiketi --------
        titresim = 0.2 if t < 80 else 0.6
        akim = 0.3 if t < 80 else 0.7

        if titresim > 0.5 and akim > 0.6:
            ariza = 1.0
        else:
            ariza = 0.0

        X.append([0.5, T0, akim, titresim, 0.7, T1, T2])
        Y_gaz.append([kesme])
        Y_ariza.append([ariza])

    return (
        torch.tensor(X),
        torch.tensor(Y_gaz),
        torch.tensor(Y_ariza)
    )
# ============================================================
# 3.EĞİTİM  (İKİ LOSS,TEK GERİ YAYILIM)
# ============================================================
model = MotorKoruyucuAI()

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

loss_control = nn.SmoothL1Loss()
loss_fault   = nn.BCELoss()

X, Y_gaz, Y_ariza = veri_uret_multihead()

for epoch in range(600):
    optimizer.zero_grad()

    gaz_pred, ariza_pred = model(X)

    loss1 = loss_control(gaz_pred, Y_gaz)
    loss2 = loss_fault(ariza_pred, Y_ariza)

    loss = loss1 + 0.5 * loss2   # arıza biraz daha sakin öğrensin
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1} | GazLoss: {loss1:.4f} | ArizaLoss: {loss2:.4f}")
# ============================================================
# 4. TEST – AYNI SICAKLIK, FARKLI ANLAM
# ============================================================
def test_multi(ad, veri):
    gaz, ariza = model(veri)
    gaz = gaz.item()
    ariza = ariza.item()

    print("--------------------------------------------------")
    print(f"SENARYO        : {ad}")
    print(f"Gaz Kesme      : %{gaz*100:.2f}")
    print(f"Ariza Riski    : %{ariza*100:.2f}")

    if ariza > 0.7:
        print("UYARI: MEKANIK ARIZA RİSKİ!")
    elif gaz > 0.1:
        print("DURUM: KADEMELI SOGUTMA")
    else:
        print("DURUM: SISTEM STABIL")


# Normal
test_multi(
    "Normal",
    torch.tensor([[0.7, 0.75, 0.3, 0.2, 0.7, 0.74, 0.73]])
)

# Titreşimli ama soğuk
test_multi(
    "Titreşim Var Ama Soğuk",
    torch.tensor([[0.7, 0.65, 0.7, 0.7, 0.7, 0.64, 0.63]])
)

# Sıcak + titreşimli
test_multi(
    "Sıcak ve Titreşimli",
    torch.tensor([[0.7, 0.88, 0.7, 0.7, 0.7, 0.87, 0.86]])
)
#============================================================
#5. PID DENETLEYICI
#============================================================
class PID:
    def __init__(self, kp, ki, kd, dt=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
#============================================================
#6. FAIL-SAFE SERT SINIRLAR (RUNTIME)
#============================================================
def fail_safe_check(sensor, gaz_kesme, ariza_risk):
    hiz, T0, akim, titresim, voltaj, T1, T2 = sensor[0]

    # Sensör kopması
    if torch.isnan(sensor).any() or torch.isinf(sensor).any():
        return True, "SENSOR HATASI"

    # Sert sıcaklık limiti
    if T0 * 100 >= 100:
        return True, "ASIRI SICAKLIK"

    # Aşırı titreşim
    if titresim > 0.85:
        return True, "ASIRI TITRESIM"

    # Mekanik arıza riski çok yüksekse
    if ariza_risk > 0.9:
        return True, "MEKANIK ARIZA"

    return False, "OK"
#============================================================
#7. ANA KONTROL DONGUSU (PID + NN + FAIL-SAFE)
#============================================================
class MotorControlSystem:
    def __init__(self, model):
        self.model = model
        self.pid = PID(kp=0.8, ki=0.1, kd=0.05)
        self.current_power = 1.0  # %100 güç

    def step(self, sensor_input):
        with torch.no_grad():
            gaz_kesme, ariza = self.model(sensor_input)

        gaz_kesme = gaz_kesme.item()
        ariza = ariza.item()

        # ---------- FAIL SAFE ----------
        stop, reason = fail_safe_check(sensor_input, gaz_kesme, ariza)
        if stop:
            self.current_power = 0.0
            return self.current_power, gaz_kesme, ariza, f"FAIL-SAFE: {reason}"

        # ---------- NN → HEDEF ----------
        target_power = 1.0 - gaz_kesme

        # ---------- PID → UYGULAMA ----------
        pid_out = self.pid.step(
            setpoint=target_power,
            measurement=self.current_power
        )

        self.current_power += pid_out
        self.current_power = max(0.0, min(1.0, self.current_power))

        return self.current_power, gaz_kesme, ariza, "NORMAL"
#============================================================
#8. ZAMAN ADIMLI CALISMA TESTI (NIHAI)
#============================================================
system = MotorControlSystem(model)

print("\n--- PID + NN + FAIL-SAFE CALISMA TESTI ---")

sicakliklar = [70, 72, 75, 78, 80, 82, 85, 88, 92, 97]

T1, T2 = 0.69, 0.68

for t in sicakliklar:
    T0 = t / 100.0

    sensor = torch.tensor([[
        0.7,   # hiz
        T0,    # T(t)
        0.6,   # akim
        0.4,   # titresim
        0.8,   # voltaj
        T1,    # T(t-1)
        T2     # T(t-2)
    ]])

    power, gaz, ariza, durum = system.step(sensor)

    print(
        f"T={t:>3}C | "
        f"NN Kesme=%{gaz*100:>5.1f} | "
        f"Guc=%{power*100:>5.1f} | "
        f"Ariza=%{ariza*100:>5.1f} | "
        f"{durum}"
    )

    T2 = T1
    T1 = T0
#============================================================
#. KARAR EĞRİSİ + PID CEVABI (GRAFİK KANITI)
#===========================================================
import matplotlib.pyplot as plt

def plot_karar_ve_pid(system, model):
    sicakliklar = list(range(60, 101))  # 60C → 100C

    nn_kesme = []
    pid_guc  = []

    # PID'yi temiz başlat
    system.current_power = 1.0
    system.pid.integral = 0.0
    system.pid.prev_error = 0.0

    # başlangıç geçmişi
    T1, T2 = 0.59, 0.58

    for t in sicakliklar:
        T0 = t / 100.0

        sensor = torch.tensor([[
            0.7,   # hiz
            T0,    # T(t)
            0.6,   # akim
            0.4,   # titresim
            0.8,   # voltaj
            T1,    # T(t-1)
            T2     # T(t-2)
        ]])

        with torch.no_grad():
            gaz_kesme, _ = model(sensor)

        power, _, _, _ = system.step(sensor)

        nn_kesme.append(gaz_kesme.item() * 100)
        pid_guc.append(power * 100)

        T2 = T1
        T1 = T0

    # ---------------- GRAFIK ----------------
    plt.figure(figsize=(10, 6))

    plt.plot(sicakliklar, nn_kesme,
             label="NN Gaz Kesme Karari (%)",
             linestyle="--")

    plt.plot(sicakliklar, pid_guc,
             label="PID Sonrasi Gercek Motor Gucu (%)",
             linewidth=2)

    plt.axvline(80, linestyle=":", label="80°C Uyari Esigi")
    plt.axvline(95, linestyle=":", label="95°C Kritik Bolge")

    plt.xlabel("Motor Sicakligi (°C)")
    plt.ylabel("Yuzde (%)")
    plt.title("NN Karar Eğrisi vs PID Sonrasi Motor Tepkisi")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_karar_ve_pid(system, model)
#============================================================
#10. SENSÖR GÜRÜLTÜSÜ ALTINDA STABILITE TESTI
#============================================================
import random

def plot_gurultulu_sistem(system, model, noise_std=0.02):
    sicakliklar = list(range(60, 101))

    nn_kesme = []
    pid_guc  = []

    # PID reset
    system.current_power = 1.0
    system.pid.integral = 0.0
    system.pid.prev_error = 0.0

    T1, T2 = 0.59, 0.58

    for t in sicakliklar:
        T0 = t / 100.0

        # ---- GURULTU EKLE ----
        noisy_T0 = T0 + random.gauss(0, noise_std)
        noisy_T1 = T1 + random.gauss(0, noise_std)
        noisy_T2 = T2 + random.gauss(0, noise_std)

        noisy_akim = 0.6 + random.gauss(0, noise_std)
        noisy_tit  = 0.4 + random.gauss(0, noise_std)
        noisy_volt = 0.8 + random.gauss(0, noise_std)

        sensor = torch.tensor([[
            0.7,
            noisy_T0,
            noisy_akim,
            noisy_tit,
            noisy_volt,
            noisy_T1,
            noisy_T2
        ]])

        with torch.no_grad():
            gaz_kesme, _ = model(sensor)

        power, _, _, _ = system.step(sensor)

        nn_kesme.append(gaz_kesme.item() * 100)
        pid_guc.append(power * 100)

        T2 = T1
        T1 = T0

    # -------- GRAFIK --------
    plt.figure(figsize=(10, 6))

    plt.plot(sicakliklar, nn_kesme,
             label="NN Gaz Kesme (Gurultulu)",
             linestyle="--",
             alpha=0.7)

    plt.plot(sicakliklar, pid_guc,
             label="PID Sonrasi Guc (Gurultulu)",
             linewidth=2)

    plt.xlabel("Motor Sicakligi (°C)")
    plt.ylabel("Yuzde (%)")
    plt.title("Sensör Gürültüsü Altında NN + PID Davranışı")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_gurultulu_sistem(system, model, noise_std=0.02)
#============================================================
#BASELINE KONTROL YÖNTEMLERİ (TOPLU)
#============================================================
def baseline_threshold(sensor):
    """
    Baseline-1: Hard Threshold Control
    Geleneksel eşik tabanlı motor koruma.
    Ani ve sert güç düşüşleri üretir.
    """
    T0 = sensor[0][1] * 100  # anlık sıcaklık

    if T0 >= 95:
        return 0.0      # %100 kes
    elif T0 >= 80:
        return 0.6      # %40 güç
    else:
        return 1.0      # tam güç


def baseline_pid(pid, sensor, current_power):
    """
    Baseline-2: PID Only Control
    Öğrenme veya arıza bilgisi olmayan klasik PID.
    """
    T0 = sensor[0][1] * 100
    target_temp = 75  # sabit hedef sıcaklık

    error = target_temp - T0
    pid_out = pid.step(error, 0)

    power = current_power + pid_out
    return max(0.0, min(1.0, power))


def baseline_nn_only(model, sensor):
    """
    Baseline-3: Neural Network Only Control
    PID yumuşatması olmayan doğrudan NN kararı.
    """
    with torch.no_grad():
        gaz_kesme, _ = model(sensor)

    return 1.0 - gaz_kesme.item()
####################################################
#MOTOR FİZİK DENKLEMİ(EULER METODU)
####################################################
def motor_fizigi_simule_et(su_anki_T, anlik_guc, ortam_T=25.0, dt=1.0):
    """
    Basit Termal Diferansiyel Denklem Çözümü
    dT/dt = alpha * Guc - beta * (T - T_ortam)
    """
    alpha = 0.6  # Isınma katsayısı (Motorun termal yükü)
    beta = 0.08  # Soğuma katsayısı (Fan ve radyatör etkisi)
    
    # Isınma etkisi
    isinma = alpha * anlik_guc
    # Newton'un Soğuma Yasası (Sıcaklık farkı arttıkça soğuma artar)
    soguma = beta * (su_anki_T - ortam_T)
    
    yeni_T = su_anki_T + (isinma - soguma) * dt
    return yeni_T
####################################################
#DİNAMİK SİMÜLASYON DÖNGÜSÜ
####################################################
# --- SİMÜLASYON BAŞLATMA ---
system = MotorControlSystem(model) # Senin eğittiğin model
anlik_T = 60.0  # Başlangıç sıcaklığı
T_gecmis = [59.5, 59.0] # T(t-1) ve T(t-2) için başlangıç
zaman_serisi = []
sicaklik_serisi = []
guc_serisi = []

print("\n--- AI + PID CANLI MOTOR KONTROLÜ BAŞLIYOR ---")

for t in range(120): # 120 saniyelik simülasyon
    # 1. Sensör Verisi Hazırlama (Normalleştirilmiş 0-1 arası)
    sensor = torch.tensor([[
        0.7,             # Sabit Hız
        anlik_T / 100.0, # T(t)
        0.6,             # Akım
        0.4,             # Titreşim
        0.8,             # Voltaj
        T_gecmis[0] / 100.0, # T(t-1)
        T_gecmis[1] / 100.0  # T(t-2)
    ]])

    # 2. AI ve PID Karar Versin
    power, gaz_kesme, ariza, durum = system.step(sensor)

    # 3. Verileri Kaydet
    zaman_serisi.append(t)
    sicaklik_serisi.append(anlik_T)
    guc_serisi.append(power * 100)

    # 4. FİZİKSEL GÜNCELLEME (Motorun tepkisi)
    yeni_T = motor_fizigi_simule_et(anlik_T, power)
    
    # Geçmişi güncelle (AI'nın momentumu anlaması için)
    T_gecmis = [anlik_T, T_gecmis[0]]
    anlik_T = yeni_T

    if t % 10 == 0:
        print(f"Saniye {t:>3} | Isı: {anlik_T:.1f}°C | Güç: %{power*100:.1f} | Durum: {durum}")
####################################################
#ANALİZ GRAFİĞİ
####################################################
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(zaman_serisi, sicaklik_serisi, 'r-', label="Motor Sıcaklığı (°C)", linewidth=2)
ax2.plot(zaman_serisi, guc_serisi, 'b--', label="Motor Gücü (%)", alpha=0.7)

ax1.axhline(80, color='orange', linestyle=':', label="Uyarı Eşiği (80°C)")
ax1.axhline(95, color='red', linestyle=':', label="Kritik Sınır (95°C)")

ax1.set_xlabel("Zaman (saniye)")
ax1.set_ylabel("Sıcaklık (°C)", color='r')
ax2.set_ylabel("Motor Gücü (%)", color='b')
plt.title("Dinamik Termal Simülasyon: AI + PID Kapalı Döngü Kontrolü")
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 1. Etkinleşme Noktası
plt.annotate('AI Müdahale Başlangıcı', 
             xy=(80, 93), xytext=(65, 85),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# 2. Kritik Bölge
plt.annotate('Kritik Soğutma', 
             xy=(95, 82), xytext=(85, 65),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1))

# 3. Fail-Safe (Grafiğin en sonu, dikey düşüşün olduğu yer)
plt.annotate('FAIL-SAFE: ACİL DURDURMA', 
             xy=(100, 5), xytext=(75, 25),
             arrowprops=dict(facecolor='darkred', shrink=0.05, width=2),
             color='red', fontweight='bold')

# Mevcut plt.show() şimdi 535-540. satırlara kaymış olacak.
plt.show()
plt.show()