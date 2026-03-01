# ==========================================================
# 1. YENİ MİAMRİ STAGE/CONTROL AYRIMI
# ==========================================================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================================
# 1. MOTOR KORUYUCU AI (STATE / CONTROL AYRIMLI)
class MotorKoruyucuAI(nn.Module):
    def __init__(self):
        super().__init__()

        # -------- STATE (Derin kısım – büyüyebilir) --------
        self.state_net = nn.Sequential(
            nn.Linear(5, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh()
        )

        # -------- CONTROL (Sade ve güvenli) --------
        self.control_net = nn.Linear(16, 1)

    def forward(self, x):
        state = self.state_net(x)

        # Sıcaklığı direkt geçir (fiziksel öncelik)
        state = state + x[:, 1:2]

        out = self.control_net(state)
        return torch.sigmoid(out)
# ==========================================================
# 2. SÜREKLİ (ANALOG) EĞİTİM VERİS
# ==========================================================
def veri_uret():
    X, Y = [], []

    for t in range(40, 96):
        sic = t / 100.0

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

        X.append([0.5, sic, 0.3, 0.2, 0.7])
        Y.append([kesme])

    return torch.tensor(X), torch.tensor(Y)
# ==========================================================
# 3. EĞİTİM AYARLARI(BU SEFER PANİKLEYİP ÇAT DİYE ŞARTEL KAPATAMAYACAK)
# ==========================================================
model = MotorKoruyucuAI()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

criterion = nn.SmoothL1Loss()

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=200,
    gamma=0.5
)

X_train, Y_train = veri_uret()
# ==========================================================
# 4. EĞİTİM DÖNGÜSÜ
# ==========================================================
print("\n--- MOTOR KORUYUCU AI EGITIMI BASLIYOR ---")

for adim in range(500):
    optimizer.zero_grad()

    preds = model(X_train)
    loss = criterion(preds, Y_train)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if (adim + 1) % 100 == 0:
        print(f"Adim {adim+1}: Loss = {loss.item():.6f}")

print("--- EGITIM TAMAMLANDI ---\n")
# ==========================================================
# 5. TEST(82 DERECE SENARYOSU)
# ==========================================================
test_derece = 82
test_veri = torch.tensor([[0.7, test_derece/100, 0.6, 0.4, 0.8]])

karar = model(test_veri).item()

print(f"Motor Sicakligi : {test_derece} °C")
print(f"Gaz Kesme Karari: %{karar*100:.2f}")

if karar > 0.8:
    print("DURUM: ACIL DURUS")
elif karar > 0.1:
    print("DURUM: KADEMELI SOGUTMA")
else:
    print("DURUM: SISTEM STABIL")
# ==========================================================
# 6. KARAR EĞRİSİ (EN KRİTİK YER)
# ==========================================================
dereceler = torch.linspace(0.4, 0.95, 100)
tepkiler = [
    model(torch.tensor([[0.5, d, 0.3, 0.2, 0.7]])).item()
    for d in dereceler
]

plt.plot(dereceler.numpy()*100, tepkiler)
plt.axvline(80, linestyle="--")
plt.axvline(95, linestyle="--")
plt.xlabel("Sicaklik (C)")
plt.ylabel("Gaz Kesme Orani")
plt.title("Motor Koruma AI – State/Control Refactor")
plt.grid(True)
plt.show()