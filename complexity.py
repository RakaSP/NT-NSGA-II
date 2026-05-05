import math
import numpy as np
import matplotlib.pyplot as plt

# Proxy ilustratif jumlah solusi:
# 1 kendaraan : n!
# 7 kendaraan : 7^n * n!
# Ini bukan rumus eksak VRP, hanya ilustrasi ruang pencarian / jumlah solusi.

jumlah_kendaraan = 7
nodes = list(range(2, 79))  # 2 sampai 78 customer

# Gamma mengatur seberapa kuat perbedaan orde besar diperlebar.
# gamma = 1.0 mirip logscale biasa.
# gamma > 1 membuat area orde besar mendapat tinggi lebih besar.
gamma = 3

def log10_factorial(n):
    return math.lgamma(n + 1) / math.log(10)

def log10_solusi_1_kendaraan(n):
    return log10_factorial(n)

def log10_solusi_7_kendaraan(n, k):
    return n * math.log10(k) + log10_factorial(n)

def transform_y(log10_value):
    return log10_value ** gamma

log_y_1 = [log10_solusi_1_kendaraan(n) for n in nodes]
log_y_7 = [log10_solusi_7_kendaraan(n, jumlah_kendaraan) for n in nodes]

y_1_plot = [transform_y(v) for v in log_y_1]
y_7_plot = [transform_y(v) for v in log_y_7]

plt.figure(figsize=(11, 6))

plt.plot(
    nodes,
    y_1_plot,
    linewidth=2.5,
    label="1 kendaraan"
)

plt.plot(
    nodes,
    y_7_plot,
    linewidth=2.5,
    label="7 kendaraan"
)

# Sorot kasus penelitian
n_kasus = 78
log_y_kasus = log10_solusi_7_kendaraan(n_kasus, jumlah_kendaraan)
y_kasus_plot = transform_y(log_y_kasus)


plt.xlabel("Jumlah node/customer")
plt.ylabel("Jumlah solusi yang memungkinkan")

# Y-ticks dihitung otomatis berdasarkan rentang eksponen, bukan manual posisi.
max_exp = math.ceil(max(log_y_7))
tick_exponents = np.arange(0, max_exp + 1, 40)

tick_positions = [transform_y(e) for e in tick_exponents]
tick_labels = [fr"$10^{{{int(e)}}}$" for e in tick_exponents]

plt.yticks(tick_positions, tick_labels)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("grafik_vrp.png", dpi=300, bbox_inches="tight")
plt.show()