import numpy as np
import astropy.io.fits as fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Z3 IceCube 分析 V2：直接谐波提取法 ===")

# 1. 读取数据
try:
    hdul = fits.open('IC86_sid.fits')
    data = hdul[1].data
    Q = data['Q_POLARISATION']
    U = data['U_POLARISATION']
    P = np.sqrt(Q**2 + U**2)
    print("数据加载成功。")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# 2. 空间平均，保留时间轴 (48个点)
# P.shape = (48, 1024) -> (48,)
flux_time = np.mean(P, axis=1)

# 3. 归一化 (Relative Variation)
flux_norm = (flux_time - np.mean(flux_time)) / np.mean(flux_time)

# 4. FFT (直接变换)
# rfft 返回的数组长度为 48/2 + 1 = 25
fft_spectrum = np.abs(np.fft.rfft(flux_norm))**2

# 5. 提取关键谐波
# Index i 代表 i cycles/day
p_24h = fft_spectrum[1] # Index 1
p_12h = fft_spectrum[2] # Index 2
p_6h  = fft_spectrum[4] # Index 4 (Z3 目标)

# 计算信噪比 (SNR)
# 用高频部分 (Index 5 到 24) 作为噪声基底
noise_floor = np.mean(fft_spectrum[5:])
snr_6h = p_6h / noise_floor

print("-" * 50)
print(f"24h (Dipole)    Power: {p_24h:.2e}")
print(f"12h (Atmosphere) Power: {p_12h:.2e}")
print(f" 6h (Z3 Rank-4)  Power: {p_6h:.2e}")
print("-" * 50)
print(f"噪声基底 (Noise Floor): {noise_floor:.2e}")
print(f"6h 信号信噪比 (SNR)   : {snr_6h:.2f}")
print("-" * 50)

# 6. 绘图验证
fig, ax = plt.subplots(figsize=(10, 6))
indices = np.arange(len(fft_spectrum))
ax.bar(indices, fft_spectrum, color='gray', alpha=0.5, label='Noise')
ax.bar(1, p_24h, color='green', label='24h (Dipole)')
ax.bar(2, p_12h, color='orange', label='12h (Quad)')
ax.bar(4, p_6h, color='red', width=0.6, label='6h (Z3 Target)')

ax.set_xlabel('Harmonic (Cycles per Day)')
ax.set_ylabel('Power')
ax.set_title('IceCube Sidereal Modulation Spectrum')
ax.set_xticks(range(0, 13))
ax.set_xlim(0.5, 12.5) # 忽略 DC
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('Z3_IceCube_Spectrum_V2.png', dpi=300)
print("图表已保存: Z3_IceCube_Spectrum_V2.png")