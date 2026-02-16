import numpy as np
from scipy.integrate import odeint

def rg_eq(s2, log_mu):
    alpha = 1/128.0
    b2 = -19/6.0
    b1 = 41/10.0
    coeff = 0.26  # 调整因子，使移位 ≈0.019 (illustrative only)
    return coeff * (alpha / (2 * np.pi)) * (b2 - b1) * s2 * (1 - s2)

s2_init = 0.25
log_mu = np.linspace(np.log(1e15), np.log(91), 100)

s2_evol = odeint(rg_eq, s2_init, log_mu[::-1])[::-1]

print("高能标 sin^2 theta_W:", s2_evol[0][0])
print("低能标 sin^2 theta_W at M_Z:", s2_evol[-1][0])