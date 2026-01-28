import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import fsolve

def f(S, beta=1.0):
    return np.where(S > 0, beta * np.tanh(S), 0)

def df(S, beta=1.0):
    return np.where(S > 0, beta * (1 - np.tanh(S)**2), 0)

def get_correlations(delta_EI, delta_IE, chi_E, chi_I, scenario_id):
    alpha, beta, h = 0.1, 1.0, 1e-6
    wEE, wII = 6.95, 6.85
    wEI, wIE = wII + delta_EI, wEE + delta_IE

    def equations(p):
        E, I = p
        sE = wEE * E - wEI * I + h
        sI = wIE * E - wII * I + h
        return [-alpha * E + (1 - E) * f(sE), -alpha * I + (1 - I) * f(sI)]

    start_pt = [0.1, 0.1] if scenario_id == 'C' else [0.9, 0.9]
    E0, I0 = fsolve(equations, start_pt)
    
    sE0 = wEE * E0 - wEI * I0 + h
    sI0 = wIE * E0 - wII * I0 + h
    fpE, fpI = df(sE0), df(sI0)
    
    A_EE = -alpha - f(sE0) + (1 - E0) * wEE * fpE
    A_EI = -(1 - E0) * wEI * fpE
    A_IE = (1 - I0) * wIE * fpI
    A_II = -alpha - f(sI0) - (1 - I0) * wII * fpI

    rE, rI = chi_E/chi_I, chi_I/chi_E

    x = 0.5 * (A_EE + (rE**1.5)*A_EI + (rI**1.5)*A_IE + A_II)
    y = 0.5 * (A_EE - (rE**1.5)*A_EI + (rI**1.5)*A_IE - A_II)
    z = 0.5 * (A_EE + (rE**1.5)*A_EI - (rI**1.5)*A_IE - A_II)
    w = 0.5 * (A_EE - (rE**1.5)*A_EI - (rI**1.5)*A_IE + A_II)
    A_mat = np.array([[x, y], [z, w]])

    G, H = alpha * (E0 + I0), alpha * (E0 - I0)
    M_noise = np.array([[G, H], [H, G]]) # Eq. (A26) [cite: 2119]
    sigma = solve_continuous_lyapunov(A_mat, -2 * M_noise)

    t_vals = np.linspace(0, 20, 600)
    C_output = []
    for t in t_vals:
        Ct = expm(A_mat * t) @ sigma
        C_output.append([Ct[0,0]/sigma[0,0], Ct[0,1]/sigma[0,1], 
                         Ct[1,0]/sigma[1,0], Ct[1,1]/sigma[1,1]])
    
    return t_vals, np.array(C_output)

scenarios = [
    {'id': 'C', 'dEI': -0.5, 'dIE': 4.0},
    {'id': 'D', 'dEI': -2.0, 'dIE': 1.0},
    {'id': 'E', 'dEI': -4.0, 'dIE': 2.5},
    {'id': 'F', 'dEI': -2.0, 'dIE': -2.0}
]

fig, axes = plt.subplots(4, 2, figsize=(14, 20), dpi=300)

for row, scene in enumerate(scenarios):
    for col, (ce, ci) in enumerate([(0.5, 0.5), (0.7, 0.3)]):
        t, C = get_correlations(scene['dEI'], scene['dIE'], ce, ci, scene['id'])
        ax = axes[row, col]
        
        ax.plot(t, C[:,0], 'k-',  label=r'$C_{\Sigma\Sigma}$')
        ax.plot(t, C[:,1], 'r--', label=r'$C_{\Sigma\Delta}$')
        ax.plot(t, C[:,2], 'g:',  label=r'$C_{\Delta\Sigma}$')
        ax.plot(t, C[:,3], 'b-.', label=r'$C_{\Delta\Delta}$')
        
        y_min, y_max = np.min(C), np.max(C)
        padding = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - padding, y_max + padding)
        
        ax.set_title(f"Point {scene['id']} ($\chi_E={ce}$)\n$\delta EI={scene['dEI']}, \delta IE={scene['dIE']}$", fontsize=10)
        ax.set_xlabel('t (ms)', fontsize=9)
        ax.set_ylabel('C(t)/C(0)', fontsize=9)
        ax.legend(loc='upper right', fontsize='x-small', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', lw=0.7)

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout()
plt.show()