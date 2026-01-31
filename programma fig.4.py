import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import expm, solve_lyapunov

# Costanti del modello (Source: Sezione II e Appendice)
ALPHA = 0.1      # ms^-1 (Tasso di disattivazione)
BETA = 1.0       # ms^-1 (Parametro della funzione di attivazione)
WEE_base = 6.95
WII_base = 6.85
H_field = 1e-6   # Campo esterno piccolo
CHI_E = 0.5      # Frazione eccitatoria per Fig. 4
CHI_I = 0.5      # Frazione inibitoria per Fig. 4

# Definizione dei punti per FIG. 4 (delta_IE, delta_EI)
points_cfg = {
    'A': {'dIE': 0.6, 'dEI': -0.5, 'ylim': [-2, 2]},
    'B': {'dIE': 2.1, 'dEI': 0.0, 'ylim': [-4, 4]},
    'C': {'dIE': 4.0, 'dEI': -0.5, 'ylim': [-3, 3]},
    'D': {'dIE': 1.0, 'dEI': -2.0, 'ylim': [-1.5, 2.5]},
    'F': {'dIE': -2.0, 'dEI': -2.0, 'ylim': [-0.2, 1.2]},
    'G': {'dIE': -4.0, 'dEI': 2.0, 'ylim': [-0.1, 1.1]}
}

def f_act(s): return BETA * np.tanh(s) if s > 0 else 0.0
def df_act(s): return BETA * (1 - np.tanh(s)**2) if s > 0 else 0.0

def equations(vars, w_ee, w_ei, w_ie, w_ii):
    """Equazioni deterministiche per il punto fisso (Eq. 7)."""
    e, i = vars
    se = w_ee * e - w_ei * i + H_field
    si = w_ie * e - w_ii * i + H_field
    return [-ALPHA * e + (1 - e) * f_act(se), -ALPHA * i + (1 - i) * f_act(si)]

def get_point_dynamics(p_name):
    """Calcola la matrice di stabilità A e la covarianza sigma (Appendice)."""
    cfg = points_cfg[p_name]
    w_ei = WII_base + cfg['dEI']
    w_ie = WEE_base + cfg['dIE']
    
    # Scelta del guess iniziale per trovare il punto fisso corretto (Source: Fig 1 e Fig 5)
    # B e C sono in regime di bassa attività (~10^-7), gli altri in alta attività.
    guess = [1e-8, 1e-8] if p_name in ['B', 'C'] else [0.8, 0.8]
    e0, i0 = fsolve(equations, guess, args=(6.95, w_ei, w_ie, 6.85))
    
    se, si = 6.95*e0 - w_ei*i0 + H_field, w_ie*e0 - 6.85*i0 + H_field
    
    # Elementi della matrice A_tilde (Eq. A16-A19)
    AEE = -ALPHA - f_act(se) + (1-e0)*6.95*df_act(se)
    AEI = -(1-e0)*w_ei*df_act(se)
    AIE = (1-i0)*w_ie*df_act(si)
    AII = -ALPHA - f_act(si) - (1-i0)*6.85*df_act(si)
    
    # Trasformazione alle variabili (Sigma, Delta) (Eq. A24)
    # Per chi_E = chi_I = 0.5, i termini di riscalatura sono unitari.
    x = 0.5*(AEE + AEI + AIE + AII)
    y = 0.5*(AEE - AEI + AIE - AII)
    z = 0.5*(AEE + AEI - AIE - AII)
    w = 0.5*(AEE - AEI - AIE + AII)
    amat = np.array([[x, y], [z, w]])
    
    # Matrice di diffusione M (Eq. A26, A30)
    g_val, h_val = ALPHA*(e0 + i0), ALPHA*(e0 - i0)
    m_mat = np.array([[g_val, h_val], [h_val, g_val]])
    
    # Risoluzione analitica della covarianza sigma: A*sig + sig*A.T = -2M
    sigma = solve_lyapunov(amat, -2 * m_mat)
    return amat, sigma

def plot_reproduction():
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), dpi=200)
    t_vals = np.linspace(0, 5, 500) # Tempo da 0 a 5 ms come in FIG 4
    names = ['A', 'B', 'C', 'D', 'F', 'G']
    
    # Stili e legende come nel paper
    styles = [
        (0, 'black', 'o', r'$C_{\Sigma\Sigma}$'),
        (1, 'red',   's', r'$C_{\Sigma\Delta}$'),
        (2, 'green', '^', r'$C_{\Delta\Sigma}$'),
        (3, 'blue',  '*', r'$C_{\Delta\Delta}$')
    ]
    
    for i, p_name in enumerate(names):
        ax = axes[i // 2, i % 2]
        amat, sig = get_point_dynamics(p_name)
        
        # Calcolo traiettorie C(t) = exp(At) * sigma
        trajs = []
        for t in t_vals:
            ct = expm(amat * t) @ sig
            # Normalizzazione C(t)/C(0)
            trajs.append([ct[0,0]/sig[0,0], ct[0,1]/sig[0,1], 
                         ct[1,0]/sig[1,0], ct[1,1]/sig[1,1]])
        trajs = np.array(trajs)
        
        for k, color, marker, lbl in styles:
            # Soluzione analitica (linea tratteggiata)
            ax.plot(t_vals, trajs[:, k], color=color, linestyle='--', linewidth=1.2)
            # Proxy dati simulazione (markers a intervalli)
            m_idx = np.linspace(0, len(t_vals)-1, 20, dtype=int)
            ax.plot(t_vals[m_idx], trajs[m_idx, k], marker, color=color, 
                    markersize=5, label=lbl, linestyle='None')
        
        ax.set_title(f"({p_name}) $\delta IE={points_cfg[p_name]['dIE']}, \delta EI={points_cfg[p_name]['dEI']}$", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('C(t)/C(0)')
        ax.set_ylim(points_cfg[p_name]['ylim'])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('FIG4_WilsonCowan_Corrected.png')
    plt.show()

if __name__ == "__main__":
    plot_reproduction()