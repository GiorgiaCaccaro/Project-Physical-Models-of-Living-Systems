import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.linalg import expm

# parameters

alpha = 0.1
beta = 1.0
h = 1e-6

wEE = 6.95
wII = 6.85

chiE = 0.5
chiI = 0.5

epsilon = 1e-3
dt = 0.005
t_max = 5.0
t_vals = np.arange(0, t_max + dt, dt)


# Smooth activation

def f(S):
    return beta * np.tanh(S)

def df(S):
    return beta * (1 - np.tanh(S)**2)

# Deterministic field

def deterministic_field(state):
    E, I = state
    SE = wEE * E - (wII + dEI_curr) * I + h
    SI = (wEE + dIE_curr) * E - wII * I + h
    dE = -alpha * E + (1 - E) * f(SE)
    dI = -alpha * I + (1 - I) * f(SI)
    return np.array([dE, dI])

def rk4_step(state):
    k1 = deterministic_field(state)
    k2 = deterministic_field(state + 0.5 * dt * k1)
    k3 = deterministic_field(state + 0.5 * dt * k2)
    k4 = deterministic_field(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# Fixed point + Jacobian

def get_system_matrices(dEI, dIE, state='high'):
    wEI = wII + dEI
    wIE = wEE + dIE

    guess = [0.8, 0.4] if state == 'high' else [1e-6, 1e-6]

    def eqs(p):
        E, I = p
        SE = wEE * E - wEI * I + h
        SI = wIE * E - wII * I + h
        return [
            -alpha * E + (1 - E) * f(SE),
            -alpha * I + (1 - I) * f(SI)
        ]

    sol = root(eqs, guess, tol=1e-12)
    E0, I0 = sol.x

    SE = wEE * E0 - wEI * I0 + h
    SI = wIE * E0 - wII * I0 + h

    # Jacobian in (E,I)
    AEE = -alpha - f(SE) + (1 - E0) * wEE * df(SE)
    AEI = -(1 - E0) * wEI * df(SE)
    AIE = (1 - I0) * wIE * df(SI)
    AII = -alpha - f(SI) - (1 - I0) * wII * df(SI)

    
    term_EI = (chiE / chiI) * AEI
    term_IE = (chiI / chiE) * AIE


    x = 0.5 * (AEE + term_EI + term_IE + AII)  # Sigma Sigma
    y = 0.5 * (AEE - term_EI + term_IE - AII)  # Sigma Delta
    z = 0.5 * (AEE + term_EI - term_IE - AII)  # Delta Sigma
    w = 0.5 * (AEE - term_EI - term_IE + AII)  # Delta Delta

    A_SD = np.array([[x, y], [z, w]])
    return E0, I0, A_SD


# Simulation

def run_simulation(E0, I0, pert_type='Sigma'):
    
    dE_val = epsilon / (2 * chiE)
    dI_val = epsilon / (2 * chiI)

    if pert_type == 'Sigma':
        # Sigma = chiE*E + chiI*I. Per avere dSigma=eps, dDelta=0:
        state = np.array([E0 + dE_val, I0 + dI_val])
    else:
        # Delta = chiE*E - chiI*I. Per avere dDelta=eps, dSigma=0:
        state = np.array([E0 + dE_val, I0 - dI_val])

    history = []
    for _ in t_vals:
        history.append(state.copy())
        state = rk4_step(state)

    history = np.array(history)

    R_Sigma = (chiE*(history[:,0]-E0) + chiI*(history[:,1]-I0)) / epsilon
    R_Delta = (chiE*(history[:,0]-E0) - chiI*(history[:,1]-I0)) / epsilon

    return R_Sigma, R_Delta


points = {
    'A': ((-0.5, 0.6), 'low'),
    'B': ((0.0, 2.1), 'low'),
    'C': ((-0.5, 4.0), 'low'),
    'D': ((-2.0, 1.0), 'high'),
    'F': ((-2.0, -2.0), 'high'),
    'G': ((2.0, -4.0), 'high')
}


plt.rcParams['figure.dpi'] = 150 
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.flatten()

for i, (label, ((dEI, dIE), state_type)) in enumerate(points.items()):
    ax = axes[i]
    global dEI_curr, dIE_curr
    dEI_curr, dIE_curr = dEI, dIE

    E0, I0, A_SD = get_system_matrices(dEI, dIE, state_type)

    t_anal = np.linspace(0, t_max, 500)
    # Calcolo analitico
    R_anal = np.array([expm(A_SD * t) for t in t_anal])

    # Simulazione numerica
    R_SS, R_DS = run_simulation(E0, I0, 'Sigma')
    R_SD, R_DD = run_simulation(E0, I0, 'Delta')

    step = len(t_vals) // 25
    
    # Plotting
    ax.plot(t_anal, R_anal[:,0,0], 'k-', label=r'$R_{\Sigma\Sigma}$')
    ax.plot(t_vals[::step], R_SS[::step], 'ko', mfc='none')

    ax.plot(t_anal, R_anal[:,0,1], 'b-', label=r'$R_{\Sigma\Delta}$')
    ax.plot(t_vals[::step], R_SD[::step], 'b^', mfc='none')

    ax.plot(t_anal, R_anal[:,1,0], 'r-', label=r'$R_{\Delta\Sigma}$')
    ax.plot(t_vals[::step], R_DS[::step], 'rs', mfc='none')

    ax.plot(t_anal, R_anal[:,1,1], 'g-', label=r'$R_{\Delta\Delta}$')
    ax.plot(t_vals[::step], R_DD[::step], 'g*', mfc='none')

    ax.set_title(f"Panel ({label})", fontweight='bold')
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("Response R(t)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize='small')

plt.tight_layout()
plt.show()
