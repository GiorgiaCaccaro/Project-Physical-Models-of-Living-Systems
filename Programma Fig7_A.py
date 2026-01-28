import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import expm

# Parametri

wEE, wII = 6.95, 6.85
h = 1e-6
alpha = 0.1
beta = 1.0

deltaEI, deltaIE = -0.5, 0.6
wEI = wII + deltaEI
wIE = wEE + deltaIE

# Parametri simulazione
epsilon = 5e-1  
dt = 0.01       
t_max = 5.0    
t_eval = np.arange(0, t_max + dt, dt)


def f(S):
    return beta * np.tanh(S) if S > 0 else 0.0

def f_prime(S):
    return beta * (1.0 / np.cosh(S))**2 if S > 0 else 0.0

def field(state):

    E, I = state
    SE = wEE * E - wEI * I + h
    SI = wIE * E - wII * I + h
    dE = -alpha * E + (1 - E) * f(SE)
    dI = -alpha * I + (1 - I) * f(SI)
    return np.array([dE, dI])

def rk4_step(state, dt):
    k1 = field(state)
    k2 = field(state + 0.5 * dt * k1)
    k3 = field(state + 0.5 * dt * k2)
    k4 = field(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def eqs_fp(p):
    E, I = p
    return [-alpha * E + (1 - E) * f(wEE*E - wEI*I + h),
            -alpha * I + (1 - I) * f(wIE*E - wII*I + h)]

E0, I0 = fsolve(eqs_fp, [0.5, 0.5])
SE0, SI0 = wEE*E0 - wEI*I0 + h, wIE*E0 - wII*I0 + h

# Jacobiano
AEE = -alpha - f(SE0) + (1 - E0) * wEE * f_prime(SE0)
AEI = -(1 - E0) * wEI * f_prime(SE0)
AIE = (1 - I0) * wIE * f_prime(SI0)
AII = -alpha - f(SI0) - (1 - I0) * wII * f_prime(SI0)

# Matrice in base (Sigma, Delta)
A_SD = np.array([
    [0.5*(AEE + AEI + AIE + AII), 0.5*(AEE - AEI + AIE - AII)],
    [0.5*(AEE + AEI - AIE - AII), 0.5*(AEE - AEI - AIE + AII)]
])


# Simulazione 

def simulate_response(pert_type='Sigma'):
    if pert_type == 'Sigma':
        state = np.array([E0 + epsilon, I0 + epsilon])
    else:
        state = np.array([E0 + epsilon, I0 - epsilon])
    
    res_history = []
    for t in t_eval:
        # Trasformiamo la risposta in coordinate Sigma/Delta
        delta_E = state[0] - E0
        delta_I = state[1] - I0

        sig = 0.5 * (delta_E + delta_I) / epsilon
        det = 0.5 * (delta_E - delta_I) / epsilon
        
        res_history.append([sig, det])
        state = rk4_step(state, dt)
    return np.array(res_history)


num_Sigma = simulate_response('Sigma')
num_Delta = simulate_response('Delta')


t_anal = np.linspace(0, t_max, 200)
R_anal = np.array([expm(A_SD * t) for t in t_anal])

fig, ax1 = plt.subplots(figsize=(10, 7))


ax1.plot(t_anal, R_anal[:,0,0], 'k-', label=r'$R_{\Sigma\Sigma}$ ')
ax1.plot(t_eval[::20], num_Sigma[::20, 0], 'ko', mfc='none')


ax1.plot(t_anal, R_anal[:,1,0], 'r-', label=r'$R_{\Delta\Sigma}$')
ax1.plot(t_eval[::20], num_Sigma[::20, 1], 'rs', mfc='none')


ax1.plot(t_anal, R_anal[:,1,1], 'g-', label=r'$R_{\Delta\Delta}$ ')
ax1.plot(t_eval[::20], num_Delta[::20, 1], 'g*', mfc='none')

ax1.set_xlabel('t (ms)')
ax1.set_ylabel('Response R(t)', fontsize=14)
ax1.set_ylim(-1.5, 1.5)



ax2 = ax1.twinx()
ax2.plot(t_anal, R_anal[:,0,1], 'b-', label=r'$R_{\Sigma\Delta}$ ')
ax2.plot(t_eval[::20], num_Delta[::20, 0], 'b^', mfc='none')

ax2.set_ylim(-30, 30)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='small')

plt.title(f"Panel (A) ", fontweight='bold', fontsize=14)
ax1.grid(True, linestyle=':', alpha=0.6)
plt.show()