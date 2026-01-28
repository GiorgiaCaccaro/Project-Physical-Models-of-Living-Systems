import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parameters
alpha = 0.1  # ms^-1
beta = 1.0   # ms^-1
h = 1e-6     # External field
dt = 0.001   # Time step (ms)
t_max_sim = 200.0  # Increased simulation time for better statistics (ms)
t_plot = 5.0       # Time window for the plot (ms)
chi_E, chi_I = 0.5, 0.5

# Synaptic weights
w_EE, w_II = 6.95, 6.85
delta_EI, delta_IE = -0.5, +4.0
w_EI = w_II + delta_EI
w_IE = w_EE + delta_IE

def f(S): 
    return beta * np.tanh(S) if S > 0 else 0.0

# FIXED POINT 
def steady_state_eqs(p):
    E, I = p
    S_E = w_EE * E - w_EI * I + h
    S_I = w_IE * E - w_II * I + h
    return [-alpha * E + (1 - E) * f(S_E), -alpha * I + (1 - I) * f(S_I)]


E0, I0 = fsolve(steady_state_eqs, [1e-7, 1e-7])

# LANGEVIN SIMULATION
def run_enhanced_simulation(N):
    steps = int(t_max_sim / dt)
    N_E, N_I = int(N * chi_E), int(N * chi_I)
    
    # Initialize at the theoretical fixed point
    E, I = E0, I0
    
    # Burn-in period to reach stationarity (50 ms)
    for _ in range(int(50.0/dt)):
        SE, SI = w_EE*E - w_EI*I + h, w_IE*E - w_II*I + h
        fE, fI = f(SE), f(SI)
        dE = (-alpha*E + (1-E)*fE)*dt + np.sqrt((alpha*E + (1-E)*fE)/N_E)*np.random.normal(0, np.sqrt(dt))
        dI = (-alpha*I + (1-I)*fI)*dt + np.sqrt((alpha*I + (1-I)*fI)/N_I)*np.random.normal(0, np.sqrt(dt))
        E, I = max(0, E + dE), max(0, I + dI)

    # Actual sampling
    sigma_history = np.zeros(steps)
    for t in range(steps):
        SE, SI = w_EE*E - w_EI*I + h, w_IE*E - w_II*I + h
        fE, fI = f(SE), f(SI)
        dE = (-alpha*E + (1-E)*fE)*dt + np.sqrt((alpha*E + (1-E)*fE)/N_E)*np.random.normal(0, np.sqrt(dt))
        dI = (-alpha*I + (1-I)*fI)*dt + np.sqrt((alpha*I + (1-I)*fI)/N_I)*np.random.normal(0, np.sqrt(dt))
        E, I = max(0, E + dE), max(0, I + dI)
        sigma_history[t] = chi_E * E + chi_I * I
    
    # Calculate fluctuations relative to the trace mean
    xi = sigma_history - np.mean(sigma_history)
    
    # FFT-based Autocorrelation for efficiency and noise reduction
    n = len(xi)
    f_xi = np.fft.fft(xi, n=2*n)
    acf = np.fft.ifft(f_xi * np.conjugate(f_xi))[:n].real
    
    # Normalize by variance C(0)
    return acf / acf[0]


N_list = [1e10, 1e11, 1e12, 1e14]
colors = ['orange', 'blue', 'green', 'red']
markers = ['v', '^', 's', 'o']
time_axis = np.linspace(0, t_plot, int(t_plot/dt))

plt.figure(figsize=(10, 7))

for i, N in enumerate(N_list):
    print(f"Computing simulation for N = 10^{int(np.log10(N))}...")
    acf_result = run_enhanced_simulation(N)
    
    # Sub-sample markers for cleaner high-resolution output
    plt.plot(time_axis[::200], acf_result[:len(time_axis)][::200], 
             label=f"$N=10^{{{int(np.log10(N))}}}$", 
             color=colors[i], marker=markers[i], linestyle='None', markersize=6)
    
    plt.plot(time_axis, acf_result[:len(time_axis)], color=colors[i], alpha=0.3)

plt.axhline(0, color='black', lw=1, ls='--')
plt.xlabel("$t$ (ms)", fontsize=14)
plt.ylabel("$C_{\Sigma\Sigma}(t)$", fontsize=14)
plt.title("Correlation function point C", fontsize=16)
plt.legend(title="System Size $N$")
plt.xlim(0, 5)
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.2)
plt.savefig('fig5_improved_N_convergence.png', dpi=300)
plt.show()