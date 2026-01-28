import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.linalg import expm, solve_lyapunov
import matplotlib.pyplot as plt

# Costanti
ALPHA = 0.1  # ms^-1
BETA = 1.0   # ms^-1
H = 1e-6     # External field
W_EE = 6.95  # Synaptic strength W_EE
W_II = 6.85  # Synaptic strength W_II

def activation_function(S):
    return np.where(S > 0, BETA * np.tanh(S), 0.0)

def derivative_activation_function(S):

    return np.where(S > 0, BETA * (1 - np.tanh(S)**2), 0.0)

def fixed_point_equations(P, alpha, w_ee, w_ii, w_ei, w_ie, h):

    E, I = P
    E = np.clip(E, 1e-6, 1.0 - 1e-6)
    I = np.clip(I, 1e-6, 1.0 - 1e-6)

    S_E = w_ee * E - w_ei * I + h
    S_I = w_ie * E - w_ii * I + h
    
    f_E = activation_function(S_E)
    f_I = activation_function(S_I)
    
    eq1 = -alpha * E + (1 - E) * f_E
    eq2 = -alpha * I + (1 - I) * f_I
    
    return [eq1, eq2]

def find_fixed_point(w_ei, w_ie):
    initial_guesses = [
        [0.8, 0.8],   # Alta attività (Equilibrio)
        [0.9, 0.7],   # Alta attività (Sbilanciato E)
        [0.7, 0.9]    # Alta attività (Sbilanciato I)
    ]
    
    for E0_guess, I0_guess in initial_guesses:
        try:
            result, _, status, _ = fsolve(fixed_point_equations, [E0_guess, I0_guess], 
                                          args=(ALPHA, W_EE, W_II, w_ei, w_ie, H), 
                                          full_output=True, xtol=1e-12)
            E0, I0 = result
            
            if status == 1 and E0 > 0.05 and I0 > 0.05:
                return np.clip(E0, 1e-6, 1.0 - 1e-6), np.clip(I0, 1e-6, 1.0 - 1e-6)
            
        except Exception:
            continue
            
    E0_low, I0_low = fsolve(fixed_point_equations, [0.001, 0.001], 
                            args=(ALPHA, W_EE, W_II, w_ei, w_ie, H))
    return np.clip(E0_low, 1e-6, 1.0 - 1e-6), np.clip(I0_low, 1e-6, 1.0 - 1e-6)

def calculate_A_sigma_C(chi_E, chi_I, d_ei, d_ie, time_points):
    
    w_ei = W_II + d_ei
    w_ie = W_EE + d_ie

    E0, I0 = find_fixed_point(w_ei, w_ie)
    
    if E0 < 0.05 and I0 < 0.05:
        print(f"Avviso: Punto fisso di bassa attività trovato (E0={E0:.2e}, I0={I0:.2e}).")
        
    S_E = W_EE * E0 - w_ei * I0 + H
    S_I = w_ie * E0 - W_II * I0 + H
    
    f_E = activation_function(S_E)
    f_I = activation_function(S_I)
    f_prime_E = derivative_activation_function(S_E)
    f_prime_I = derivative_activation_function(S_I)
    
    # Matrice A 
    chi_ratio_E_I = chi_E / chi_I
    chi_ratio_I_E = chi_I / chi_E
    
    A_EE = -ALPHA - f_E + (1 - E0) * W_EE * f_prime_E 
    A_EI = -(1 - E0) * w_ei * f_prime_E
    A_IE = (1 - I0) * w_ie * f_prime_I
    A_II = -ALPHA - f_I - (1 - I0) * W_II * f_prime_I

    x = 0.5 * (A_EE + chi_ratio_E_I**(3/2) * A_EI + chi_ratio_I_E**(3/2) * A_IE + A_II)
    y = 0.5 * (A_EE - chi_ratio_E_I**(3/2) * A_EI + chi_ratio_I_E**(3/2) * A_IE - A_II)
    z = 0.5 * (A_EE + chi_ratio_E_I**(3/2) * A_EI - chi_ratio_I_E**(3/2) * A_IE - A_II)
    w = 0.5 * (A_EE - chi_ratio_E_I**(3/2) * A_EI - chi_ratio_I_E**(3/2) * A_IE + A_II)
    
    A = np.array([[x, y], [z, w]])

    # Matrice M 
    G = ALPHA * E0 + ALPHA * I0
    H_noise = ALPHA * E0 - ALPHA * I0
    M = np.array([[G, H_noise], [H_noise, G]])

    # Matrice sigma
    try:
        sigma = solve_lyapunov(A, -2 * M)
    except Exception as e:
        print(f"Errore Lyapunov per d_EI={d_ei}, d_IE={d_ie}: {e}")
        return None
    sigma = (sigma + sigma.T) / 2
    
    # Matrice di Correlazione
    C_results = []
    for t in time_points:
        eAt = expm(A * t)
        C_t = np.dot(eAt, sigma)
        C_results.append(C_t.flatten())
        
    C_results = np.array(C_results)
    
    # 6. Normalizzazione per C(0)
    C0_SigmaSigma, C0_SigmaDelta = sigma[0, 0], sigma[0, 1]
    C0_DeltaSigma, C0_DeltaDelta = sigma[1, 0], sigma[1, 1]
    
    C_SigmaSigma_norm = C_results[:, 0] / C0_SigmaSigma
    C_SigmaDelta_norm = C_results[:, 1] / C0_SigmaDelta
    C_DeltaSigma_norm = C_results[:, 2] / C0_DeltaSigma
    C_DeltaDelta_norm = C_results[:, 3] / C0_DeltaDelta


    results_df = pd.DataFrame({
        't (ms)': time_points,
        'C_SigmaSigma': C_SigmaSigma_norm,
        'C_SigmaDelta': C_SigmaDelta_norm,
        'C_DeltaSigma': C_DeltaSigma_norm,
        'C_DeltaDelta': C_DeltaDelta_norm,
    })
    results_df['chi_E'] = chi_E
    results_df['d_EI'] = d_ei
    results_df['d_IE'] = d_ie
    
    return results_df




def print_calculation_summary(scenario_name, E0, I0, Sigma_0, Delta_0, eigenvalues, df):
    """Stampa un riassunto dei punti fissi e dei primi valori di correlazione."""
    
    print("-" * 50)
    print(f"RIEPILOGO CALCOLO: {scenario_name}")
    print("-" * 50)
    print(f"Punto Fissio (E0, I0): E0 = {E0:.6f}, I0 = {I0:.6f}")
    print(f"Attività (Sigma_0, Delta_0): Sigma_0 = {Sigma_0:.6f}, Delta_0 = {Delta_0:.6f}")

    print("\nAutovalori della Matrice A:")
    for i, lam in enumerate(eigenvalues):
        if np.iscomplex(lam):
            print(f"  Lambda_{i+1}: {lam.real:.4f} + {lam.imag:.4f}i (Oscillatorio)")
        else:
            tau = 1.0 / abs(lam) if abs(lam) > 1e-6 else np.inf
            print(f"  Lambda_{i+1}: {lam.real:.6f} (Decadimento, Tau = {tau:.4f} ms)")


def plot_fig3_correlation(df):
   
    if df.empty:
        print("Impossibile tracciare il grafico: il DataFrame è vuoto.")
        return

    points = ['A', 'B', 'G']
    chi_settings = [0.5, 0.7] # 50% (colonna sinistra), 70% (colonna destra)
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), constrained_layout=True)
    
    corr_types = ['C_SigmaSigma', 'C_SigmaDelta', 'C_DeltaSigma', 'C_DeltaDelta']
    labels = [r'$C_{\Sigma\Sigma}$', r'$C_{\Sigma\Delta}$', r'$C_{\Delta\Sigma}$', r'$C_{\Delta\Delta}$']
    colors = ['k', 'r', 'g', 'b'] # Nero, Rosso, Verde, Blu
    row_map = {'A': 0, 'B': 1, 'G': 2}

    for i, point in enumerate(points):
        for j, chi_E in enumerate(chi_settings):
            ax = axes[row_map[point], j]
            point_name = f'{point}_{int(chi_E*100)}'
            subset_df = df[df['Point'] == point_name]
            
            if subset_df.empty:
                ax.text(0.5, 0.5, f"Nessun dato per {point_name}", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, color='red')
                ax.set_title(f'({point}) $\\chi_E={int(chi_E*100)}\\%$')
                continue

            d_EI = subset_df['d_EI'].iloc[0]
            d_IE = subset_df['d_IE'].iloc[0]
            chi_I = 1.0 - chi_E


            for k, corr in enumerate(corr_types):
                ax.plot(subset_df['t (ms)'], subset_df[corr], label=labels[k], 
                        color=colors[k], linewidth=2)
            
            title_text = f'({point}) $\\chi_E={int(chi_E*100)}\\%$ $\\chi_I={int(chi_I*100)}\\%$'
            param_text = f'$\\delta EI={d_EI}$ $\\delta IE={d_IE}$'
            

            ax.text(0.95, 0.95, title_text, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right')
            ax.text(0.95, 0.85, param_text, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right')
            
    
            # if j == 0:
            ax.set_ylabel(r'$C(t)/C(0)$')
            # if i == len(points) - 1:
            ax.set_xlabel('$t$ (ms)')
            
            ax.legend(loc='lower right', fontsize=8, ncol=2)
            
            ax.set_xlim(0, 20)
            
            if point == 'G' and i == 2:
                ax.text(10, 0.7, r'$\tau_1=0.92$ $\tau_2=0.23$', fontsize=10)
                
    
    plt.savefig('figure_3_correlation_functions.png')
    plt.close(fig)



time_points = np.linspace(0, 20, 2001)


scenarios_to_run = [
    {"point": "A_50", "chi_E": 0.5, "chi_I": 0.5, "d_EI": -0.5, "d_IE": +0.6},
    {"point": "A_70", "chi_E": 0.7, "chi_I": 0.3, "d_EI": -0.5, "d_IE": +0.6},
    {"point": "B_50", "chi_E": 0.5, "chi_I": 0.5, "d_EI": 0.0, "d_IE": +2.1},
    {"point": "B_70", "chi_E": 0.7, "chi_I": 0.3, "d_EI": 0.0, "d_IE": +2.1},
    # {"point": "G_50", "chi_E": 0.5, "chi_I": 0.5, "d_EI": +2.0, "d_IE": -4.0},
    # {"point": "G_70", "chi_E": 0.7, "chi_I": 0.3, "d_EI": +2.0, "d_IE": -4.0},
]

final_df = []
for s in scenarios_to_run:
    result = calculate_A_sigma_C(s['chi_E'], s['chi_I'], s['d_EI'], s['d_IE'], time_points)
    if result is not None:
        result['Point'] = s['point']
        final_df.append(result)

if final_df:
    final_df = pd.concat(final_df, ignore_index=True)

    plot_fig3_correlation(final_df)