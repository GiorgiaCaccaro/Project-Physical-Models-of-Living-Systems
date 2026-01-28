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
    E_guesses = np.linspace(0.5, 0.95, 10)
    I_guesses = np.linspace(0.5, 0.95, 10)
    initial_guesses = [(e_g, i_g) for e_g in E_guesses for i_g in I_guesses]
            
    for E0_guess, I0_guess in initial_guesses:
        try:
            result, _, status, _ = fsolve(fixed_point_equations, [E0_guess, I0_guess], 
                                          args=(ALPHA, W_EE, W_II, w_ei, w_ie, H), 
                                          full_output=True, xtol=1e-10, maxfev=1000)
            E0, I0 = result
            
            if status == 1 and E0 > 0.05 and I0 > 0.05:
                return np.clip(E0, 1e-6, 1.0 - 1e-6), np.clip(I0, 1e-6, 1.0 - 1e-6)
            
        except Exception:
            continue
            
    return None, None 

def calculate_A_sigma_C(chi_E, chi_I, d_ei, d_ie, time_points):
    
    w_ei = W_II + d_ei
    w_ie = W_EE + d_ie

    # punti fissi 
    E0, I0 = find_fixed_point(w_ei, w_ie)
    
    if E0 is None:
        print(f"ERRORE DI CALCOLO: Impossibile trovare il punto fisso di alta attività per w_EI={w_ei:.2f}, w_IE={w_ie:.2f}.")
        return None, None, None, None, None, None, None, None # 8 valori None
        
    # Calcolo di Sigma_0 e Delta_0
    Sigma_0 = chi_E * E0 + chi_I * I0
    Delta_0 = chi_E * E0 - chi_I * I0
        
    # parametri al punto fisso
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

    # Matrice di Covarianza
    try:
        M = np.array([[ALPHA * E0 + ALPHA * I0, ALPHA * E0 - ALPHA * I0], 
                      [ALPHA * E0 - ALPHA * I0, ALPHA * E0 + ALPHA * I0]])
        sigma = solve_lyapunov(A, -2 * M)
    except Exception as e:
        print(f"Errore Lyapunov per d_EI={d_ei}, d_IE={d_ie}: {e}")
        return None, None, None, None, None, None, None, None # 8 valori None
    sigma = (sigma + sigma.T) / 2
    
    # Tau
    eigenvalues = np.linalg.eigvals(A)
    

    if all(np.isreal(eigenvalues)):
        lambdas_real = np.sort(np.real(eigenvalues))
        taus = np.sort([1.0 / abs(lam) if abs(lam) > 1e-6 else np.inf for lam in lambdas_real])
        tau1 = taus[1] 
        tau2 = taus[0] 
    else:
        tau1, tau2 = np.nan, np.nan 
        
    # Calcolo della Matrice di Correlazione
    C_results = np.array([np.dot(expm(A * t), sigma).flatten() for t in time_points])
    
    # Normalizzazione per C(0)
    C0_SigmaSigma, C0_SigmaDelta = sigma[0, 0], sigma[0, 1]
    C0_DeltaSigma, C0_DeltaDelta = sigma[1, 0], sigma[1, 1]
    min_c0 = 1e-12 
    
    C_SigmaSigma_norm = C_results[:, 0] / max(C0_SigmaSigma, min_c0)
    C_SigmaDelta_norm = C_results[:, 1] / max(C0_SigmaDelta, min_c0)
    C_DeltaSigma_norm = C_results[:, 2] / max(C0_DeltaSigma, min_c0)
    C_DeltaDelta_norm = C_results[:, 3] / max(C0_DeltaDelta, min_c0)
    

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
    
    return results_df, E0, I0, Sigma_0, Delta_0, eigenvalues, tau1, tau2


def print_calculation_summary(scenario_name, E0, I0, Sigma_0, Delta_0, eigenvalues, df):
    
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
    
    print("\nPrimi 5 valori di C(t)/C(0):")
    print(df.head(5).to_string(index=False, columns=['t (ms)', 'C_SigmaSigma', 'C_SigmaDelta', 'C_DeltaSigma', 'C_DeltaDelta']))
    print("-" * 50)

def plot_fig3_correlation(df, tau_data):
    
    if df.empty:
        print("Impossibile tracciare il grafico: il DataFrame è vuoto.")
        return

    points = ['A', 'B', 'G']
    chi_settings = [0.5, 0.7]
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), constrained_layout=True)
    
    corr_types = ['C_SigmaSigma', 'C_SigmaDelta', 'C_DeltaSigma', 'C_DeltaDelta']
    labels = [r'$C_{\Sigma\Sigma}$', r'$C_{\Sigma\Delta}$', r'$C_{\Delta\Sigma}$', r'$C_{\Delta\Delta}$']
    colors = ['k', 'r', 'g', 'b']
    row_map = {'A': 0, 'B': 1, 'G': 2}

    for i, point in enumerate(points):
        for j, chi_E in enumerate(chi_settings):
            ax = axes[row_map[point], j]
            point_name = f'{point}_{int(chi_E*100)}'
            subset_df = df[df['Point'] == point_name]
            
            if subset_df.empty:
                ax.text(0.5, 0.5, f"Nessun dato per {point_name}", transform=ax.transAxes, color='red')
                continue

            d_EI = subset_df['d_EI'].iloc[0]
            d_IE = subset_df['d_IE'].iloc[0]
            chi_I = 1.0 - chi_E

            for k, corr in enumerate(corr_types):
                ax.plot(subset_df['t (ms)'], subset_df[corr], label=labels[k], color=colors[k], linewidth=2)
            
            title_text = f'({point}) $\\chi_E={int(chi_E*100)}\\%$ $\\chi_I={int(chi_I*100)}\\%$'
            param_text = f'$\\delta EI={d_EI}$ $\\delta IE={d_IE}$'
            
            ax.text(0.95, 0.95, title_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
            ax.text(0.95, 0.85, param_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
            
            if j == 0: ax.set_ylabel(r'$C(t)/C(0)$')
            if i == len(points) - 1: ax.set_xlabel('$t$ (ms)')
            
            ax.legend(loc='lower right', fontsize=8, ncol=2)
            ax.set_xlim(0, 20)


            if point == 'G':
                tau1_calc, tau2_calc = tau_data[point_name]
                

                tau_text = r'$\tau_1={:.2f}$ $\tau_2={:.2f}$'.format(tau1_calc, tau2_calc)
                
                ax.set_ylim(0, 1.0)
                ax.text(10, 0.7, tau_text, fontsize=10)

            elif point == 'A':
                if j == 0: ax.set_ylim(-1.5, 1.5)
                else: ax.set_ylim(-0.5, 1.0)
            elif point == 'B':
                if j == 0: ax.set_ylim(-1.5, 4.0)
                else: ax.set_ylim(-0.5, 1.0)
                
    plt.suptitle("Funzioni di Correlazione Analitiche $C(t)/C(0)$ (FIG. 3)", y=1.02)
    plt.savefig('figure_3_correlation_functions_updated_tau.png')
    plt.close(fig)


time_points = np.linspace(0, 20, 2001)

scenarios_to_run = [
    # {"point": "A_50", "chi_E": 0.5, "chi_I": 0.5, "d_EI": -0.5, "d_IE": +0.6},
    # {"point": "A_70", "chi_E": 0.7, "chi_I": 0.3, "d_EI": -0.5, "d_IE": +0.6},
    # {"point": "B_50", "chi_E": 0.5, "chi_I": 0.5, "d_EI": 0.0, "d_IE": +2.1},
    # {"point": "B_70", "chi_E": 0.7, "chi_I": 0.3, "d_EI": 0.0, "d_IE": +2.1},
    {"point": "G_50", "chi_E": 0.5, "chi_I": 0.5, "d_EI": +2.0, "d_IE": -4.0},
    {"point": "G_70", "chi_E": 0.7, "chi_I": 0.3, "d_EI": +2.0, "d_IE": -4.0},
]

final_df_list = []
tau_data_for_plot = {} 

print("Calcolo e aggiornamento dei dati analitici e dei Tau...")
for s in scenarios_to_run:
    result = calculate_A_sigma_C(s['chi_E'], s['chi_I'], s['d_EI'], s['d_IE'], time_points)

    if result[0] is not None:
        result_df, E0, I0, Sigma_0, Delta_0, eigenvalues, tau1, tau2 = result
        point_name = s['point']
        result_df['Point'] = point_name
        final_df_list.append(result_df)

        if point_name.startswith('G'):
            tau_data_for_plot[point_name] = (tau1, tau2)

        print_calculation_summary(point_name, E0, I0, Sigma_0, Delta_0, eigenvalues, result_df)


if final_df_list:
    final_df = pd.concat(final_df_list, ignore_index=True)
    

    plot_fig3_correlation(final_df, tau_data_for_plot)
    print("\nGrafico con Tau calcolati dinamicamente per il Punto G è stato salvato come 'figure_3_correlation_functions_updated_tau.png'.")
else:
    print("\nLa generazione dei dati è fallita. Impossibile tracciare il grafico.")