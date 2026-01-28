import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.colors import ListedColormap


# Parametri 

alpha = 0.1
beta = 1.0
h = 1e-6

wEE_base = 6.95
wII_base = 6.85

chi_E = 0.5
chi_I = 0.5

# Funzioni di attivazione

def activation(S):
    return np.where(S > 0, beta * np.tanh(S), 0.0)

def activation_deriv(S):
    return np.where(S > 0, beta * (1.0 - np.tanh(S)**2), 0.0)

# Equazioni punti fissi

def fixed_point_equations(vars, wEE, wEI, wIE, wII):
    E, I = vars
    SE = wEE * E - wEI * I + h
    SI = wIE * E - wII * I + h
    fSE = activation(SE)
    fSI = activation(SI)
    return [-alpha * E + (1 - E) * fSE,
            -alpha * I + (1 - I) * fSI]

# Jacobiano

def jacobian_elements(E, I, wEE, wEI, wIE, wII):
    SE = wEE * E - wEI * I + h
    SI = wIE * E - wII * I + h
    fSE = activation(SE)
    fSI = activation(SI)
    dfSE = activation_deriv(SE)
    dfSI = activation_deriv(SI)
    A_EE = -alpha - fSE + (1 - E) * wEE * dfSE
    A_EI = -(1 - E) * wEI * dfSE
    A_IE = (1 - I) * wIE * dfSI
    A_II = -alpha - fSI - (1 - I) * wII * dfSI
    return A_EE, A_EI, A_IE, A_II


# Calcolo autovalori

def compute_eigenvalues(delta_EI, delta_IE):
    wEI = wII_base + delta_EI
    wIE = wEE_base + delta_IE
    if wEI < 0 or wIE < 0:
        return None, None
    guesses = [(0.1, 0.1), (0.8, 0.8), (0.0, 0.0)]
    E_sol, I_sol = None, None
    for guess in guesses:
        sol, info, ier, _ = fsolve(
            fixed_point_equations,
            guess,
            args=(wEE_base, wEI, wIE, wII_base),
            full_output=True,
            maxfev=3000
        )
        if ier == 1 and 0 <= sol[0] <= 1 and 0 <= sol[1] <= 1:
            E_sol, I_sol = sol
            break
    if E_sol is None:
        return None, None

    a_ee, a_ei, a_ie, a_ii = jacobian_elements(E_sol, I_sol, wEE_base, wEI, wIE, wII_base)
    r_EI = (chi_E / chi_I)**1.5
    r_IE = (chi_I / chi_E)**1.5
    x = 0.5 * (a_ee + r_EI * a_ei + r_IE * a_ie + a_ii)
    y = 0.5 * (a_ee - r_EI * a_ei + r_IE * a_ie - a_ii)
    z = 0.5 * (a_ee + r_EI * a_ei - r_IE * a_ie - a_ii)
    w = 0.5 * (a_ee - r_EI * a_ei - r_IE * a_ie + a_ii)
    eigvals = np.linalg.eigvals([[x, y], [z, w]])
    eigvals = eigvals[np.argsort(np.real(eigvals))]
    return eigvals[-1], eigvals[-2]



resolution = 80
delta_IE_vals = np.linspace(-8.0, 6.0, resolution)
delta_EI_vals = np.linspace(-8.0, 6.0, resolution)

grid_complex = np.full((resolution, resolution), np.nan)
grid_imag = np.full((resolution, resolution), np.nan)

print("Calcolo diagramma di fase (solo A e B)...")

for i, d_EI in enumerate(delta_EI_vals):
    for j, d_IE in enumerate(delta_IE_vals):
        l1, _ = compute_eigenvalues(d_EI, d_IE)
        if l1 is None:
            continue
        grid_imag[i, j] = abs(np.imag(l1))
        grid_complex[i, j] = 1.0 if abs(np.imag(l1)) > 1e-8 else 0.0


extent = [delta_IE_vals[0], delta_IE_vals[-1],
          delta_EI_vals[0], delta_EI_vals[-1]]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))


cmap_phase = ListedColormap(['cyan', 'red'])
cmap_phase.set_bad('white')
grid_complex_masked = np.ma.masked_invalid(grid_complex)
axes[0].imshow(grid_complex_masked, extent=extent, origin='lower',
               cmap=cmap_phase, aspect='equal')
axes[0].set_title('Eigenvalue phase diagram', fontsize=16)
axes[0].set_xlabel(r'$\delta_{IE}$', fontsize=16)
axes[0].set_ylabel(r'$\delta_{EI}$', fontsize=16)


grid_imag_masked = np.ma.masked_where(grid_imag < 1e-6, grid_imag)
im1 = axes[1].imshow(grid_imag_masked, extent=extent, origin='lower',
                     cmap='jet', aspect='auto')
axes[1].set_title(r'$|Im(\lambda_1)|$', fontsize=16)
axes[1].set_xlabel(r'$\delta_{IE}$', fontsize=16)
axes[1].set_ylabel(r'$\delta_{EI}$', fontsize=16)
plt.colorbar(im1, ax=axes[1], label=r'$|Im(\lambda_1)|$')

# plt.suptitle('Eigenvalue Phase Diagram (A e B)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Plot completati.")
