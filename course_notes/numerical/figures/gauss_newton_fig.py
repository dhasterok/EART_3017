import numpy as np
import matplotlib.pyplot as plt

# Define - Phi(m-h)) / (2*h)
# Define nonlinear objective function
def Phi(m): 
    #return (m - 1)**2 + 0.5 * np.sin(3*m)
    return (m - 1)**2 + 0.2 * np.sin(2*m)

# First derivative (gradient in 1D)
def dPhi(m):
    h = 1e-5
    return (Phi(m + h) - Phi(m - h)) / (2*h)

# Second derivative (curvature)
def d2Phi(m):
    h = 1e-5
    return (Phi(m+h) - 2*Phi(m) + Phi(m-h)) / (h**2)

# Domain
m = np.linspace(-2, 3, 400)

# Current model
m_i = -1.0
Phi_i = Phi(m_i)
g_i = dPhi(m_i)
H_i = d2Phi(m_i)

# Newton step
m_next = m_i - g_i / H_i

# Local domain for approximations
m_local = np.linspace(m_i-1.5, m_i+3.0, 200)

# Quadratic approximation (objective view)
Phi_quad = Phi_i + g_i*(m_local-m_i) + 0.5*H_i*(m_local-m_i)**2

# Tangent to gradient (gradient view)
g_local = g_i + H_i*(m_local - m_i)

# ---------- FIGURE 1: Objective ----------
plt.figure()
plt.plot(m, Phi(m), label=r'$\Phi(m)$')
plt.plot(m_local, Phi_quad, '--', label='Quadratic approximation')

plt.axvline(m_i, linestyle=':', color='gray')
plt.axvline(m_next, linestyle=':', color='gray')

plt.scatter([m_i], [Phi_i])
plt.scatter([m_next], [Phi(m_next)])

plt.annotate(r'$m_i$', (m_i, Phi_i), xytext=(0,10), textcoords='offset points', ha='center')
plt.annotate(r'$m_{i+1}$', (m_next, Phi(m_next)), xytext=(0,10), textcoords='offset points', ha='center')

plt.xlabel('Model parameter $m$')
plt.ylabel(r'Objective $\Phi(m)$')
plt.title('Gauss--Newton Step (Objective View)')
plt.legend()
plt.grid()

plt.savefig('gauss_newton_objective.png')
plt.close()

# ---------- FIGURE 2: Gradient ----------
plt.figure()
plt.plot(m, dPhi(m), label=r'$\nabla \Phi(m)$')
plt.plot(m_local, g_local, '--', label='Tangent line')

plt.axhline(0, color='black', linewidth=1)

plt.axvline(m_i, linestyle=':', color='gray')
plt.axvline(m_next, linestyle=':', color='gray')

plt.scatter([m_i], [g_i])
plt.scatter([m_next], [0])

plt.annotate(r'$m_i$', (m_i, g_i), xytext=(0,10), textcoords='offset points', ha='center')
plt.annotate(r'$m_{i+1}$', (m_next, 0), xytext=(0,10), textcoords='offset points', ha='center')

plt.xlabel('Model parameter $m$')
plt.ylabel(r'$\nabla \Phi(m)$')
plt.title('Equivalent Newton Step (Gradient View)')
plt.legend()
plt.grid()

plt.savefig('gauss_newton_gradient.png')
#plt.close()


# Starting point
m_i = -1.2
Phi_i = Phi(m_i)

# Compute derivatives
g_i = dPhi(m_i)
H_i = d2Phi(m_i)

# Gauss–Newton / Newton step
m_gn = m_i - g_i / H_i
Phi_gn = Phi(m_gn)

# Steepest descent step
alpha = 0.2  # small step size
m_sd = m_i - alpha * g_i
Phi_sd = Phi(m_sd)

# ---------------------------
# Plot
# ---------------------------
plt.figure()

# Objective
plt.plot(m, Phi(m), label=r'$\Phi(m)$')

# Vertical lines
plt.axvline(m_i, linestyle=':', color='gray')
plt.axvline(m_gn, linestyle='--', color='red')
plt.axvline(m_sd, linestyle='--', color='green')

# Points
plt.scatter([m_i], [Phi_i], color='black')
plt.scatter([m_gn], [Phi_gn], color='red')
plt.scatter([m_sd], [Phi_sd], color='green')

# Labels
plt.annotate(r'$m_i$', (m_i, Phi_i),
             xytext=(0,10), textcoords='offset points', ha='center')

plt.annotate(r'$m_{i+1}^{\mathrm{GN}}$', (m_gn, Phi_gn),
             xytext=(0,10), textcoords='offset points', ha='center')

plt.annotate(r'$m_{i+1}^{\mathrm{SD}}$', (m_sd, Phi_sd),
             xytext=(0,10), textcoords='offset points', ha='center')

# Arrows to show movement
plt.arrow(m_i, Phi_i, m_gn - m_i, Phi_gn - Phi_i,
          length_includes_head=True, head_width=0.05, color='red')

plt.arrow(m_i, Phi_i, m_sd - m_i, Phi_sd - Phi_i,
          length_includes_head=True, head_width=0.05, color='green')

# Formatting
plt.xlabel('Model parameter $m$')
plt.ylabel(r'Objective $\Phi(m)$')
plt.title('Gauss–Newton vs Steepest Descent')
plt.legend(['Objective', 'GN step', 'SD step'])
plt.grid()

plt.savefig("gn_vs_sd.png")
plt.close()
