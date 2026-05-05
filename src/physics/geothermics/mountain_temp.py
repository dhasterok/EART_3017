import numpy as np
import matplotlib.pyplot as plt

# Months
months = np.arange(1, 13)

# Approximate 1991–2020 monthly mean temperatures (°C)
# Colorado Springs (east side)
# Innsbruck (Austria)
innsbruck = np.array([
    -2, 0, 4, 8, 13, 17,
    19, 18, 14, 9, 3, -1
])

# Bolzano (Italy)
bolzano = np.array([
    1, 3, 7, 11, 15, 19,
    21, 20, 16, 11, 5, 1
])


# Difference (East - West)
diff = innsbruck - bolzano

# -----------------------------
# Plot 1: monthly temperatures
# -----------------------------
plt.figure()
plt.plot(months, innsbruck, label="Innsbruck (Austria)")
plt.plot(months, bolzano, label="Bolzano (Italy)")
plt.xticks(months)
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.title("Monthly Mean Temperature: Swiss Alps North vs South")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Plot 2: difference
# -----------------------------
plt.figure()
plt.plot(months, diff, color="black")
plt.xticks(months)
plt.xlabel("Month")
plt.ylabel("ΔT (°C)  [Innsbruck − Bolzano]")
plt.title("Thermal Contrast Across the Swiss Alps")
plt.axhline(0, color="gray", linewidth=1)
plt.grid(True)
plt.show()