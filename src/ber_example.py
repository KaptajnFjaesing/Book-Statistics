#%%
import matplotlib.pyplot as plt
import numpy as np

step = 0.01
w = np.arange(0, 1 + step, step)
n = 10

plt.figure()
plt.plot(w, w * (1 - w) / n, label = "Arithmetic mean")
plt.plot(w, w * (1 - w), label = r"The first data entry, $X_1$")
plt.plot(w, (0.5 - w) ** 2, label = r"The number $0.5$")
plt.xlabel(r"$w$", fontsize=12)
plt.ylabel(r"$\mathbb{E}[(\hat{w} - w)^2]$", fontsize=12)
plt.title("Quadratic Cost for Different Decision Rules (n = 10)", fontsize=14)
plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('./figures/ber_example.pdf')
# %%
