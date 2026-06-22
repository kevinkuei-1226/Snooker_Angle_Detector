import matplotlib
matplotlib.use('qtagg')  # Bypasses Homebrew completely using the Qt package we just installed
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.show()