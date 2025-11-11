import matplotlib.pyplot as plt

# Raw values from your six cases (1→M10, …, 6→M16)
y_amp_all  = [6.358174022293425, 7.704602027587635, 6.606856768649342,
              10.356299134713346, 7.711722442333659, 7.822514744083444]
y_freq_all = [5.373323705264275, 5.856343993664965, 5.617346703748377,
              5.873357421098144, 5.866893006771267, 5.907647989346704]

# Map cases to Mj indices
Mj_all = [10, 11, 12, 13, 14, 16]

# Exclude M15
mask = [m != 15 for m in Mj_all]
Mj      = [m for m, keep in zip(Mj_all, mask) if keep]
y_amp   = [v for v, keep in zip(y_amp_all, mask) if keep]
y_freq  = [v for v, keep in zip(y_freq_all, mask) if keep]

# ---- Plot both curves on one figure ----
plt.figure(figsize=(8, 5))
plt.plot(Mj, y_amp,  marker='o', linewidth=2, label='y Amplitude (um)')
plt.plot(Mj, y_freq, marker='s', linewidth=2, label='y Frequency (MHz)')
plt.title('Ion Oscillation Metrics vs Multipole Coefficient Index (Mj)')
plt.xlabel('Multipole Coefficient Index Mj')
plt.ylabel('Value')
plt.xticks(Mj)             # show integer Mj ticks
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# ---- (Optional) save to file ----
# plt.savefig('Mj_vs_Amplitude_Frequency.png', dpi=300)