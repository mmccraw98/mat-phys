from optimization import harmonic_bond
import numpy as np

p1 = np.array([0.0, -1.5, 0.0])
p2 = np.array([0.0, 0.0, 0.0])
p3 = np.array([2.0, 0.0, 0.0])
p4 = np.array([0.0, 5.0, -1.0])

OH = harmonic_bond(1, 3)
HH = harmonic_bond(2, 10)

potential = np.inf
while potential > 1e-10:

    p2_grad = OH.gradient_wrt_r2(p1, p2) + HH.gradient_wrt_r1(p2, p3)
    p3_grad = HH.gradient_wrt_r2(p2, p3) + OH.gradient_wrt_r1(p3, p4)
    p4_grad = OH.gradient_wrt_r2(p3, p4)

    p2 -= p2_grad / 1000
    p3 -= p3_grad / 1000
    p4 -= p4_grad / 1000

    potential = OH.potential(p1, p2) + HH.potential(p2, p3) + OH.potential(p3, p4)

print(p1, p2, p3, p4, potential)
print(np.sqrt(np.sum((p1-p2)**2)), np.sqrt(np.sum((p2-p3)**2)), np.sqrt(np.sum((p3-p4)**2)))