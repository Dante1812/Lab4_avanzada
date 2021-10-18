import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

def Regresion(x, y, f, beta0):
    model = odr.Model(f)
    data = odr.RealData(x, y)
    myodr = odr.ODR(data, model, beta0)
    out = myodr.run()
    popt = out.beta
    perr = out.sd_beta
    x_fit = np.linspace(min(x), max(x), 100)
    fit = f(popt, x_fit)
    return popt, perr, x_fit, fit

def f(B, x):
    return B[0] + B[1]*x + B[2]*x**2

l = np.array([600, 629.647, 663.529, 703.764, 751.764, 807.528, 873.882])
nd = np.array([7200.00, 6296.47, 6303.53, 5630.11, 5638.23, 5248.93, 5243.29])

nd_coef, I_nd_coef, nd_x, nd_fit = Regresion(l, nd, f, [1, 1, 1])
print('Coeficientes de ajuste para curva nd vs lambda e incertidumbres')
print(nd_coef)
print(I_nd_coef)
print('')
print('Valor de nd para lambda = 650 nm')
print(f(nd_coef, 650))
print('')
print('Valor de d')
print(f(nd_coef, 650)/2.14)
print('')
nd2 = nd/(f(nd_coef, 650)/2.14)

nd_coef2, I_nd_coef2, nd_x2, nd_fit2 = Regresion(l, nd2, f, [1, 1, 1])
print('Coeficientes de ajuste para curva n vs lambda e incertidumbres')
print(nd_coef2)
print(I_nd_coef2)

plt.figure(figsize = (8, 8))
plt.plot(l, nd, 'ko')
plt.plot(nd_x, nd_fit, 'k--')
plt.legend(['Datos', 'Ajuste'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$\lambda$ [nm]', fontsize = 15)
plt.ylabel(r'$nd$ [nm]', fontsize = 15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(l, nd2, 'ko')
plt.plot(nd_x2, nd_fit2, 'k--')
plt.legend(['Datos', 'Ajuste'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$\lambda$ [nm]', fontsize = 15)
plt.ylabel(r'$n$', fontsize = 15)
plt.grid()
plt.show()