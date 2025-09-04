import numpy as np
import matplotlib.pyplot as plt
        
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def curve_calculate(H, t, F0):
    R = 6371000
    g = 9.80665
    c = 299792458
    F = 1/(1+((1/c)*np.sqrt((g*R)/(1+H/R)))*(np.sin((np.sqrt(g/R)/(1+H/R)**(3/2))*t)/np.sqrt((1+H/R)**2-2*(1+H/R)*np.cos((np.sqrt(g/R)/(1+H/R)**(3/2))*t)+1)))*F0
    return F

def verify_static_doppler_error(tg, fvector, f0, max_tolerate):
    yes_indices = []
    no_indices = []
    for i in range(len(tg)):
        error = abs(fvector[i] - f0)
        if error < max_tolerate:
            yes_indices.append(i)
        else:
            no_indices.append(i)
    return yes_indices, no_indices

if __name__ == "__main__":
    SF = 12
    B = 125e3
    L = 128

    if SF == 12:
        max_tolerate = B / 50
    elif SF == 11:
        max_tolerate = B / 100
    elif SF == 10:
        max_tolerate = B / 200
    else:
        max_tolerate = B / 4

    H = 550000
    f0 = 436900000
    tg = np.arange(-300, 300.01, 1)

    fvector = curve_calculate(H, tg, f0)
    yes_indices_static, no_indices_static = verify_static_doppler_error(tg, fvector, f0, max_tolerate)
    
    plt.plot()

    if yes_indices_static:
        plt.scatter(tg[yes_indices_static], fvector[yes_indices_static], color='blue', label='Within Tolerance', s=3)
    
    if no_indices_static:
        plt.scatter(tg[no_indices_static], fvector[no_indices_static], color='red', label='Out of Tolerance', s=3)

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()