import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import fft, ifft
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import pandas as pd

PRINT_TO_FILE = False               # save the graph to pdf file
GENERATE_CSV = False                # generate a csv file for putting in the 2D graph
STOP_IN_THE_FIRST_SYMBOL = True     # when have an error

if PRINT_TO_FILE or GENERATE_CSV:
    image_folder = r"Path\to\your\folder"
    figure_1 = "Doppler3DMAComplete.pdf"
    archive_csv = image_folder + 'values.csv'
    archive_csv2 = image_folder +'model_droppler_valuesB.csv'
        
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral' 
    
def d_rate(time, vector_search, f0, t_generator):
    index_time = np.abs(t_generator - time).argmin()
    return vector_search[index_time] * f0

def calculate_deltaF(H, t):
    R = 6371000         # radius of the Earth in meters
    g = 9.80665         # gravity acceleration in m/s^2
    c = 299792458       # light speed in vacuum in m/s
    deltaF = 1/(1+((1/c)*np.sqrt((g*R)/(1+H/R)))*(np.sin((np.sqrt(g/R)/(1+H/R)**(3/2))*t)/np.sqrt((1+H/R)**2-2*(1+H/R)*np.cos((np.sqrt(g/R)/(1+H/R)**(3/2))*t)+1)))-1
    return deltaF

def calculateF(H, t, F0):
    R = 6371000         # radius of the Earth in meters
    g = 9.80665         # gravity acceleration in m/s^2
    c = 299792458       # light speed in vacuum in m/s
    F = 1/(1+((1/c)*np.sqrt((g*R)/(1+H/R)))*(np.sin((np.sqrt(g/R)/(1+H/R)**(3/2))*t)/np.sqrt((1+H/R)**2-2*(1+H/R)*np.cos((np.sqrt(g/R)/(1+H/R)**(3/2))*t)+1)))*F0
    return F

def calculate_estimate_doppler(SF, s, p, Ts, ti, H, f0):
    N = 2 ** SF                 # number of samples
    n = np.arange(0, N)         # generates the samples
    T = Ts * N                  # sampling period
    time = ti + (p - 1) * T

    DR = d_rate(time, deltaF_ppm_sec, f0, tg)
    DS = sum(vector_DS)

    y = np.exp(2 * np.pi * 1j * n * ((s / N) + DS * Ts + n * DR * (Ts ** 2) / 2))
    y = y.astype(np.complex64)
    
    DR_acum = DR * T
    vector_DS.append(DR_acum)
    
    fft_abs = np.abs(np.array(fft(y)))
    estimated_symbol = np.argmax(fft_abs)

    return fft_abs, estimated_symbol, fft_abs[estimated_symbol]

def calculate_doppler_graph(SF, B, Ts, line, column, graph,H,f0):
    number_symbols = 300        # maximum number of symbols to be estimated
    N = 2 ** SF                 # amont of samples
    fs = B                      # sampling rate
    Ts = 1 / B                  # sampling period
    n = np.arange(0,N)          # generates the samples
    Rs = B / N                  # frequency resolution
    s = 512                     # choose the symbol to be estimated
    am = 10                     # define the interval of bins that will appear in the graph N/2 - 10 to N/2 + 10

    if STOP_IN_THE_FIRST_SYMBOL:
        start = int(s - am)
        end = int(s + am)
        current_ticks_x = [start, end]
    else:
        start = 0
        end = 2 * s  
        current_ticks_x= [start, end]

    previous_estimated_symbol = 0       # use to keep track of the last estimated symbol
    p_values = []                       # list of p values for the y-axis
    p_wrong = []                        # list of p values where the estimated symbol is wrong
    last_estimated = 0                  # detect when the value varies

    ax = fig.add_subplot(line, column, graph, projection='3d')
    ax.view_init(azim = -45, elev = 32)

    vector_DS.clear()
    vector_DS.append(0)

    for p in range(1, number_symbols + 1):
        fft_abs, estimate_symbol, peak_value_fft = calculate_estimate_doppler(SF, s, p, Ts, ti, H, f0)
        peak_value_fft = peak_value_fft / (N - 1)
        
        if GENERATE_CSV:   
            peak_values_fft.append(peak_value_fft)
            values_sf.append(SF)
        
        sinal_seg = fft_abs[start:end]
        sinal_seg = sinal_seg / (N - 1)

        n_seg = n[start:end]
        ax.plot(n_seg,sinal_seg, zs = p, zdir = 'y', color = FFTcolor, linestyle = '-', lw = lwFFT, alpha = valorAlpha)
        p_values.append(p)

        if estimate_symbol not in current_ticks_x:
            current_ticks_x = np.sort(np.append(current_ticks_x, estimate_symbol))
            ax.set_xticks(current_ticks_x)
        
        if estimate_symbol != s and estimate_symbol != previous_estimated_symbol:
            ax.plot(n_seg, sinal_seg, zs = p, zdir = 'y', color = 'red', linestyle = '-', lw = lwFFT)
            print(f'p:{p}')    
            if STOP_IN_THE_FIRST_SYMBOL:
                break
            else:
                p_wrong.append(p)   
        previous_estimated_symbol = estimate_symbol

    # plot graph adjustments

    ax.set_yticks(p_values)
    ax.set_xlim(start, end)
    ax.set_zlim3d([0,1])
    ax.set_zticks([0,1])
    ax.set_xticks(current_ticks_x)
    ax.set_xticklabels([str(int(t)) for t in current_ticks_x])

    for label, val in zip(ax.get_xticklabels(), current_ticks_x):
        if val not in [int(s), start, end]:
            label.set_color('red')
        if SF >= 10:
            xtick_labels = [str(int(t)) + ('        '  if t not in [int(s), start, end] else '') for t in current_ticks_x]
        else:
            xtick_labels = [str(int(t)) + ('            '  if t not in [int(s), start, end] else '') for t in current_ticks_x]
    
    ax.set_xticklabels(xtick_labels)
    ax.set_zlabel(r'$ \frac{ | \ Y[k,p] \ |}{N-1} $', fontsize = fonteTituloEixo, labelpad = zlabelPad)
    ax.set_zlabel(r'$  | \ \tilde{Y}[k,p] \ | $', fontsize = fonteTituloEixo, labelpad = zlabelPad)
    ax.set_xlabel(r'$k$', fontsize = fonteTituloEixo, labelpad = xlabelPad)
    ax.set_ylabel(r'$p \ symbol$', fontsize = fonteTituloEixo, labelpad = ylabelPad)
    ax.tick_params(axis = 'x', labelsize = fonteTick, pad = tickPadX)
    ax.tick_params(axis = 'y', labelsize = fonteTick, pad = tickPadY)  
    ax.tick_params(axis = 'z', labelsize = fonteTick, pad = tickPadZ)  
    ax.set_title(rf'$ SF={SF}$', fontsize = fonteTituloEixo, pad = titlePad)

    if STOP_IN_THE_FIRST_SYMBOL:
        y_start = 1
        y_end = p
        y_ticks_middle = np.linspace(y_start, y_end, 7, dtype=int)[1:-1]
        y_ticks = np.concatenate(([y_start], y_ticks_middle, [y_end]))
        ax.set_yticks(y_ticks)
        ytick_labels = ax.get_yticklabels()
        if ytick_labels:
            ytick_labels[-1].set_color('red')
    else:
        y_start = 1
        y_end = number_symbols
        y_ticks_middle = p_wrong
        y_ticks = np.concatenate(([y_start], y_ticks_middle, [y_end]))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(y) for y in y_ticks])
        for label in ax.get_yticklabels():
            value = int(label.get_text())
            if value in p_wrong:
                label.set_color('red')
          
if __name__ == "__main__":
    peak_values_fft = []
    values_sf = []
    values_doppler = []
    sample_values_sf = []
    vector_DS = []

    # graph configuration
    
    lwid = 2
    lwid1 = 1.5
    lwFFT = 1 
    fonteTituloEixo = 14
    fonteTick = 11
    ylabelPad = -2
    zlabelPad = -12
    xlabelPad = -2
    tickPadX = -3
    tickPadY = -3
    tickPadZ = 0
    titlePad = -50
    FFTcolor = 'black'  
    espacoTextoPico = 10
    multiplo = 20
    fonteValorMs = 15
    tamanhoDoPonto = 18
    valorAlpha = 0.7          

    fig = plt.figure(figsize=(14, 6))

    B = 125e3                               # bandwidth
    Ts = 1 / B                              # sampling period
    H = 550000                              # altitude in meters
    f0 = 436900000                          # frequency of the carrier
    ti = 0                                 # initial time
    tg = np.arange(-300, 300.01, 0.001)     # calculate the time vector with a resolution of 0.001

    deltaFppm = calculate_deltaF(H,tg)
    deltaF_ppm_sec = np.gradient(deltaFppm) / np.gradient(tg)

    line = 1        # number line of the subplot
    column = 3      # number column of the subplot
                                                 
    calculate_doppler_graph(10, B, Ts, line, column, 1,H,f0)
    calculate_doppler_graph(11, B, Ts, line, column, 2,H,f0)
    calculate_doppler_graph(12, B, Ts, line, column, 3,H,f0)

    if PRINT_TO_FILE:
        plt.savefig(image_folder + figure_1, bbox_inches = 'tight') # bbox_inches='tight' # prevents cuts
    if GENERATE_CSV:
        df = pd.DataFrame({'SF': values_sf,'valorPicoFFTnormalizado': peak_values_fft})
        df.to_csv(archive_csv, header=True, index=False)
        df2 = pd.DataFrame({'SF': sample_values_sf,'valoresDoppler': peak_values_fft})
        df2.to_csv(archive_csv2, header=True, index=False)
    plt.show()