##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple

def chebyshev_nodes(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)
    
    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    
    if isinstance(n, int) and n > 0:
        pass
    else:
        return None 
    
    wezly = []
    for i in range(n+1):
        wezly.append(np.cos(i*np.pi/n))
    wezly = np.array(wezly)
    return wezly
    
def bar_czeb_weights(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int) and n > 0:
        pass
    else:
        return None
    
    omega = []
    for i in range(n+1):
        if i == 0 or i == n:
            sigma = 0.5
        else:
            sigma = 1
        omega.append(((-1)**i)*sigma)
        
    omega = np.array(omega)
    return omega
    
def  barycentric_inte(xi:np.ndarray,yi:np.ndarray,wi:np.ndarray,x:np.ndarray)-> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(xi, np.ndarray) and isinstance(yi, np.ndarray) and isinstance(wi, np.ndarray) and isinstance(x, np.ndarray):
        pass
    else:
        return None
    if np.size(xi)==np.size(yi)==np.size(wi):
        pass
    else:
        return None
    
    Y = []
    for j in np.nditer(x):
        suma_licznik = 0
        suma_mianownik = 0
        for i in range(len(xi)) :
            wyr = wi[i]/(j - xi[i])
            suma_mianownik += wyr
            suma_licznik += wyr * yi[i]
                
        Y.append(suma_licznik/suma_mianownik)
    Y = np.array(Y)
    return Y


    
def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(xr, (int, float, List, np.ndarray)) and isinstance(x, (int, float, List, np.ndarray)):
        pass
    else:
        return np.nan
    if type(xr) == type(x):
        pass
    else:
        return np.nan
    if not isinstance(x, (float, int)):
        if isinstance(xr, List) and len(xr) == len(x):
            roznica = 0
            for i in range(len(xr)):
                if abs(xr[i]-x[i]) > roznica:
                    roznica = abs(xr[i]-x[i])
            return roznica
        elif isinstance(xr, np.ndarray) and np.shape(xr) == np.shape(x):
            pass
        else:
            return np.nan
    
    return np.max(abs(xr - x))
        


def nierozniczkowalna(x: int)-> float:
    return np.sign(x) * x + x ** 2

def rozniczkowalna_jednokrotnie(x: int)-> float:
    return np.sign(x) * (x ** 2)

def rozniczkowalna_trzyrotnie(x: int)-> float:
    return np.abs(np.sin(5 *x)) ** 3

def pierwsza_analityczna(x: int)-> float:
    return 1 / (1 + 1 * (x ** 2))

def druga_analityczna(x: int)-> float:
    return 1 / (1 + 25 * (x ** 2))

def trzecia_analityczna(x: int)-> float:
    return 1 / (1 + 100 * (x ** 2))

def nieciagla(x: int)-> int:
    return np.sign(x)