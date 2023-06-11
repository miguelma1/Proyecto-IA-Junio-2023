from statistics import mode
import pandas as pd
import numpy as np
import scipy.stats 


def prediccionAlg(datosTesteo, conjunto):
    l = list()
    for i in conjunto:
        pred = i.predict(datosTesteo)
        l.add(pred)
    return mode(l)