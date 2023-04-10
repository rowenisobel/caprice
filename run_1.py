import lightkurve as lk
from lightkurve import search_targetpixelfile, search_lightcurve
from lightkurve.lightcurve import LightCurve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams["figure.dpi"] = 250
from matplotlib import rc; rc('text', usetex=True); rc('font', family='serif')
from tqdm import tqdm_notebook
import eleanor
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table, join, hstack
from astropy import units as u
import os
import sys
import stella
from scipy.signal import savgol_filter

import stella_rewrite
from stella_rewrite import *
import test
from test import *


import transitleastsquares as tls
from transitleastsquares import (
    transitleastsquares,
    cleaned_array,
    catalog_info,
    transit_mask
    )

import batman
from pylab import *
from astropy.timeseries import LombScargle
from wotan import flatten
from scipy.interpolate import interp1d
import scipy as sy
from scipy.integrate import simpson

from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

from random import uniform, choice

data_table = Table.read('data_table.ecsv', format='ascii.ecsv')
times = np.load('times.npy', allow_pickle=True)
fluxes = np.load('fluxes.npy', allow_pickle=True)
flux_errs = np.load('flux_errs.npy', allow_pickle=True)

modl = create_models(data_table, times, fluxes, 10, per_lims=(2.0, 12.0), rad_lims=(3.0, 10.0))

results, supp_data = find_the_munchkins(data_table, times, fluxes, flux_errs,
                                        4., detrend_method="pspline", injection=True, 
                                        model_dict=modl, chi2_bound=10**(-5), rat_bound=5.)

np.save('results.npy', results, allow_pickle=True)
np.save('supp_data.npy', supp_data, allow_pickle=True)