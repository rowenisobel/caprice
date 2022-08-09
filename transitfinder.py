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
from astropy.table import Table
from astropy import units as u
import os
import sys
import stella
from scipy.signal import savgol_filter


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

__all__ = ['transit_model', 'transit_check', 'k2_reduction', 'tess_reduction', 'lowess_iron', 
            'lc_chi2_output', 'remove_flares_tess', 'init_stella', 'model_generator', 'lc_characteristic_extractor', 'create_models']


def transit_model(time_arr, t0, period=1., radius=0.03, sm_axis=2., orb_inc=87.0, ecc=0., 
                    long_per=90., ab=[0.1, 0.3], limb_dark="quadratic"):
    
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = period                      #orbital period
    params.rp = radius                      #planet radius (in units of stellar radii)
    params.a = sm_axis                       #semi-major axis (in units of stellar radii)
    params.inc = orb_inc                     #orbital inclination (in degrees)
    params.ecc = ecc                     #eccentricity
    params.w = long_per                       #longitude of periastron (in degrees)
    params.u = list(ab)                #limb darkening coefficients [u1, u2]
    params.limb_dark = limb_dark
    
    m = batman.TransitModel(params, time_arr)
    flux_co = m.light_curve(params) -1 
    return flux_co

def transit_check(time_arr, flux_arr, ind, trend, window=50, 
                    period=1., radius=0.03, sm_axis=2., orb_inc=87.0, ecc=0., 
                    long_per=90., ab=[0.1, 0.3], limb_dark="quadratic"):
    t0 = time_arr[ind]
    fl_mod_coeff0 = transit_model(np.array(time_arr), t0, period=period, 
                                    radius=radius, sm_axis=sm_axis, orb_inc=orb_inc, 
                                    ecc=ecc, long_per=long_per, ab=ab, limb_dark=limb_dark)
    mod_thing = np.zeros(len(time_arr))
    mod_thing[ind - window : ind + window] = fl_mod_coeff0[ind - window : ind + window]
    mod = flux_arr + mod_thing
    
    chi2_0 = ((flux_arr[3:-3] - mod[3:-3])**2) / mod[3:-3]
    trend_nans = np.argwhere(np.isnan(trend))


    chi2 = np.nansum(chi2_0)

    return chi2

def k2_reduction(path, filename, pad=24):
    
    hdu = fits.open(os.path.join(path, filename))
    
    ID = int(filename.split('_')[4][:-4])
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(EPIC_ID=ID)
    ab = list(ab)
    flux_key = 'FLUX'
    
    """
    Here we identify and clean the time and flux data contained in the FITS file input.
    """
    
    time = hdu[1].data['TIME']
    flux = hdu[1].data[flux_key]
    time, flux = cleaned_array(time, flux)  # remove invalid values such as nan, inf, non, negative
    flux = flux / np.median(flux)
    
    """
    Here we cut off the first and last 24 data points of the light curve. Then, we flatten the curve 
    and identify the trend with the Wotan 'lowess' method.
    """
    
    time_arr = time[pad:-pad]
    flux_arr = flux[pad:-pad]

    results = [ab,time_arr, flux_arr]

    return results

def tess_reduction(path, filename, pad=125, models=None):

    hdu = fits.open(os.path.join(path, filename))
    
    ID = hdu[0].header['TICID']
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=ID)
    ab = list(ab)

    """
    Here we identify and clean the time and flux data contained in the FITS file input.
    """

    time = hdu[1].data['TIME']
    flux = hdu[1].data['PDCSAP_FLUX']
    flux_error = hdu[1].data['PDCSAP_FLUX_ERR']
    time, flux, flux_error = cleaned_array(time, flux, flux_error)  # remove invalid values such as nan, inf, non, negative
    flux = flux / np.median(flux)

    """
    Here we find the indices where the break between orbits is documented and remove
    those bad time and flux data. We break the time and flux arrays into two - one for 
    each orbit. This step is NOT taken for K2 light curves, because they
    do not have multiple orbits.

    We follow up the separation of the light curve with a detrend using Wotan's 'lowess' method.
    """

    brie = np.where(max(np.diff(time)) == np.diff(time))[0][0]
    low = brie 
    up = brie + 1

    time1 = time[:low]
    time1 = time1[pad:-pad]
    time2 = time[up:]
    time2 = time2[pad:-pad]

    flux1 = flux[:low]
    flux1 = flux1[pad:-pad]
    flux2 = flux[up:]
    flux2 = flux2[pad:-pad]

    flux_err1 = flux_error[:low]
    flux_err1 = flux_err1[pad:-pad]
    flux_err2 = flux_error[up:]
    flux_err2 = flux_err2[pad:-pad]

    time_f = np.concatenate([time1, time2])
    flux_f = np.concatenate([flux1, flux2])
    flux_err_f = np.concatenate([flux_err1, flux_err2])

    """
    if models != None:
        print(len(models[0]))
        models_stripped = []
        for m in models:
            m1 = m[:low]
            m1 = m1[pad:-pad]
            m2 = m[up:]
            m2 = m2[pad:-pad]
            m = np.concatenate([m1, m2])
            models_stripped.append(m)
    else:
        models_stripped = 'No injection'
    """
 
    return ab, time_f, flux_f, flux_err_f, models
    


def lowess_iron(time, flux, window_len=0.375, break_tol=0.5, method='lowess'):
    
    flat_flux, trend = flatten(
        time,
        flux,
        method='lowess',
        window_length=window_len,
        break_tolerance=break_tol,
        return_trend=True,
    )
    
    return flat_flux, trend


def remove_flares_tess(time, flux, flux_error, stella_models, neural_net):
    """
    Use Stella to first identify and then remove the datapoints corresponding to
    probable flares in a given light curve.
    """
    cnn = neural_net
    preds = []
    for i, model in tqdm_notebook(enumerate(stella_models)):
        cnn.predict(modelname=model,
                   times=time,
                   fluxes=flux,
                   errs=flux_error)
       
        preds.append(cnn.predictions)

    avg_pred = np.nanmedian(preds, axis=0)

    indices_to_del = np.where(avg_pred[0] > 0.99)

    for p in sorted(indices_to_del, reverse=True):
        time = np.delete(time, p)
        flux = np.delete(flux, p)

    return time, flux


def lc_chi2_output(path=None, filename=None, injection=False, model_dict=None,
                    window_length=0.375, break_tolerance=0.5, detrend_method='lowess'):
    
    """
    Check if the file is a K2 or TESS file and identify its characteristics appropriately.
    """

    hdu = fits.open(os.path.join(path, filename))
    
    if 'k2' in filename:
        ab, time_arr, flux_arr = k2_reduction(path, filename)
        flat_flux, trend = lowess_iron(time_arr, flux_arr, method=detrend_method)
        
    elif 'tess' in filename:
        
        MODELS, cnn = init_stella()

        #create n_inj models using transit_model, store them in array

        if injection == True:

            file_specific_models = model_dict[filename]

            ab, time, flux, flux_err, models_stripped = tess_reduction(path, filename, models=file_specific_models)

            time_arr, flux_arr = remove_flares_tess(time, flux, flux_err, MODELS, cnn)

            chi2_array = np.zeros((len(models_stripped), len(time_arr)))

            for m in tqdm_notebook(range(len(file_specific_models))):
                mod = models_stripped[m]

                flux_synth = flux_arr + mod    

                flat_flux, trend = lowess_iron(time_arr, flux_synth, window_len = window_length, 
                                                break_tol=break_tolerance, method=detrend_method)

                chi2_array_single = np.zeros(len(time_arr))
                for i in range(len(time_arr)):
                    chi2 = transit_check(time_arr, flux_synth, i, trend)
                    chi2_array_single[i] = chi2

                chi2_array[m] = chi2_array_single

            return time_arr, flux, flux_synth, chi2_array


        else:
            ab, time, flux, flux_err, models = tess_reduction(path, filename)

            time_arr, flux_arr = remove_flares_tess(time, flux, flux_err)

            flat_flux, trend = lowess_iron(time_arr, flux_arr, window_len = window_length, 
                                                break_tol=break_tolerance, method=detrend_method)

            """
            Here we slide through the time and flux chunks. For each slice of the light curve, we 
            calculate a best fit curve and transit model, then measure the chi-squared value of
            the flux data against the model. The chi-squared value for each datapoint is appended
            to an array of chi-squared values for the entire light curve. This array effectively
            represents the likelihood of a transit at every point on the curve.
            """

            chi2_array = np.zeros(len(time_arr))
    
            for i in range(len(time_arr)):
                chi2 = transit_check(time_arr, flux_arr, i, trend)
                chi2_array[i]=chi2
                
            return time_arr, flux_arr, chi2_array
        
def init_stella():

    """
    """

    ds = stella.DownloadSets()
    ds.download_models()
    MODELS = ds.models
    OUT_DIR = '/Users/rowenglusman/Summer2022/AU_Mic/'
    cnn = stella.ConvNN(output_dir=OUT_DIR)

    return MODELS, cnn

def model_params_randomizer(model_number, per_lims, rad_lims):

    p1, p2 = per_lims
    r1, r2 = rad_lims

    mod_settings={}

    poss_per = np.zeros(model_number)
    poss_rad = np.zeros(model_number)

    for i in range(model_number):
        poss_per[i] = uniform(p1, p2)
        poss_rad[i] = uniform(r1, r2)

    poss_per = poss_per * u.d
    poss_rad = poss_rad * u.earthRad

    mod_settings['periods'] = poss_per #in days
    mod_settings['radii_earth'] = poss_rad #in earth radii

    return mod_settings

def lc_characteristic_extractor(path, files, MODELS, cnn):

    lc_char = {}
    for file in files:
        lc_char[file] = {}
        hdu = fits.open(os.path.join(path, file))
        ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=hdu[0].header['TICID'])
        ab = list(ab)
        radius = radius * u.solRad
        radius = radius.to(u.earthRad)
        
        #time, flux, flux_err = hdu[1].data['TIME'], hdu[1].data['PDCSAP_FLUX'], hdu[1].data['PDCSAP_FLUX_ERR']
        #time_clean, flux_clean, flux_err_clean = cleaned_array(time, flux, flux_err)

        ab, time, flux, flux_err, not_inj = tess_reduction(path, file)
        time_shaved, flux_shaved = remove_flares_tess(time, flux, flux_err, MODELS, cnn)

        lc_char[file]['ab'] = ab
        lc_char[file]['star_radius'] = radius #in earth radii
        lc_char[file]['time'] = time_shaved
        lc_char[file]['flux'] = flux_shaved

    return lc_char

def create_models(file, model_number, lc_char, mod_settings):

    time = lc_char[file]['time']
    flux = lc_char[file]['flux']
    star_rad = lc_char[file]['star_radius']
    ab = lc_char[file]['ab']
    
    possible_planet_radii = mod_settings['radii_earth'] / star_rad  #in stellar radii
    possible_planet_periods_y = mod_settings['periods'].to(u.yr)      #in years

    
    file_mods=[]
    for k in range(model_number):
        t0 = choice(time)
        per = mod_settings['periods'][k].value
        rad = possible_planet_radii[k].value
        
        p = possible_planet_periods_y[k]
        sm_ax = p**(2/3)
        a = ((p**(2/3))) / star_rad.to(u.AU)

        params_inj = batman.TransitParams()
        params_inj.t0 = t0                       #time of inferior conjunction
        params_inj.per = per                      #orbital period
        params_inj.rp = rad                      #planet radius (in units of stellar radii)
        params_inj.a = a.value                       #semi-major axis (in units of stellar radii)
        params_inj.inc = 87.0                     #orbital inclination (in degrees)
        params_inj.ecc = 0.                     #eccentricity
        params_inj.w = 90.                       #longitude of periastron (in degrees)
        params_inj.u = ab               #limb darkening coefficients [u1, u2]
        params_inj.limb_dark = 'quadratic'

        m = batman.TransitModel(params_inj, time)
        flux_co = m.light_curve(params_inj) -1 
        
        file_mods.append(flux_co)

    return file_mods

def model_generator(path, model_number, per_lims=(1., 8.), rad_lims=(2., 12.)):

    files0 = os.listdir(path)
    files1 = [i for i in files0 if i.endswith('_lc.fits')]
    files = np.sort(files1)

    MODELS, cnn = init_stella()

    lc_char = lc_characteristic_extractor(path, files, MODELS, cnn)
    mod_settings = model_params_randomizer(model_number=model_number, per_lims=per_lims, rad_lims=rad_lims)
    
    models = {}
    for file in files:
        models[file] = create_models(file, model_number, lc_char, mod_settings)

    return models


