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

__all__ = ['tess_reduction', 'remove_flares_tess', 'init_stella', 'init_tess_processing', 'model_params_randomizer', \
            'transit_model', 'create_models', 'transit_check', 'lowess_iron', 'lc_chi2_output', 'calculate_planet_period', \
            'find_the_munchkins']

def tess_reduction(path, filename, pad):

    """
    Inputs: one SPOC light curve
    Outputs: the cleaned time, flux, and flux error arrays, plus the limb-darkening coefficients and stellar radius.
    """

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
 
    return time_f, flux_f, flux_err_f, ab, radius


def remove_flares_tess(time_arr, flux_arr, flux_error_arr, stella_models, cnn):

    """
    Inputs: cleaned time, flux, and flux_error arrays, stella models, and stella cnn
    Outputs: cleaned time, flux, and flux_error arrays stripped of all identified flares.
    """
    preds = []
    for i, model in enumerate(stella_models):
        cnn.predict(modelname=model,
                   times=time_arr,  
                   fluxes=flux_arr,
                   errs=flux_error_arr)

        preds.append(cnn.predictions)
       
    for t in range(len(time_arr)):
        temp_pred_arr = np.zeros((len(stella_models), len(time_arr[t])))
        for j in range(len(stella_models)):
            temp_pred_arr[j] = preds[j][t]
            temp_avg_pred = np.nanmedian(temp_pred_arr, axis = 0)
             
            indices_to_del = np.where(temp_avg_pred[0] > 0.99)
            for p in sorted(indices_to_del, reverse=True):
                time_arr[t] = np.delete(time_arr[t], p)
                flux_arr[t] = np.delete(flux_arr[t], p)
                flux_error_arr[t] = np.delete(flux_error_arr[t], p)

    return time_arr, flux_arr, flux_error_arr

def init_stella():

    """
    Initiates Stella's convolutional neural network and downloads its base models.
    """

    ds = stella.DownloadSets()
    ds.download_models()
    MODELS = ds.models
    OUT_DIR = '/Users/rowenglusman/Summer2022/AU_Mic/'
    cnn = stella.ConvNN(output_dir=OUT_DIR)

    return MODELS, cnn

def init_tess_processing(path, catalog, pad=125):

    """
    Inputs: filepath
    Outputs: a Table of cleaned and de-flared time, flux, and flux_error arrays alongside the accompanying 
            filenames, TIC IDs, limb-darkening coefficients, and stellar radii.
    """

    files0 = os.listdir(path)
    files = [i for i in files0 if i.endswith('_lc.fits')]

    filenames = []
    tics = []
    times_c = []
    fluxes_c = []
    flux_errs_c = []
    ab_arr = []
    radius_arr = []

    print('Reducing SPOC Data...')
    for j in tqdm_notebook(range(len(files))):
        filename = files[j]
        tic = int(filename.split('-')[2].lstrip('0'))

        time_c, flux_c, flux_err_c, ab, radius = tess_reduction(path, filename, pad)

        filenames.append(filename)
        tics.append(tic)
        times_c.append(time_c)
        fluxes_c.append(flux_c)
        flux_errs_c.append(flux_err_c)
        ab_arr.append(ab)
        radius_arr.append(radius)

    MODELS, cnn = init_stella()

    times_s, fluxes_s, flux_errs_s = remove_flares_tess(times_c, fluxes_c, flux_errs_c, MODELS, cnn)

    times = []
    fluxes = []
    flux_errs =[]

    print('Outlier removal in progress...')
    for k in tqdm_notebook(range(len(times_s))):
        temp_lc = LightCurve(time=times_s[k], flux=fluxes_s[k], flux_err=flux_errs_s[k]).remove_outliers()
        times.append(temp_lc.time.value)
        fluxes.append(temp_lc.flux.value)
        flux_errs.append(temp_lc.flux_err.value)

    res = Table([filenames, tics, times, fluxes, flux_errs, ab_arr, radius_arr], \
                names=('filename', 'tic', 'time', 'flux', 'flux_err', 'ab', 'star_rad'))
    tic_per_cat = Table([catalog['tic'], catalog['period']], names=('tic', 'star_period'))
    
    data_table = join(res, tic_per_cat, 'tic')
    
    return data_table


"""
INJECTION MODELS
"""

def model_params_randomizer(model_number, per_lims, rad_lims):

    """
    Takes the limits given for the period and radius of a transit model and creates an array of length
    model_number containing a randomized period and radius within the bounds. Each row of the resulting 
    array will hold the settings used to generate that model.
    """

    mod_settings={}

    poss_per = np.random.uniform(low=per_lims[0], high=per_lims[1], size=(model_number,))
    poss_rad_earth = (np.random.uniform(low=rad_lims[0], high=rad_lims[1], size=(model_number,))) * u.earthRad

    poss_per = poss_per * u.d

    mod_settings['periods'] = poss_per #in days
    mod_settings['radii_earth'] = poss_rad_earth #in earth radii

    return mod_settings

def transit_model(time_arr, t0, period, radius, sm_axis, ab, orb_inc=87.0, \
                ecc=0., long_per=90., limb_dark="quadratic"):

    """
    This function creates a transit model using the batman package. Such a model can either be injected into a lightcurve
    or passed over a lightcurve to detect a true transit.
    """

    
    params = batman.TransitParams()
    params.t0 = t0                           #time of inferior conjunction
    params.per = period                      #orbital period
    params.rp = radius.value                       #planet radius (in units of stellar radii)
    params.a = sm_axis                      #semi-major axis (in units of stellar radii)
    params.inc = orb_inc                     #orbital inclination (in degrees)
    params.ecc = ecc                         #eccentricity
    params.w = long_per                      #longitude of periastron (in degrees)
    params.u = list(ab)                      #limb darkening coefficients [u1, u2]
    params.limb_dark = limb_dark
    
    m = batman.TransitModel(params, time_arr)
    flux_co = m.light_curve(params) - 1.0
    
    return flux_co

def create_models(data_table, model_number, per_lims, rad_lims):

    """
    Takes the number of desired models, the period and radial limits thereof,
    and data table created above and creates a dictionary of models, their radii, and  
    their periods arranged by TIC.
    """
    mod_settings = model_params_randomizer(model_number, per_lims, rad_lims)

    models = {}

    for i in tqdm_notebook(range(len(data_table))):
        file_data = data_table[i]
        filename = file_data['filename']
        tic = file_data['tic']
        time = file_data['time']
        flux = file_data['flux']
        star_rad = file_data['star_rad'] * u.solRad
        ab = file_data['ab']
    
        poss_radii_sol = mod_settings['radii_earth'].to(u.solRad)  #in solar radii
        poss_periods_yr = mod_settings['periods'].to(u.yr)         #in years
        sm_axes = (poss_periods_yr ** (2/3))
    

        file_mods=[]
        t0s = []
        for k in range(model_number):
            t0 = choice(time)
            t0s.append(t0)

            mod_per = poss_periods_yr[k].to(u.d).value
            mod_rad = poss_radii_sol[k].value / star_rad
            a = (sm_axes[k].value*u.AU).to(u.solRad).value

            flux_co = transit_model(time, t0, mod_per, mod_rad, a, ab)
        
            file_mods.append(flux_co)


        models[filename] = {}
        models[filename]['tic'] = tic
        models[filename]['mod_radius'] = poss_radii_sol.to(u.earthRad)
        models[filename]['mod_per'] = poss_periods_yr.to(u.d)
        models[filename]['models'] = file_mods 

    return models


"""
TRANSIT CHECK
"""

def transit_check(time_arr, flux_arr, ind, trend, injection=False, inj_flux=False, 
                    radius=0.08, window=50, period=1., sm_axis=2., orb_inc=87.0, ecc=0., 
                    long_per=90., ab=[0.1, 0.3], limb_dark="quadratic"):

    """
    This function takes a certain index of the given time array and selects a window around that index. It then calls the 
    transit_model function to create a test model from the given parameters. This transit model is added to the trend of the
    light curve calculated via the LOWESS method and calculates the chi-square value between that
    test transit and the light curve within the selected window of time.
    """

    t0 = time_arr[ind] + 0.0
    fl_mod_coeff0 = transit_model(np.array(time_arr), t0, period=period, 
                                    radius=radius, sm_axis=sm_axis, 
                                    orb_inc=orb_inc, ecc=ecc, 
                                    long_per=long_per, ab=ab, 
                                    limb_dark=limb_dark)
    
    test_mod_co = np.zeros(len(time_arr))

    for i in range(ind-window, ind+window):
        if i < (len(fl_mod_coeff0)):
            test_mod_co[i] = fl_mod_coeff0[i]  


    test_mod = trend + test_mod_co

    test_mod = test_mod[~np.isnan(test_mod)]
    flux_arr = flux_arr[~np.isnan(test_mod)]
    time_arr = time_arr[~np.isnan(test_mod)]
    

    chi2_0 = np.square(flux_arr[window:-window] - test_mod[window:-window]) / test_mod[window:-window]
     
    chi2_0_clean = chi2_0[~np.isnan(chi2_0)]

    time_arr_c = time_arr[window:-window][~np.isnan(chi2_0)]

    chi2 = np.sum(chi2_0_clean) / len(chi2_0_clean)

    return chi2, test_mod
    

def lowess_iron(time, flux, window_len, break_tol, detrend_method):

    """
    This function scrolls through the entire light curve and models it using the LOWESS algorithm contained in the
    wotan Python package.
    """
    #window_len=0.375 0.2
    
    flat_flux, trend = flatten(
        time,
        flux,
        method=detrend_method,
        window_length=window_len,
        break_tolerance=break_tol,
        return_trend=True,
    )
    
    return flat_flux, trend


def lc_chi2_output(time, flux, star_rad, injection=False, file_mods=None,
                    window_length=0.2, break_tolerance=0.5, detrend_method='lowess'):
    
    """
    Inputs: time and flux arrays, test radius; if injecting, must also be provided a dictionary of file-specific 
            models to add to the flux array.
    Outputs: an array of chi2 values along with the trend of the flux array; if injecting, also outputs the 
            flux arrays corresponding to each injected model and their trends.
    """
    flat_flux, trend = lowess_iron(time, flux, window_len = window_length, 
                                    break_tol=break_tolerance, 
                                    detrend_method=detrend_method)
    test1_e = [5.]*u.earthRad
    test1_s = test1_e.to(u.solRad)
    test1 = test1_s / star_rad

    test2_e = [10.]*u.earthRad
    test2_s = test2_e.to(u.solRad)
    test2 = test2_s / star_rad


    if injection == True:
        inj_fluxes = [i + flux for i in file_mods]

        chi2_array = []
        inj_trends=[]
        test_mods = []

        for m in tqdm_notebook(range(len(file_mods))):   

            inj_flat_flux, inj_trend = lowess_iron(time, inj_fluxes[m], window_len = window_length, 
                                            break_tol=break_tolerance, detrend_method=detrend_method)
            inj_trends.append(inj_trend)

            chi2_array_single = np.zeros((2, len(time)))
            trend_mods = []

            for t in range(len(time)):
                chi2_5, trend_mod_5 = transit_check(time, inj_fluxes[m], t, inj_trend, radius=test1)
                chi2_10, trend_mod_10 = transit_check(time, inj_fluxes[m], t, inj_trend, radius =test2)
                chi2_array_single[0][t] = chi2_5
                chi2_array_single[1][t] = chi2_10
                trend_mods.append([trend_mod_5, trend_mod_10])

            chi2_array.append(chi2_array_single)
            test_mods.append(trend_mods)

        return chi2_array, trend, inj_fluxes, inj_trends


    else:

        chi2_array = np.zeros(len(time))

        for t in range(len(time)):
            chi2_5, trend_mod_5 = transit_check(time, flux, t, trend, radius=[5.]*u.earthRad)
            chi2_10, trend_mod_10 = transit_check(time, flux, t, trend, radius =[10.]*u.earthRad)
            chi2_array[i]=[chi2_5, chi2_10]
            
        return chi2_array, [trend_mod_5, trend_mod_10]
        

def calculate_planet_period(file_data, test_radius, MODELS=None, cnn=None, 
                            injection=False, file_mod_dict=False, detrend_method='lowess'):

    """
    Inputs: data table with the filename, tic, time, flux, flux_error, limb-darkening coefficients, and stellar radius
            for each SPOC file; a catalog of all said SPOC files, particularly including stellar periods; the test radius 
            for the sliding window transit check function; and the stella models and cnn.
            
            IF INJECTING: also takes a dictionary of models specific to the file at hand.

    Outputs: mProt.results, chi2_array, associated trends, ls_period, and ls_power
    """

    tic = file_data['tic']
    time = file_data['time']
    flux = file_data['flux']
    star_period = file_data['star_period']
    star_rad = [file_data['star_rad']]*u.solRad
   
    if star_period > 2.:
        minp = star_period + 0.1
    else:
        minp = 2.


    if injection == True:
        file_models = file_mod_dict['models']
        chi2_array, trend, inj_fluxes, inj_trends = lc_chi2_output(time, flux, star_rad = star_rad, 
                                                                   injection=True, file_mods=file_models, 
                                                                   detrend_method=detrend_method)

        results05     = []
        results010    = []
        periods5  = []
        powers5       = []
        periods10 = []
        powers10      = []
        for m in range(len(file_models)):
            frequency5, power5 = LombScargle(time, chi2_array[m][0]).autopower(minimum_frequency=1/9., 
                                                                            maximum_frequency=1./2)
            ls_period5 = [1./i for i in frequency5]

            frequency10, power10 = LombScargle(time, chi2_array[m][1]).autopower(minimum_frequency=1/9., 
                                                                                maximum_frequency=1./2)
            ls_period10 = [1./i for i in frequency10]


            placeholder_error = np.ones(len(chi2_array))

            mProt5 = stella_rewrite.MeasureProt([tic], [time], [chi2_array[m][0]], [placeholder_error])
            mProt5.run_LS_re()

            mProt10 = stella_rewrite.MeasureProt([tic], [time], [chi2_array[m][1]], [placeholder_error])
            mProt10.run_LS_re()

            results05.append(mProt5.LS_results)
            results010.append(mProt10.LS_results)
            periods5.append(ls_period5)
            powers5.append(power5)
            periods10.append(ls_period10)
            powers10.append(power10)

        results5 = hstack(results05)
        results10 = hstack(results010)

        return chi2_array, inj_trends, results5, results10, periods5, periods10, powers5, powers10

    elif injection == False:
        chi2_array, trend = lc_chi2_output(time, flux, test_radius=test_radius)

        frequency, power = LombScargle(time, chi2_array).autopower(minimum_frequency=1/9., 
                                                                    maximum_frequency=1/2.)
        ls_period = [1./i for i in frequency]

        placeholder_error = np.ones(len(chi2_array))

        mProt = stella_rewrite.MeasureProt([tic], [time_arr], [chi2_array], [placeholder_error])

        mProt.run_LS_re() 

        return mProt.LS_results, chi2_array, trend, ls_period, power


def find_the_munchkins(data_table, test_radius, detrend_method, injection=False, model_dict=None):

    """
    Inputs: data table, test_radius, and model dictionary if required
    Outputs: a dictionary of results along with the chi2, trend, time and flux arrays; and a dictionary
            of supplemental data, including the models, model_settings, etc for each file.
    """

    MODELS, cnn = init_stella()

    full_res = {}
    supp_data = {}
    if injection == True:
        
        for i in tqdm_notebook(range(len(data_table))):
            file_data = data_table[i]
            tic = file_data['tic']
            filename = file_data['filename']
            file_mod_dict = model_dict[filename]


            chi2_array, inj_trends, results5, results10, periods5, \
            periods10, powers5, powers10 = calculate_planet_period(file_data, test_radius=test_radius,
                                                                    MODELS=MODELS, cnn=cnn, 
                                                                    injection = injection,
                                                                    file_mod_dict=file_mod_dict,
                                                                    detrend_method=detrend_method)

            full_res[file_data['filename']] = {}
            full_res[file_data['filename']]['tic'] = tic
            full_res[file_data['filename']]['chi2'] = chi2_array
            full_res[file_data['filename']]['inj_trend'] = inj_trends
            full_res[file_data['filename']]['results_5'] = results5
            full_res[file_data['filename']]['results_10'] = results10
            
            model_dict[file_data['filename']]['periods5'] = periods5
            model_dict[file_data['filename']]['periods10'] = periods10
            model_dict[file_data['filename']]['powers5'] = powers5
            model_dict[file_data['filename']]['powers10'] = powers10
    

    else:

        for i in tqdm_notebook(range(len(files))):
            file_data = data_table[i]
            res, chi2_array, trend, ls_period, power = calculate_planet_period(file_data, test_radius=test_radius,
                                                                                MODELS=MODELS, cnn=cnn, 
                                                                                detrend_method=detrend_method)
            full_res[file_data['filename']] = {}
            full_res[file_data['filename']]['tic'] = tic
            full_res[file_data['filename']]['results'] = res
            full_res[file_data['filename']]['chi2'] = chi2_array
            full_res[file_data['filename']]['trend'] = trend

            model_dict[file_data['filename']]['ls_power'] = power
            model_dict[file_data['filename']]['ls_period'] = ls_period

    return full_res, model_dict



#tic, models, mod_settings, results
#tic, time, flux, power, period

