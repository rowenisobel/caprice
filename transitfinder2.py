import lightkurve as lk
from lightkurve import search_targetpixelfile
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

import stella_rewrite as stre
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

__all__ = ['tess_reduction', 'remove_flares_tess', 'init_stella', 'stitch_observations', 'init_tess_processing', 'model_params_randomizer', \
            'transit_model', 'create_models', 'transit_check', 'flux_iron', 'lc_chi2_output', 'calculate_planet_period', \
            'find_the_munchkins']

def tess_reduction(path, filename, pad):

    """
    Inputs
    -------- 
    path :  string
            the filepath for the SPOC files to be reduced

    filename :  string
                the name of the file to be reduced
            

    Outputs
    ---------
    time_f :    np.ndarray
                an array of the cleaned times

    flux_f :    np.ndarray
                an array of the cleaned flux

    flux_err_f :    np.ndarray
                    an array of the cleaned flux errors

    ab :    tuple
            quadratic limb-darkening coefficients for the star

    radius :    np.float64
                stellar radius in Solar radii
    """

    hdu = fits.open(os.path.join(path, filename))
    
    ID = hdu[0].header['TICID']
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=ID)   
        # Accesses TESS catalog data via transitleastsquares
    ab = list(ab)


    """
    Here we identify and clean the time and flux data contained in the FITS file input.
    """

    time = hdu[1].data['TIME']
    flux = hdu[1].data['PDCSAP_FLUX']
    flux_error = hdu[1].data['PDCSAP_FLUX_ERR']
    time, flux, flux_error = cleaned_array(time, flux, flux_error)  # removes invalid values such as nan, inf, non, negative
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


def remove_flares_tess(tics, time_arr, flux_arr, flux_error_arr, stella_models, cnn):

    """
    Inputs
    -------- 
    tics :  np.ndarray
            a list of TIC IDs to be reduced

    time_arr :  np.ndarray
                the time data arrays of all files to be reduced

    flux_arr :  np.ndarray
                the flux arrays of all files to be reduced

    flux_err_arr :  np.ndarray
                    the flux error arrays of all files to be reduced

    stella_models : np.ndarray
                    the models used by stella to train on flare finding

    cnn :   

            

    Outputs
    ---------
    time_arr :  np.ndarray
                time arrays, stripped of all flares

    flux_arr :  np.ndarray
                flux arrays, stripped of all flares

    flux_error_arr :    np.ndarray
                        flux error arrays, stripped of all flares

    flare_table :   astropy.table
                    a table containing data on all flares found by stella in all
                    lightcurves being analysed

    """

    preds = []
    for i, model in tqdm_notebook(enumerate(stella_models)):
        cnn.predict(modelname=model,
                   times=time_arr,  
                   fluxes=flux_arr,
                   errs=flux_error_arr)

        preds.append(cnn.predictions)

    ff = stella.FitFlares(id=[tics],
                      time=[time_arr],
                      flux=[flux_arr],
                      flux_err=[flux_error_arr],
                      predictions=[preds[0]])
    #ff.identify_flare_peaks(threshold=0.5)
    #flare_table = ff.flare_table
       
    for t in tqdm_notebook(range(len(time_arr))):
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
    Inputs
    -------- 
            

    Outputs
    ---------
    MODELS :    np.ndarray
                the models used by stella to train on flare finding

    cnn :   stella.neural_network.ConvNN
            the convolutional neural network used by stella to find flares
    """

    ds = stella.DownloadSets()
    ds.download_models()
    MODELS = ds.models
    OUT_DIR = '/Users/rowenglusman/Summer2022/AU_Mic/'
    cnn = stella.ConvNN(output_dir=OUT_DIR)

    return MODELS, cnn

def stitch_observations(data_table, files):

    """
    Inputs
    -------- 
    data_table :    astropy.table
                    a table containing the cleaned and de-flared time, flux,
                    and flux_err arrays of each star file, alongside the filenames,
                    TIC IDs, limb-darkening coefficients, and stellar radii

    files : np.ndarray
            a list of all star files to be reduced        

    Outputs
    ---------
    data_table_concat : np.ndarray
                        a table containing the cleaned and de-flared time, flux,
                        and flux_err arrays of each star (all files concatenated), 
                        alongside the filenames, TIC IDs, limb-darkening coefficients, 
                        and stellar radii

    times_fin : np.ndarray
                the cleaned and de-flared time arrays of all stars

    fluxes_fin :    np.ndarray
                    the cleaned and de-flared flux arrays of all stars

    flux_errs_fin : np.ndarray
                    the cleaned and de-flared flux error arrays of all stars
    """

    data_table.sort('tic')
    tics = np.array(data_table['tic'])

    unique_tics, unique_inds, unique_counts = np.unique(tics, return_index=True, return_counts=True)
    tics_fin = []
    files_fin = []
    ab_fin = []
    star_rad_fin = []
    star_per_fin = []
    lcs_fin = []
    times_fin = []
    fluxes_fin =[]
    flux_errs_fin = []

    for f in range(len(unique_tics)):
        tic_inds = np.arange(unique_inds[f], unique_inds[f] + unique_counts[f], 1)
        files = data_table[unique_inds[f]:unique_inds[f] + unique_counts[f]]

        lcs_per_tic = lk.LightCurveCollection(files['lc']).stitch()


        tics_fin.append(files['tic'][0])
        files_fin.append(np.array(files['filename']))
        ab_fin.append(files['ab'][0])
        star_rad_fin.append(files['star_rad'][0])
        star_per_fin.append(files['star_period'][0])
        lcs_fin.append(lcs_per_tic)
        times_fin.append(lcs_per_tic.time.value)
        fluxes_fin.append(lcs_per_tic.flux.value)
        flux_errs_fin.append(lcs_per_tic.flux_err.value)


    data_table_concat = Table([tics_fin, files_fin, ab_fin, star_rad_fin, star_per_fin, lcs_fin], 
                              names=('tic', 'filename', 'ab', 'star_rad', 'star_period', 'lc'))

    return data_table_concat, times_fin, fluxes_fin, flux_errs_fin




def init_tess_processing(path, catalog, concat=False, pad=125):
    """
    Inputs
    -------- 
    path :  string
            filepath to SPOC data

    catalog :   astropy.table
                a table containing data on all stars to be analysed

    concat :    bool
                a boolean indicating whether light curves originating from the same stars
                should be concatenated into single renormalized light curves

    pad :   np.int64
            the number of data points to remove before or after the end 
            or beginning of an orbit


    Outputs
    ---------
    data_table :    astrpy.table
                    a table containing the cleaned and de-flared time, flux,
                    and flux_err arrays of each star file, alongside the filenames,
                    TIC IDs, limb-darkening coefficients, and stellar radii

    times_fin : np.ndarray
                the cleaned and de-flared time arrays of all star files

    fluxes_fin :    np.ndarray
                    the cleaned and de-flared flux arrays of all star files

    flux_errs_fin : np.ndarray
                    the cleaned and de-flared flux error arrays of all star files

    flare_table :   astropy.Table
                    a table containing data on all flares found by stella in all
                    lightcurves being analysed
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

    times_s, fluxes_s, flux_errs_s = remove_flares_tess(tics, times_c, fluxes_c, flux_errs_c, MODELS, cnn)

    times = []
    fluxes = []
    flux_errs =[]

    lcs = []

    print('Outlier removal in progress...')
    for k in tqdm_notebook(range(len(times_s))):
        temp_lc = LightCurve(time=times_s[k], flux=fluxes_s[k], flux_err=flux_errs_s[k]).remove_outliers()

        lcs.append(temp_lc)
        times.append(temp_lc.time.value)
        fluxes.append(temp_lc.flux.value)
        flux_errs.append(temp_lc.flux_err.value)

    res = Table([filenames, tics, ab_arr, radius_arr, lcs], \
                names=('filename', 'tic', 'ab', 'star_rad', 'lc'))
    tic_per_cat = Table([catalog['tic'], catalog['period']], names=('tic', 'star_period'))
    
    data_table = join(res, tic_per_cat, keys='tic', join_type='left')

    if concat == True:

        return stitch_observations(data_table, files)

    elif concat == False:

        times_fin = [list(i) for i in times]
        fluxes_fin = [list(i) for i in fluxes]
        flux_errs_fin = [list(i) for i in flux_errs]
    
        return data_table, times_fin, fluxes_fin, flux_errs_fin


"""
INJECTION MODELS
"""

def model_params_randomizer(model_number, per_lims, rad_lims):

    """
    Inputs
    -------- 
    model_number :  np.int
                    the number of models per star file

    per_lims :  tuple
                the maximum and minimum periods (in days) at which to generate models

    rad_lims :  tuple
                the maximum and minimum radii (in Earth radii) at which to generate models

    
    Outputs
    ---------
    mod_settings :  dictionary
                    a dictionary encoding the array of randomly generated periods and radii
                    pertaining to each model within the given bounds
    
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
    Inputs
    -------- 
    time_arr :  np.ndarray
                the time array pertaining to one star file

    t0 :    np.int
            a random index at which to begin producing the transits in the time array

    period :    np.float64
                the period (in days) at which to generate the transit model

    radius :    np.float64
                the radius (in Earth radii) at which to generate the transit model

    sm_axis :   np.float64
                the semi-mejor axis at which to generate the transit model

    ab :    tuple
            the quadratic limb-darkening coefficients for the star file in question

    orb_inc :   np.float64
                the orbital inclination at which to generate the transit model

    ecc :   np.float64
            the eccentricity at which to generate the tranit model

    long_per :  np.float64
                the longitude of periastron (in degrees)

    limb_dark : string
                the regime in which to calculate the limb-darkening of the star file


    Outputs
    ---------
    
    flux_co :   np.ndarray
                the transit model to add to the associated flux array of the star
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

def create_models(data_table, times, fluxes, model_number, per_lims, rad_lims):

    """
    Inputs
    -------- 
    data_table :    astrpoy.table
                    a table containing the cleaned and de-flared time, flux,
                    and flux_err arrays of each star file, alongside the filenames,
                    TIC IDs, limb-darkening coefficients, and stellar radii

    times : np.ndarray
            the cleaned and de-flared time arrays of all star files

    fluxes :    np.ndarray
                the cleaned and de-flared flux arrays of all star files

    model_number:   np.int
                    the number of models per star file to be injected and recovered

    per_lims :  tuple
                the maximum and minimum periods (in days) at which to generate models

    rad_lims :  tuple
                the maximum and minimum radii (in Earth radii) at which to generate models

    
    Outputs
    ---------
    models :    astropy.table
                a table of all the models (n models per star file) to be injected and recovered,
                in addition to their respective metadata
    """


    models = {}

    for i in tqdm_notebook(range(len(data_table))):
        mod_settings = model_params_randomizer(model_number, per_lims, rad_lims)
        file_data = data_table[i]
        filename = file_data['filename']
        tic = file_data['tic']
        time = np.array(times[i])
        flux = np.array(fluxes[i])
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
    Inputs
    -------- 
    time_arr :  np.ndarray
                the time array pertaining to one star file

    flux_arr :  np.ndarray
                the flux array pertaining to one star file

    ind :   np.int
            a given index at which to begin producing the transits in the time array

    trend : np.ndarray
            the trend of the data as calculated in the flux_iron function

    injection : bool
                the value, True or False, of whether this run of the program is meant to
                calculate injection-recovery statistics or analyse real data

    inj_flux :  np.ndarray
                the flux array of the star, injected with a transit model

    radius :    np.float64
                the radius (in Earth radii) at which to generate the comparator model

    window :    np.int
                the number of data points pertaining to the width of the sliding window

    period :    np.float64
                the period in days at which to generate the comparator model. Since the 
                comparator model is only generated for a single transit, this is an unimportant

    sm_axis :   np.float64
                the semi-mejor axis at which to generate the comparator model


    orb_inc :   np.float64
                the orbital inclination at which to generate the comparator model

    ecc :   np.float64
            the eccentricity at which to generate the comparator model

    ab :    tuple
            default quadratic limb-darkening coefficients for the comparator model

    long_per :  np.float64
                the longitude of periastron (in degrees) of the comparator model

    limb_dark : string
                the regime in which to calculate the limb-darkening of the comparator model


    Outputs
    ---------
    
    chi2 :  np.ndarray
            the array of chi2 values representing the difference between the injected
            flux and the comparator model at the central index of the sliding window for
            each step

    test_mod :  np.nd_array
                the flux of the comparator model being used in the sliding window chi2 
                calculation
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
    

def flux_iron(time, flux, window_len, break_tol, detrend_method):

    """
    Inputs
    -------- 

    time : np.ndarray
            the cleaned and de-flared time array of a single star file

    flux :  np.ndarray
            the cleaned and de-flared flux array of a single star file

    window_len :    np.int
                    the length of the sliding window in indices

    break_tol : np.float64
                a direction for the detrending algorithm to split its process at breaks
                longer than this value in units of "time"

    detrend_method :    string
                        the statistical method by which to detrend
    
    Outputs
    ---------
    flat_flux : np.ndarray
                the flattened (detrended) flux array

    trend : np.ndarray
            the trend removed from the flux
    """

    
    flat_flux, trend = flatten(
        list(time),
        list(flux),
        method=detrend_method,
        window_length=window_len,
        break_tolerance=break_tol,
        return_trend=True,
    )
    
    return flat_flux, trend


def lc_chi2_output(time, flux, star_rad, injection=False, file_mods=None,
                    window_length=0.2, break_tolerance=0.5, detrend_method='lowess'):
    
    """
    Inputs
    -------- 

    time : np.ndarray
            the cleaned and de-flared time array of a single star file

    flux :  np.ndarray
            the cleaned and de-flared flux array of a single star file

    star_rad :  np.float64
                the radius of the star in question in Solar radii

    injection : bool
                the value, True or False, of whether this run of the program is meant to
                calculate injection-recovery statistics or analyse real data

    file_mods : dictionary
                a dictionary of the N models to be injected into the flux array


    window_length : np.int
                    the length of the sliding window in indices

    break_tolerance :   np.float64
                        a direction for the detrending algorithm to split its process at breaks
                        longer than this value in units of "time"

    detrend_method :    string
                        the statistical method by which to detrend
    
    Outputs
    ---------
    chi2 :  np.ndarray
            the array of chi2 values representing the difference between the injected
            flux and the comparator model at the central index of the sliding window for
            each step, for N injected models

    trend : np.ndarray
            the trends removed from the flux, for N injected models
    """

    
    flat_flux, trend = flux_iron(time, flux, window_len = window_length, 
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
        inj_trends = []
        test_mods = []
        for m in tqdm_notebook(range(len(file_mods))):   

            inj_flat_flux, inj_trend = flux_iron(time, inj_fluxes[m], window_len = window_length, 
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

        chi2_array = np.zeros((2, len(time)))

        flat_flux, trend = flux_iron(time, flux, window_len = window_length, 
                                            break_tol=break_tolerance, detrend_method=detrend_method)

        for t in range(len(time)):
            chi2_5, trend_mod_5 = transit_check(time, flux, t, trend, radius=test1)
            chi2_10, trand_mod_10 = transit_check(time, flux, t, trend, radius =test2)
            chi2_array[0][t] += chi2_5 
            chi2_array[1][t] += chi2_10
            
        return chi2_array, trend
        

def calculate_planet_period(file_data, time, flux, flux_err, MODELS=None, cnn=None, 
                            injection=False, file_mod_dict=False, detrend_method='lowess', chi2_bound=10**(-5),
                            rat_bound=5.):

    """
    Inputs
    -------- 

    file_data : astropy.table
                a table containing the metadata for the file being analysed

    time : np.ndarray
            the cleaned and de-flared time array of a single star file

    flux :  np.ndarray
            the cleaned and de-flared flux array of a single star file

    flux_err :  np.ndarray
                the cleaned and de-flared flux error array of a single star file

    MODELS :    np.ndarray
                the models used by stella to train on flare finding

    cnn:    



    injection : bool
                the value, True or False, of whether this run of the program is meant to
                calculate injection-recovery statistics or analyse real data

    file_mod_dict : dictionary
                    a dictionary of the N models to be injected into the flux array

    detrend_method :    string
                        the statistical method by which to detrend

    chi2_bound :    np.float64
                    the chi2 value at which to remove unreasonably deviant data

    rat_bound : np.float64
                the LS periodogram peak FWHM bound at which to remove unreasonably
                deviant data
    
    Outputs
    ---------
    chi2_array :    np.ndarray
                    the array of chi2 values representing the difference between the injected
                    flux and the comparator model at the central index of the sliding window for
                    each step, for N injected models per star

    inj_trend :     np.ndarray
                    the trends removed from the fluxes plus the injected transits, for N models

    results5 :  astropy.table
                the results from the LS periodogram run by stella on each star for a comparator
                model of 5 Earth radii

    results10 : astropy.table
                the results from the LS periodogram run by stella on each star for a comparator
                model of 10 Earth radii 

    periods5 :  np.ndarray
                the period space array(s) for each star with a comparator model of 5 Earth radii

    periods10 : np.ndarray
                the period space array(s) for each star with a comparator model of 10 Earth radii

    powers5 :   np.ndarray
                the power space array(s) for each star with a comparator model of 5 Earth radii 

    powers10 :  np.ndarray
                the power space array(s) for each star with a comparator model of 10 Earth radii 

    """


    tic = file_data['tic']
    star_period = file_data['star_period']
    star_rad = [file_data['star_rad']]*u.solRad
   
    minp = 0.5


    if injection == True:
        file_models = file_mod_dict['models']
        chi2_array, trend, inj_fluxes, inj_trends = lc_chi2_output(np.array(time), np.array(flux), star_rad = star_rad, 
                                                                   injection=True, file_mods=file_models, 
                                                                   detrend_method=detrend_method)

        results05     = []
        results010    = []
        periods5      = []
        powers5       = []
        periods10     = []
        powers10      = []
        for m in range(len(file_models)):
            frequency5, power5 = LombScargle(time, chi2_array[m][0]).autopower(minimum_frequency=1/12.5, 
                                                                            maximum_frequency=1., 
                                                                            samples_per_peak=50)
            ls_period5 = [1./i for i in frequency5]

            frequency10, power10 = LombScargle(time, chi2_array[m][1]).autopower(minimum_frequency=1/12.5, 
                                                                                maximum_frequency=1.,
                                                                                samples_per_peak=50)
            ls_period10 = [1./i for i in frequency10]


            placeholder_error = np.ones(len(chi2_array))

            mProt5 = test.MeasureProt([tic], [time], [chi2_array[m][0]], [placeholder_error])
            mProt5.run_LS(star_period = star_period, minf=1./12.5, maxf=1., chi2_bound=chi2_bound, rat_bound=rat_bound)
                                    
            mProt10 = test.MeasureProt([tic], [time], [chi2_array[m][1]], [placeholder_error])
            mProt10.run_LS(star_period = star_period, minf=1./12.5, maxf=1., chi2_bound=chi2_bound, rat_bound=rat_bound)


            results05.append(mProt5.LS_results)
            results010.append(mProt10.LS_results)
            periods5.append(ls_period5)
            powers5.append(power5)
            periods10.append(ls_period10)
            powers10.append(power10)

        results5 = vstack(np.array(results05))
        results10 = vstack(np.array(results010))

        return chi2_array, inj_trends, results5, results10, periods5, periods10, powers5, powers10

    elif injection == False:
        results05     = []
        results010    = []
        periods5      = []
        powers5       = []
        periods10     = []
        powers10      = []

        chi2_array, trend = lc_chi2_output(np.array(time), np.array(flux), star_rad = star_rad, detrend_method=detrend_method)

        frequency5, power5 = LombScargle(time, chi2_array[0]).autopower(minimum_frequency=1/12.5, 
                                                                            maximum_frequency=1., 
                                                                            samples_per_peak=50)
        ls_period5 = [1./i for i in frequency5]

        frequency10, power10 = LombScargle(time, chi2_array[1]).autopower(minimum_frequency=1/12.5, 
                                                                            maximum_frequency=1.,
                                                                            samples_per_peak=50)
        ls_period10 = [1./i for i in frequency10]


        placeholder_error = np.ones(len(chi2_array))

        mProt5 = test.MeasureProt([tic], [time], [chi2_array[0]], [placeholder_error])
        mProt5.run_LS(star_period = star_period, minf=1./12.5, maxf=1., chi2_bound=chi2_bound, rat_bound=rat_bound)
                                
        mProt10 = test.MeasureProt([tic], [time], [chi2_array[1]], [placeholder_error])
        mProt10.run_LS(star_period = star_period, minf=1./12.5, maxf=1., chi2_bound=chi2_bound, rat_bound=rat_bound)


        results05.append(mProt5.LS_results)
        results010.append(mProt10.LS_results)
        periods5.append(ls_period5)
        powers5.append(power5)
        periods10.append(ls_period10)
        powers10.append(power10)

        results5 = hstack(results05)
        results10 = hstack(results010)

        return chi2_array, trend, results5, results10, periods5, periods10, powers5, powers10


def find_the_munchkins(data_table, times, fluxes, flux_errs, detrend_method, injection=False, model_dict=None, 
                       chi2_bound=10**(-5), rat_bound=5.):

    """
    Inputs
    -------- 

    data_table :    astropy.table
                    a table containing the metadata for all star files being analysed

    times : np.ndarray
            the cleaned and de-flared time arrays of all star files

    fluxes :    np.ndarray
                the cleaned and de-flared flux arrays of all star files

    flux_errs : np.ndarray
                the cleaned and de-flared flux error arrays of all star files


    detrend_method :    string
                        the statistical method by which to detrend star fluxes


    injection : bool
                the value, True or False, of whether this run of the program is meant to
                calculate injection-recovery statistics or analyse real data

    model_dict :    dictionary
                    a dictionary of the N models to be injected into the flux array


    chi2_bound :    np.float64
                    the chi2 value at which to remove unreasonably deviant data

    rat_bound : np.float64
                the LS periodogram peak FWHM bound at which to remove unreasonably
                deviant data
    
    Outputs
    ---------
    full_res :  astropy.table
                a table containing all the results for the data run

    supp_data : astropy.table
                a table containing the period and power space data of the LS periodograms for
                each star file, and injected model, if applicable

    """


    MODELS, cnn = init_stella()

    full_res = {}
    supp_data = {}
    if injection == True:
        
        for i in tqdm_notebook(range(len(data_table))):
            file_data = data_table[i]
            time = times[i]
            flux = fluxes[i]
            flux_err = flux_errs[i]
            tic = file_data['tic']
            filename = file_data['filename']
            
            file_mod_dict = model_dict[filename]


            chi2_array, inj_trends, results5, results10, periods5, \
            periods10, powers5, powers10 = calculate_planet_period(file_data, time, flux, flux_err,
                                                                    MODELS=MODELS, cnn=cnn, 
                                                                    injection = injection,
                                                                    file_mod_dict=file_mod_dict,
                                                                    detrend_method=detrend_method,
                                                                    chi2_bound=chi2_bound,
                                                                    rat_bound=rat_bound)

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

        for i in tqdm_notebook(range(len(data_table))):
            file_data = data_table[i]
            time = times[i]
            flux = fluxes[i]
            flux_err = flux_errs[i]
            tic = file_data['tic']
            filename = file_data['filename']

            chi2_array, trend, results5, results10, periods5, \
            periods10, powers5, powers10 = calculate_planet_period(file_data, time, flux, flux_err,
                                                                                MODELS=MODELS, cnn=cnn, 
                                                                                detrend_method=detrend_method)
            full_res[filename] = {}
            full_res[filename]['tic'] = tic
            full_res[filename]['chi2'] = chi2_array
            full_res[filename]['trend'] = trend
            full_res[filename]['results_5'] = results5
            full_res[filename]['results_10'] = results10

            supp_data[filename] = {}
            supp_data[filename]['periods5'] = periods5
            supp_data[filename]['periods10'] = periods10
            supp_data[filename]['powers5'] = powers5
            supp_data[filename]['powers10'] = powers10

    return full_res, supp_data

    



#tic, models, mod_settings, results
#tic, time, flux, power, period

