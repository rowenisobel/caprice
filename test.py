import numpy as np
import matplotlib.pyplot as plt; plt.rcParams["figure.dpi"] = 250
from tqdm import tqdm
import statistics as stats
from astropy import units as u
import scipy
from scipy.signal import medfilt
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
from astropy.table import Table, Column
from astropy.timeseries import LombScargle
from numpy import exp, linspace, random
from scipy.signal import argrelextrema

__all__ = ['MeasureProt']

class MeasureProt(object):
	"""
	Used for measuring rotation periods.
	"""
	
	def __init__(self, IDs, time, flux, flux_err):
		"""
		Takes in light curve identifiers, time, flux, 
		and flux errors.
		"""
		self.IDs  = IDs
		self.time = time
		self.flux = flux
		self.flux_err = flux_err



	def gauss_curve(self, x, std, scale, mu):
		""" Fits a Gaussian to the peak of the LS
			periodogram.
		Parameters
		----------
		x : np.array
		std : float
			 Standard deviation of gaussian.
		scale : float
			 Scaling for gaussian.
		mu : float
			 Mean to fit around.
		Returns
		-------
		Gaussian curve.
		"""
		term1 = 1.0 / (std * np.sqrt(2 * np.pi) )
		term2 = np.exp(-0.5 * ((x-mu)/std)**2)
		return term1 * term2 * scale


	def chiSquare(self, var, mu, x, y, yerr):
		""" Calculates chi-square for fitting a Gaussian
			to the peak of the LS periodogram.
		Parameters
		----------
		var : list
			 Variables to fit (std and scale for Gaussian curve).
		mu : float
			 Mean to fit around.
		x : np.array
		y : np.array
		yerr : np.array
		Returns
		-------
		chi-square value.
		"""
		m = self.gauss(x, var[0], var[1], mu)
		return np.sum( (y-m)**2 / yerr**2 )

	
	def fit_LS_peak(self, period, power, arg):
		""" Fits the LS periodogram at the peak power. 
		Parameters
		----------
		period : np.array
			 Array of periods from Lomb Scargle routine.
		power : np.array
			 Array of powers from the Lomb Scargle routine.
		arg : int
			 Argmax of the power in the periodogram.
		Returns
		-------
		popt : np.array
			 Array of best fit values for Gaussian fit.
		"""
		def fitting_routine():
			popt, pcov = curve_fit(self.gauss_curve, period[m], power[m],
								   p0 = [(np.nanmax(period[subm]) - np.nanmin(period[subm]))/2.0,
										 0.02,
										 period[arg]],
								   maxfev = 10000)
			return popt

		if arg-40 < 0:
			start = 0
		else:
			start = arg-40
		if arg+40 > len(period):
			end = len(period)-1
		else:
			end = arg+40

		m = np.arange(start, end, 1, dtype=int)

		if arg-20 < 0:
			start = 0
		else:
			start = arg-20
		if arg + 20 > len(period):
			end = len(period)-1
		else:
			end = arg+20

		subm = np.arange(start, end, 1, dtype=int)

		try:
			popt = fitting_routine()
		except RuntimeError:
			popt = np.full(3, np.nan)

		# TRIES TO READJUST FITTING WINDOW IF RANGE IS LARGER THAN PERIOD ARRAY
		except IndexError:
			if np.min(m) <= 0:
				m = np.arange(0,arg+40,1,dtype=int)
				subm = np.arange(0,arg+20,1, dtype=int)
			elif np.max(m) > len(period):
				diff = np.max(m) - len(period)
				m = np.arange(arg-40-diff, len(period)-diff, 1, dtype=int)
				subm = np.arange(arg-20-diff, len(period)-diff-20, 1, dtype=int)

			popt = fitting_routine()


		return popt

	
	def run_LS(self, star_period, minf, maxf, spp=50, chi2_bound=10**(-5), rat_bound=5.):
		""" Runs LS fit for each light curve. 
		Parameters
		----------
		minf : float, optional
			 The minimum frequency to search in the LS routine. Default = 1/20.
		maxf : float, optional
			 The maximum frequency to search in the LS routine. Default = 1/0.1.
		spp : int, optional
			 The number of samples per peak. Default = 50.
		Attributes
		----------
		LS_results : astropy.table.Table
		"""
		def data_fit(p0, func, xvar, yvar, err, cen, bounds, tmi=0):

			try:
				fit = least_squares(residual, p0, args=(func, xvar, yvar, err, cen), verbose=tmi, max_nfev=10000000)
			except Exception as error:
				print("Something has gone wrong:",error)
				return p0, np.zeros_like(p0), np.nan, np.nan

			pf = fit['x']

			try:
				cov = np.linalg.inv(fit['jac'].T.dot(fit['jac']))
			except:
				print('Fit did not converge')
				print('Result is likely a local minimum')
				print('Try changing initial values')
				print('Status code:', fit['status'])
				print(fit['message'])
				return pf, np.zeros_like(pf), np.nan, np.nan

			chisq = sum(residual(pf, func, xvar, yvar, err, cen) **2)
			#print((residual(pf, func, xvar, yvar, err, cen))**2)
			dof = len(xvar) - len(pf)
			red_chisq = chisq/dof
			pferr = np.sqrt(np.diagonal(cov))
			return pf, pferr, red_chisq, dof

		def residual(p, func, xvar, yvar, err, cen):
			err[err == 0] = 0.01
			return (func(p, cen, xvar) - yvar)/err

		def gaussian_lin(p, cen, x):
			return (p[0]*np.exp(-(x-cen)**2/(p[1]**2))) + p[2]*x + p[3]

		def two_gauss_lin(p, cen, x):
			return np.abs((p[0]*np.exp(-(x-cen)**2/(p[1]**2))) + p[2]*x + p[3] + \
					(p[4]*np.exp(-(x-p[5])**2/(p[6]**2))) + p[7]*x + p[8])
		def three_gauss_lin(p, cen, x):
			return np.abs((p[0]*np.exp(-(x-cen)**2/(p[1]**2))) + p[2]*x + p[3] + \
					(p[4]*np.exp(-(x-p[5])**2/(p[6]**2))) + p[7]*x + p[8] + \
					(p[9]*np.exp(-(x-p[10])**2/(p[11]**2))) + p[12]*x + p[13])

		def local_mins(ind, min_array, per_array):

			loc_mins_low = min_array[min_array < ind]
			loc_mins_high = min_array[min_array > ind]

			if len(loc_mins_low) == 0:
				lml = 0
			else:
				lml = loc_mins_low[-1]

			if len(loc_mins_high) == 0:
				lmh = len(per_array) - 1
			else:
				lmh = loc_mins_high[0]

			return lml, lmh

		def maxmin(power_array):

			mx = argrelextrema(power_array, np.greater)[0]
			mn = argrelextrema(power_array, np.less)[0]
			
			if mx[0] == 0:
				mx.remove(0)
			elif mx[-1] == len(mx):
				mx.remove(len(mx))

			return mx, mn

		def clean_options(per_array, power_array, chi2_bound, rat_bound):

			mx, mn = maxmin(power_array)

			inviable_peaks = np.zeros(len(per_array))
			mx_2 = mx
			for m in mx:

				lml, lmh = local_mins(m, mn, per_array)
				rang = (m - lml) * 0.5
				rang2 = (lmh - m) * 0.5
				should = round(m + rang) 
				should2 = round(m - rang2)
				per_val_m = per_array[m]

				if should > len(power_array):
					inviable_peaks[lml:] += 1
					mx_2 = np.delete(mx_2, np.where(mx_2 == m))
					continue
					
				elif should2 < 0:
					inviable_peaks[0:lmh] = 1
					mx_2 = np.delete(mx_2, np.where(mx_2 == m))
					continue


				l = power_array[lmh]
				h = power_array[lml]
				ln = per_array[lml] - per_array[lmh]
				slope = (h - l) / ln 
				b = power_array[lmh] - slope * per_array[lmh]
				a = power_array[m]
				c = per_array[m]

				init_vals1 = [a, ln/5, slope, b]
				bounds1 = [np.inf, np.inf, np.inf, np.inf]

				try:
					per_err = np.ones(len(per_array))
					pf1, pferr1, red_chisq1, dof1 = data_fit(init_vals1, gaussian_lin, per_array[lml:lmh], 
															 power_array[lml:lmh], per_err[lml:lmh],
															 cen=c, bounds=bounds1)
					
					if red_chisq1 > chi2_bound:
						inviable_peaks[lml:lmh] += 1.
						mx_2 = np.delete(mx_2, np.where(mx_2 == m))
						
					else:
						fwhm = pf1[1] * 2 * np.sqrt((2 * np.log(2)))
						rat = fwhm / pf1[0]
						if rat < rat_bound:
							inviable_peaks[lml:lmh] += 1.

					
				except RuntimeError:
					print('Runtime error')
			
			per_array_new = per_array[inviable_peaks == 0]
			power_array_new = power_array[inviable_peaks == 0]


			return per_array_new, power_array_new

		def remove_resonances(value, per_array, power_array):

			res_list = [0.5, 1.0, 2.0, 4.0] #, 8.0]
			res_init =  np.array([value * i for i in res_list])
			res_boundary_mask = res_init >= min(per_array)
			res = res_init[res_boundary_mask]

			mx, mn = maxmin(power_array)
			remove_res = np.zeros(len(per_array))
			for m in mx:
				lml, lmh = local_mins(m, mn, per_array)
				per_val_m = per_array[m]
				mask = abs(res - per_val_m) <= 0.05
				if any(mask) == True:
					remove_res[lml:lmh] += 1.
					continue

			if power_array[0] > power_array[mn[-1]]:
				remove_res[mn[-1]:] += 1

			if power_array[-1] > power_array[mn[0]]:
				remove_res[:mn[0]] += 1
			
			rr = remove_res == 0
			return per_array[rr], power_array[rr]


		def per_orbit(t, f, star_period):
			nonlocal minf, maxf, spp, chi2_bound, rat_bound

			freq, power = LombScargle(t, f).autopower(minimum_frequency=minf,
													  maximum_frequency=maxf,
													  samples_per_peak=spp)
			per = 1/freq
			d = np.diff(freq)[0]
			sigma = np.diff(power)
			
			
			res_list = [0.5, 1.0, 2.0, 4.0, 8.0]
			star_res_init = np.array([star_period * i for i in res_list])
			star_res_boundary_mask = star_res_init >= min(per)
			star_res = star_res_init[star_res_boundary_mask]
			

			per_clean, power_clean = clean_options(per, power, chi2_bound, rat_bound)

			per_no_star, power_no_star = remove_resonances(star_period, per_clean, power_clean)

	
			diff = np.diff(power_no_star)
			pt_ind = np.where(diff > np.nanmedian(diff) + (5 * np.nanstd(diff)))
			power_no_star = np.delete(power_no_star, pt_ind[0]-1)
			per_no_star = np.delete(per_no_star, pt_ind[0]-1)

			
			arg = np.where(power_no_star == max(power_no_star))[0][0]

			popt = self.fit_LS_peak(per_no_star, power_no_star, arg)

			per_sec, power_sec = remove_resonances(per_no_star[arg], per_no_star, power_no_star)

			arg1 = np.argmax(power_sec)
			## REDOS PERIOD ROUTINE FOR SECOND HIGHEST PEAK 
			if arg1 == len(per_sec):
				arg1 = int(arg1-3)

			popt2 = self.fit_LS_peak(per_sec, power_sec, arg1)

			
			maxpower = power_no_star[arg]
			secpower = power_sec[arg1]

			bestperiod = per_no_star[arg]
			secbperiod = per_sec[arg1]

			bestwidth = popt[0]


			return bestperiod, secbperiod, maxpower, secpower, bestwidth, per_no_star, power_no_star, per, power


		tab = Table()

		periods = np.zeros(len(self.IDs))
		stds = np.zeros(len(self.IDs))
		peak_power = np.zeros(len(self.IDs))
		per = [[] for i in range(len(self.IDs))]
		power = [[] for i in range(len(self.IDs))]
		per1 = [[] for i in range(len(self.IDs))]
		power1 = [[] for i in range(len(self.IDs))]

		periods2 = np.zeros(len(self.IDs))
		peak_power2 = np.zeros(len(self.IDs))

		orbit_flag = np.zeros(len(self.IDs))
		orbit_flag1 = np.zeros(len(self.IDs))
		orbit_flag2 = np.zeros(len(self.IDs))


		for i in tqdm(range(len(self.flux)), desc="Finding most likely periods"):

			time, flux, flux_err = self.time[i], self.flux[i], self.flux_err[i]
			
			# SPLITS BY ORBIT
			diff = np.diff(time)
			brk = np.where(diff >= np.nanmedian(diff)+14*np.nanstd(diff))[0]
			
			if len(brk) > 1:
				brk_diff = brk - (len(time)/2)
				try:
					brk_diff = np.where(brk_diff<0)[0][-1]
				except IndexError:
					brk_diff = np.argmin(brk_diff)
				brk = np.array([brk[brk_diff]], dtype=int)

			# DEFINITELY TRIMS OUT EARTHSHINE MOFO
			t1, f1 = time[:brk[0]], flux[:brk[0]]#[300:-500], flux[:brk[0]]#[300:-500]
			t2, f2 = time[brk[0]:], flux[brk[0]:]#[800:-200], flux[brk[0]:]#[800:-200]

			o0_params = per_orbit(time, flux, star_period)

			

			flag1 = self.assign_flag(o0_params[0], o0_params[2], o0_params[4],
									o0_params[0], o0_params[3], time[5]-time[0])
			
				
			periods[i] = np.nanmedian([o0_params[0]]) #, o2_params[0]])           
			orbit_flag1[i] = flag1
			#orbit_flag2[i] = flag2
				
			stds[i]    = o0_params[4]
			peak_power[i] = o0_params[2]
			periods2[i] = o0_params[1]
			peak_power2[i] = o0_params[3]
			
			per[i] = [i for i in o0_params[-2]]
			power[i] = [i for i in o0_params[-1]]
			per1[i] = [i for i in o0_params[-4]]
			power1[i] = [i for i in o0_params[-3]]

		
		tab.add_column(Column(self.IDs, name='Target_ID'))
		tab.add_column(Column(periods, name='period_days'))
		tab.add_column(Column(periods2, name='secondary_period_days'))
		tab.add_column(Column(stds, name='gauss_width'))
		tab.add_column(Column(peak_power, name='max_power'))
		tab.add_column(Column(peak_power2, name='secondary_max_power'))
		tab.add_column(Column(orbit_flag, name='orbit_flag'))
		tab.add_column(Column(orbit_flag1, name='oflag1'))
		#tab.add_column(Column(orbit_flag2, name='oflag2'))
		#tab.add_column(Column(per[0], name='per_array'))
		#tab.add_column(Column(power[0], name='power_array'))
		

		tab = self.averaged_per_sector(tab)

		self.LS_results = tab


			
	def assign_flag(self, period, power, width, avg, secpow, 
					maxperiod, orbit_flag=0):
		""" Assigns a flag in the table for which periods are reliable.
		"""
		flag = 100
		if period > maxperiod:
			flag = 4
		if (period < maxperiod) and (power > 0.005):
			flag = 3
		if (period < maxperiod) and (width <= period*0.6) and (power > 0.005):
			flag = 2
		if ( (period < maxperiod) and (width <= period*0.6) and
			 (secpow < 0.96*power) and (power > 0.005)):
			flag = 1
		if ( (period < maxperiod) and (width <= period*0.6) and 
			 (secpow < 0.96*power) and (np.abs(period-avg)<1.0) and (power > 0.005)):
			flag = 0
		if flag == 100:
			flag = 5
		return flag

			
	def averaged_per_sector(self, tab):
		""" Looks at targets observed in different sectors and determines
			which period measured is likely the best period. Adds a column
			to MeasureRotations.LS_results of 'true_period_days' for the 
			results.
		Returns
		-------
		astropy.table.Table
		"""
		def flag_em(val, mode, lim):
			if np.abs(val-mode) < lim:
				return 0
			else:
				return 1

		averaged_periods = np.zeros(len(tab))
		flagging = np.zeros(len(tab), dtype=int)

		limit = 0.3

		for tic in np.unique(self.IDs):
			inds = np.where(tab['Target_ID']==tic)[0]
			primary = tab['period_days'].data[inds]
			secondary = tab['secondary_period_days'].data[inds]
			all_periods = np.append(primary, secondary)

#            ind_flags = np.append(tab['oflag1'].data[inds],
#                                  tab['oflag2'].data[inds])
			avg = np.array([])
			tflags = np.array([])

			if len(inds) > 1:
				try:
					mode = stats.mode(np.round(all_periods,2))
					if mode > 11.5:
						avg = np.full(np.nanmean(primary), len(inds))
						tflags = np.full(2, len(inds))
					else:
						for i in range(len(inds)):
							if np.abs(primary[i]-mode) < limit:
								avg = np.append(avg, primary[i])
								tflags = np.append(tflags,0)
								
							elif np.abs(secondary[i]-mode) < limit:
								avg = np.append(avg, secondary[i])
								tflags = np.append(tflags,1)
								
							elif np.abs(primary[i]/2.-mode) < limit:
								avg = np.append(avg, primary[i]/2.)
								tflags = np.append(tflags,0)

							elif np.abs(secondary[i]/2.-mode) < limit:
								avg = np.append(avg, secondary[i]/2.)
								tflags = np.append(tflags,1)
								
							elif np.abs(primary[i]*2.-mode) < limit:
								avg = np.append(avg, primary[i]*2.)
								tflags = np.append(tflags,0)
								
							elif np.abs(secondary[i]*2.-mode) < limit:
								avg = np.append(avg, secondary[i]*2.)
								tflags = np.append(tflags,1)
								
							else:
								tflags = np.append(tflags, 2)

				except:
					for i in range(len(inds)):
						if tab['oflag1'].data[inds[i]]==0 and tab['oflag2'].data[inds[i]]==0:
							avg = np.append(avg, tab['period_days'].data[inds[i]])
							tflags = np.append(tflags, 0)
						else:
							tflags = np.append(tflags,2)
							
					
			else:
				avg = np.nanmean(primary)
				if tab['oflag1'].data[inds] == 0 and tab['oflag2'].data[inds]==0:
					tflags = 0
				else:
					tflags = 2

			averaged_periods[inds] = np.nanmean(avg)
			flagging[inds] = tflags

						
		tab.add_column(Column(flagging, 'Flags'))
		tab.add_column(Column(averaged_periods, 'avg_period_days'))
		return tab


	def phase_lightcurve(self, table=None, trough=-0.5, peak=0.5, kernel_size=101):
		""" 
		Finds and creates a phase light curve that traces the spots.
		Uses only complete rotations and extrapolates outwards until the
		entire light curve is covered.
		Parameters
		----------
		table : astropy.table.Table, optional
			 Used for getting the periods of each light curve. Allows users
			 to use already created tables. Default = None. Will search for 
			 stella.FindTheSpots.LS_results.
		trough : float, optional
			 Sets the phase value at the minimum. Default = -0.5.
		peak : float, optional
			 Sets the phase value t the maximum. Default = 0.5.
		kernel_size : odd float, optional
			 Sets kernel size for median filter smoothing. Default = 15.
		Attributes
		----------
		phases : np.ndarray
		"""
		def map_per_orbit(time, flux, kernel_size, cadences):
			mf = medfilt(flux, kernel_size=kernel_size)
			argmin = np.argmin(mf[:cadences])
			mapping = np.linspace(0.5,-0.5, cadences)
			phase = np.ones(len(flux))

			full = int(np.floor(len(time)/cadences))
			
			phase[0:argmin] = mapping[len(mapping)-argmin:]
			
			points = np.arange(argmin, cadences*(full+1)+argmin, cadences, dtype=int)
			for i in range(len(points)-1):
				try:
					phase[points[i]:points[i+1]] = mapping            
				except:
					pass
			remainder = len(np.where(phase==1.0)[0])            
			phase[len(phase)-remainder:] = mapping[0:remainder]
			return phase

		if table is None:
			table = self.LS_results

		PHASES = np.copy(self.flux)

		for i in tqdm(range(len(table)), desc="Mapping phases"):
			flag = table['Flags'].data[i]
			if flag == 0 or flag == 1:
				period = table['avg_period_days'].data[i] * u.day
				cadences = int(np.round((period.to(u.min)/2).value))

				all_time = self.time[i]
				all_flux = self.flux[i]
				
				diff = np.diff(all_time)
				gaptime = np.where(diff>=np.nanmedian(diff)+12*np.nanstd(diff))[0][0]
				
				t1, f1 = all_time[:gaptime+1], all_flux[:gaptime+1]
				t2, f2 = all_time[gaptime+1:], all_flux[gaptime+1:]
				
				o1map = map_per_orbit(t1, f1, kernel_size=101, cadences=cadences)
				o2map = map_per_orbit(t2, f2, kernel_size=101, cadences=cadences)
				
				phase = np.append(o1map, o2map)

			else:
				phase = np.zeros(len(self.flux[i]))
			
			PHASES[i] = phase

		self.phases = PHASES