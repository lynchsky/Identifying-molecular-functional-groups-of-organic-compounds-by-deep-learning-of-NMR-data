# -*- coding: utf-8 -*-
#@Author  : lynch

import numpy as np
import Proton_processing as pp
import nmrglue as ng
import matplotlib.pyplot as plt

file = 'C:\\Users\\lcc\\Desktop\\数据库\\NMR-files\\Xupf-Xujt-2019-JOC(33)-files\\3ab\\3ab-H\\15\\456.jdx'
settings = 'Chloroform'
shuju = pp.process_proton(file, settings, 'jcamp')







#shuju = pp.spectral_processing(file ,'jcamp')

#dic,total_spectral_ydata=ng.jcampdx.read(file)

'''print(len(total_spectral_ydata))
print(total_spectral_ydata[1])
total_spectral_ydata = total_spectral_ydata[0] + 1j * total_spectral_ydata[1]
print(total_spectral_ydata)
print(type(total_spectral_ydata))
print('123',total_spectral_ydata.shape)
total_spectral_ydata = ng.proc_base.ifft_positive(total_spectral_ydata)
print(total_spectral_ydata)'''
'''total_spectral_ydata = ng.proc_base.zf_double(total_spectral_ydata, 4)
total_spectral_ydata = ng.proc_base.fft_positive(total_spectral_ydata)
print(total_spectral_ydata)
corr_distance = pp.estimate_autocorrelation(total_spectral_ydata)
udic = pp.guess_udic(dic, total_spectral_ydata)
uc = ng.fileiobase.uc_from_udic(udic)  # unit conversion element
spectral_xdata_ppm = uc.ppm_scale()  # ppmscale creation
# baseline and phasing
tydata = pp.ACMEWLRhybrid(total_spectral_ydata, corr_distance)
# find final noise distribution
classification, sigma = pp.baseline_find_signal(tydata, corr_distance, True, 1)
# fall back phasing if fit doesnt converge
# calculate negative area
# draw regions
peak_regions = []
c1 = np.roll(classification, 1)
diff = classification - c1
s_start = np.where(diff == 1)[0]
s_end = np.where(diff == -1)[0] - 1
for r in range(len(s_start)):
    peak_regions.append(np.arange(s_start[r], s_end[r]))
tydata = tydata / np.max(abs(tydata))
print("corrected")
print(tydata)'''
