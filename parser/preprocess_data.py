import numpy as np
import spacepy
import os
import sys
import cdflib
import scipy
import h5py
import networkx
import xarray
import pickle
import scipy


mag_path = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\mag\\mag\\"
mag_data_files = os.listdir(mag_path)
mag_data = []
mag_time = []
for file_path in mag_data_files:
    file_path = mag_path + file_path
    print(file_path)
    with open(file_path, 'rb') as f:
        mag  = xarray.Dataset(pickle.load(f))
        time = mag["B1GSE"]["Epoch1"].data
        BGSE = mag["B1GSE"].data
        mag_data.append(BGSE)
        mag_time.append(time)
mag_data = np.stack(mag_data).reshape(-1, 3)
#mag_data = np.where(mag_data > -1e5, mag_data, 0)
mag_time = np.stack(mag_time).flatten()
mfi_path = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\mfi\\"
mfi_data_files = os.listdir(mfi_path)
mfi_data = []
mfi_time = []
for file_path in mfi_data_files:
    file_path = mfi_path + file_path
    with open(file_path, 'rb') as f:
        mfi  = xarray.Dataset(pickle.load(f))
        time = xarray.core.utils.Frozen(mfi["BGSE"].indexes.variables).mapping.mapping["Epoch"].data
        BGSE = mfi["BGSE"].data
        mfi_data.append(BGSE)
        mfi_time.append(time)
mfi_data = np.concatenate(mfi_data, axis=0).reshape(-1, 3)
mfi_data = np.where(mfi_data > -1e5, mfi_data, np.nan)
mfi_time = np.concatenate(mfi_time, axis=0).flatten()
print(np.min(mfi_time), np.min(mag_time))
print(np.max(mfi_time), np.max(mag_time))
mag_datax = mag_data[:, 0]
nans = np.isnan(mag_datax)
f_nonan = scipy.interpolate.interp1d(mag_time[~nans], mag_datax[~nans])
mag_datax[nans] = f_nonan(mag_time[nans])
f = scipy.interpolate.interp1d(mfi_time, mfi_data[:, 0], fill_value="extrapolate")
mfi_data = f(mag_time)
np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"magtime", mag_time)
np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"magdata", mag_data)
np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"mfitime", mfi_time)
np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"mfidatas", mfi_data)

#swe_path = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\swe\\swe\\"
# swe_data_files = os.listdir(swe_path)
# swe_data = []
# swe_time = []
# with open("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\swe\\swe\\wi_h1_swe_20220101_v01.pickle", 'rb') as file:
#     doc = xarray.Dataset(pickle.load(file))
#     time = xarray.core.utils.Frozen(doc["BX"].indexes.variables).mapping.mapping["Epoch"].data
#     time2 = xarray.core.utils.Frozen(doc["BY"].indexes.variables).mapping.mapping["Epoch"].data
#     time3 = xarray.core.utils.Frozen(doc["BZ"].indexes.variables).mapping.mapping["Epoch"].data
#     BXdata = doc["BX"].data
#     BYdata = doc["BY"].data
#     BZdata = doc["BZ"].data
#     sameval = True
#     list_of_list = [time.shape, time2.shape, time3.shape, BXdata.shape, BYdata.shape, BZdata.shape]
#
#     print([x == time.shape for x in list_of_list])
#     for x,y in zip(time, time2):
#         if(x!=y):
#             sameval = False
#     for x,y in zip(time2, time3):
#         if(x!=y):
#             sameval = False
#     if sameval:
#         print([(d, t) for d, t in zip(BXdata, time)])
