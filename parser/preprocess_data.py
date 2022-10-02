import numpy as np
import os
import xarray
import pickle
import scipy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def preprocess():

    mag_path = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\mag\\mag\\"
    mag_data_files = os.listdir(mag_path)
    mag_data = []
    mag_time = []
    for file_path in mag_data_files:
        file_path = mag_path + file_path
        with open(file_path, 'rb') as f:
            mag  = xarray.Dataset(pickle.load(f))
            time = mag["B1GSE"]["Epoch1"].data
            BGSE = mag["B1GSE"].data
            mag_data.append(BGSE)
            mag_time.append(time)
    mag_data = np.stack(mag_data).reshape(-1, 3)
    mag_data = np.where(mag_data > -1e5, mag_data, np.nan)
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
    mag_datax = mag_data[:, 0]
    nans = np.isnan(mag_datax)
    f_nonan = scipy.interpolate.interp1d(mag_time[~nans], mag_datax[~nans])
    mag_datax[nans] = f_nonan(mag_time[nans])

    f = scipy.interpolate.interp1d(mfi_time, mfi_data[:, 0], fill_value="extrapolate")
    mfi_data = f(mag_time)

    #resize
    index1 = len(mag_time)//3
    index2 = index1*2
    index1_value = mag_time[index1]
    index2_value = mag_time[index2]
    mfi_data = mfi_data[index1:index2]
    mag_datax = mag_datax[index1:index2]
    mag_time = mag_time[index1:index2]

    # np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"time", mag_time)
    # np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"magdata", mag_datax)
    # # np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"mfitime", mfi_time)
    # np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"mfidata", mfi_data)

    swe_path = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\swe\\swe\\"
    swe_data_files = os.listdir(swe_path)
    swe_density = []
    swe_temperature = []
    swe_velocity = []
    swe_time = []
    for file in swe_data_files:
        file = swe_path + file
        with open(file, 'rb') as file:
            doc = xarray.Dataset(pickle.load(file))
            time = doc["Epoch"].data
            density = doc["Proton_Np_nonlin"].data
            temperature = doc["Proton_W_nonlin"].data
            Xvelocity = doc["Proton_VX_nonlin"].data
            Yvelocity = doc["Proton_VY_nonlin"].data
            Zvelocity = doc["Proton_VZ_nonlin"].data
            velocity = [np.array([x,y,z]) for x, y, z in zip(Xvelocity, Yvelocity, Zvelocity)]
            swe_density.append(density)
            swe_temperature.append(temperature)
            swe_velocity.append(velocity)
            swe_time.append(time)

    swe_density =np.concatenate(swe_density, axis=0)
    swe_temperature = np.concatenate(swe_temperature, axis=0)
    swe_velocity = np.concatenate(swe_velocity, axis=0).reshape(-1, 3)
    swe_time = np.concatenate(swe_time, axis=0)

    #resize
    index1 = np.abs(swe_time - index1_value).argmin()
    index2 = np.abs(swe_time - index2_value).argmin()
    swe_time = swe_time[index1:index2]
    swe_temperature = swe_temperature[index1:index2]
    swe_velocity = swe_velocity[index1:index2]
    swe_density = swe_density[index1:index2]

    swe_density = np.where(swe_density < 99998, swe_density, np.nan)
    swe_temperature = np.where(swe_temperature < 99998, swe_temperature, np.nan)
    swe_velocity = np.where(swe_velocity < 99998, swe_velocity, np.nan)

    f = scipy.interpolate.interp1d(swe_time, swe_temperature, fill_value="extrapolate")
    swe_temperature = f(mag_time)
    f = scipy.interpolate.interp1d(swe_time, swe_density, fill_value="extrapolate")
    swe_density = f(mag_time)
    f = scipy.interpolate.interp1d(swe_time, swe_velocity[:, 0], fill_value="extrapolate")
    swe_velocity = f(mag_time)
    swe_time = f(mag_time)

    plt.plot(swe_density)
    plt.show()
    plt.plot(swe_temperature)
    plt.show()
    plt.plot(swe_velocity)
    plt.show()

    np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"swedensity", swe_density)
    np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"swetemperature", swe_temperature)
    np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"swevelocity", swe_velocity)
    np.save("C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\"+"swetime", swe_time)

def print_plots():
    swedensity = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\swedensity.npy" #'Proton number density Np (n/cc) from non-linear analysis (linear scale)'
    swetemperature = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\swetemperature.npy" #'Scalar [isotropic] proton thermal speed [km/s], from the trace of the anisotropic temperatures.'
    swetime = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\swetime.npy"
    swevelocity = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\swevelocity.npy" #'Proton velocity component Vx (GSE, km/s) from moment analysis'
    magdata = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\magdata.npy"
    magtime = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\magtime.npy"
    mfidata = "C:\\Users\\garri\\Desktop\\programming\\spaceapp-challenge\\mfidata.npy"

    time = np.load(magtime)

    data = np.load(swedensity)
    plt.plot(time[60*60*4-60*30:60*60*8+30*60]/(60*60*1000)-time[0]/(60*60*1000), data[60*60*4-60*30:60*60*8+30*60])
    plt.title("Proton number density (n/cc)")
    plt.ylabel("cm$^{-3}$")
    plt.xlabel("hour of the day ")
    plt.show()

    data = np.load(swetemperature)
    plt.plot(time[60*60*4-60*30:60*60*8+30*60]/(60*60*1000)-time[0]/(60*60*1000), data[60*60*4-60*30:60*60*8+30*60])
    plt.title("Scalar [isotropic] proton thermal speed [km/s]")
    plt.ylabel("km/s")
    plt.xlabel("hour of the day ")
    plt.show()

    data = np.load(swevelocity)
    plt.plot(time[60*60*4-60*30:60*60*8+30*60]/(60*60*1000)-time[0]/(60*60*1000), data[60*60*4-60*30:60*60*8+30*60])
    plt.title("Proton velocity component Vx (GSE, km/s)")
    plt.xlabel("hour of the day ")
    plt.ylabel("km/s")
    plt.show()
