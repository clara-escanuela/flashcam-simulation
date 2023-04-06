import numpy as np
from astropy.coordinates import Angle
from scipy.stats import multivariate_normal, skewnorm, norm
import astropy.units as u
from ctapipe.image.toymodel import *
from ctapipe.image.hillas import camera_to_shower_coordinates
from fc_sim.camera import ctapipe_subarray, camera_description
from fc_sim.fun import time_jitter

def get_fc_subarray(tel_id=10):
    subarray = ctapipe_subarray(tel_id, url="dataset://gamma_prod5.simtel.zst")
    return subarray

def get_random_toy_waveforms(subarray, minCharge=100, maxCharge=1000, n_samples=128):
    """
    Get fake image/pulse from ctapipe existing tools
    """
    tel_id = list(subarray.tel.keys())[0]
    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    readout = subarray.tel[tel_id].camera.readout

    random = np.random.RandomState(1)

    charge = random.uniform(minCharge, maxCharge, n_pixels)
    mid = (n_samples // 2) / readout.sampling_rate.to_value(u.GHz)
    time = random.uniform(mid - 1, mid + 1, n_pixels)

    waveform_model = WaveformModel.from_camera_readout(readout)
    waveform = waveform_model.get_waveform(charge, time, n_samples)

    selected_gain_channel = np.zeros(charge.size, dtype=np.int64)

    return waveform, subarray, tel_id, selected_gain_channel, charge, time

def get_toy_waveforms(subarray, charge, time, n_samples=128):
    """
    Get fake image/pulse from ctapipe existing tools
    """
    tel_id = list(subarray.tel.keys())[0]
    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    readout = subarray.tel[tel_id].camera.readout

    mid = (n_samples // 2) / readout.sampling_rate.to_value(u.GHz)

    waveform_model = WaveformModel.from_camera_readout(readout)
    waveform = waveform_model.get_waveform(charge, time, n_samples)

    selected_gain_channel = np.zeros(charge.size, dtype=np.int64)

    return waveform

def get_toy_image(
    psi, x, y, width, length, intensity, time_gradient, time_intercept, subarray, nsb_level_pe=3
):
    geometry = subarray.tel[10].camera.geometry

    model = Gaussian(x=x, y=y, width=width, length=length, psi=psi)
    rng = np.random.default_rng(0)
    image, signal, noise = model.generate_image(
        geometry, intensity, nsb_level_pe, rng=rng
    )
    
    time = rng.uniform(0, camera_description.trace_length(), camera_description.n_pixels())
    time[signal>0] = obtain_time_image(geometry.pix_x[signal>0], geometry.pix_y[signal>0], x, y, psi, time_gradient, time_intercept)
    time = time_jitter(time, time_jitter=2)

    return image, signal, noise, time

def get_waveform_from_image(
        psi, x, y, width, length, intensity, time_gradient, time_intercept, subarray, nsb_level_pe=3
):
    image, signal, noise, time = get_toy_image(psi, x, y, width, length, intensity, time_gradient, time_intercept, subarray, nsb_level_pe)
    waveform = get_toy_waveforms(subarray, signal[signal>0], time[signal>0])

    return waveform, signal, noise, time


