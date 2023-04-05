import numpy as np
from camera import camera_description

def simulate_nsb(rate):
    """
    Obtain the photoelectron arrays for random Night-Sky Background light
    Parameters
    ----------
    rate : float
        NSB rate in MHz (number of photoelectrons per microsecond)
        This is the rate after already accounting for Photon Detection Efficiency
    Returns
    -------
    Photoelectrons
        Container for the NSB photoelectron arrays
    """
    

    return pe

def image_add_poisson_noise(image, noise_level, rng=None, correct_bias=True):
    """
    Create a new image with added poissonian noise
    """
    if not rng:
        rng = np.random.default_rng()

    noisy_image = image.copy()
    noise = rng.poisson(noise_level, size=image.shape)
    noisy_image += noise

    if correct_bias:
        noisy_image -= noise_level

    return noisy_image

def gaussian_noise(readout, stddev=1, seed=None):
    rng = np.random.default_rng(seed=seed)
    return rng.normal(readout, stddev)

def excess_noise_factor():


def time_jitter(trigger_time, time_jitter):
    t = np.random.normal(0, time_jitter, n)
    
    return trigger_time + t
    



