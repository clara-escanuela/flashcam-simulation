import numpy as np
from camera import camera_description

def simulate_nsb(ratei, initial_pe):
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
    rng = np.random.default_rng(seed=self.seed)

    # Number of NSB photoelectrons per pixel in this event
    duration = camera.continuous_readout_duration
    n_pixels = 1764
    avg_photons_per_waveform = rate * 1e6 * duration * 4e-9
    n_nsb_per_pixel = rng.poisson(avg_photons_per_waveform, n_pixels)

    # Pixel containing each photoelectron
    pixel = np.repeat(np.arange(n_pixels), n_nsb_per_pixel)

    # Uniformly distribute NSB photoelectrons in time across waveform
    n_pe = pixel.size
    time = rng.uniform(0, duration, size=n_photoelectrons)

    charge = np.ones(n_pe)
    initial_pe = Photoelectrons(pixel=pixel, time=time, charge=charge)

    # Process the photoelectrons through the SPE spectrum
    pe = camera.photoelectron_spectrum.apply(initial_pe, rng)

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

#def pulse_add_noise(self):


