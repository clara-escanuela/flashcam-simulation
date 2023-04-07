import numpy as np
from fc_sim.camera import camera_description

class broken_pixels():
    def __init__(self, geometry):
        self.geometry = geometry
        self.broken_pixels = np.full(camera_description.n_pixels(), False)

    def camera_border(self, n):
        border = self.geometry.get_border_pixel_mask(n)
        self.broken_pixels[border] = True

        return self.broken_pixels

    def brightest_pix(self, charge, geometry, n=1, neighbors=False):
        brightest_pixel = np.argmax(charge)
        
        if neighbors == True:
            neighbors = self.geometry.neighbor_matrix_sparse
            indptr = neighbors.indptr
            indices = neighbors.indices
            neighbors = indices[indptr[brightest_pixel] : indptr[brightest_pixel + 1]]
            brightest_pixel = np.append(brightest_pixel, neighbors)

        self.broken_pixels[brightest_pixel] = True

        return self.broken_pixels

def ref_convolution():
    upsampling = 10
    reference_pulse_sample_width = camera_description.ref_sample_width_nsec()
    sample_width_ns = camera_description.camera_sample_width_nsec()
    ref_max_sample = camera_description.reference_pulse().size * reference_pulse_sample_width
    reference_pulse_x = np.arange(0, ref_max_sample, reference_pulse_sample_width)
    ref_width_ns = sample_width_ns / upsampling
    ref_interp_x = np.arange(0, reference_pulse_x.max(), ref_width_ns)
    ref_interp_y = np.interp(
        ref_interp_x, reference_pulse_x, reference_pulse
    )
    ref_interp_y /= ref_interp_y.sum() * ref_width_ns
    origin = ref_interp_y.argmax() - ref_interp_y.size // 2

    n_pixels = charge.size
    n_upsampled_samples = n_samples * upsampling
    readout = np.zeros((n_pixels, n_upsampled_samples))

    sample = (time / ref_width_ns).astype(np.int64)
    outofrange = (sample < 0) | (sample >= n_upsampled_samples)
    sample[outofrange] = 0
    charge[outofrange] = 0
    readout[np.arange(n_pixels), sample] = charge
    convolved = convolve1d(
        readout, ref_interp_y, mode="constant", origin=origin
    )
    sampled = (
        convolved.reshape(
            (n_pixels, convolved.shape[-1] // upsampling, upsampling)
        ).sum(-1)
        * ref_width_ns  # Waveform units: p.e.
    )
    return sampled

def simulate_nsb(rate, uncertainty, no_baseline=True, seed=0):
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
    
    rng = np.random.default_rng(seed=seed)

    # Number of NSB photoelectrons per pixel in this event
    duration = camera_description.trace_length()
    n_pixels = camera_description.n_pixels()
    avg_photons_per_waveform = rate * 1e-3 * duration * 4
    n_nsb_per_pixel = rng.poisson(avg_photons_per_waveform, n_pixels)

    n_nsb = np.reshape(n_nsb_per_pixel/duration, (len(n_nsb_per_pixel), -1))
    pixel = np.repeat(n_nsb, duration, axis=-1)
    size = np.shape(pixel)

    nsb_per_sample_per_pix = pixel + np.reshape(np.random.normal(0, uncertainty, size[0]*size[1]), (size[0], size[1]))

    if no_baseline:
        nsb_per_sample_per_pix = nsb_per_sample_per_pix - n_nsb

    return nsb_per_sample_per_pix


class GaussianNoise():
    def __init__(self, stddev=1, seed=None):
        """
        Fluctuate readout with Gaussian noise
        Parameters
        ----------
        stddev : float
            Standard deviation of the gaussian noise
            Units: photoelectrons / ns
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self.stddev = stddev
        self.seed = seed

    def add_to_readout(self, readout):
        rng = np.random.default_rng(seed=self.seed)
        return rng.normal(readout, self.stddev)


class noise_from_template():
    default_path = "files/noise_FC_blabla.txt"

    def __init__(self, n_samples, sample_width, filepath=default_path, stddev=1, seed=None):
        """
        Noise defined by a template
        Parameters
        ----------
        n_samples : int
            Number of samples in the waveform
        sample_width : float
            Width of samples in the waveform (ns)
        stddev : float
            Standard deviation of the noise
            Units: photoelectrons / ns
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self._n_samples = n_samples
        self._sample_width = sample_width
        self._filepath = filepath
        self._stddev = stddev
        self._seed = seed

        self._frequency, self._v_root = np.loadtxt(filepath, delimiter=',', unpack=True)

        # Find scaling for requested stddev
        n_samples_long = int(1e7)
        voltage = self.get_interpolated_voltage(n_samples_long, sample_width)
        frequency_spectrum = self.get_frequency_spectrum(voltage)
        noise = self.get_noise(frequency_spectrum, n_samples_long)
        self.scale = stddev / np.std(noise)

    def get_interpolated_voltage(self, n_samples, sample_width):
        df = np.fft.fftfreq(n_samples) / sample_width
        df_positive = df[:len(df)//2]
        delta_df_positive = df_positive[1] - df_positive[0]
        f = interp1d(self._frequency, self._v_root)
        frequency_min = np.min(self._frequency)
        frequency_max = np.max(self._frequency)
        frequency_range = frequency_max - frequency_min
        frequency_interp = np.arange(frequency_min, frequency_max, frequency_range/n_samples)
        v_root_interp = f(frequency_interp)
        return v_root_interp * np.sqrt(delta_df_positive)

    def get_frequency_spectrum(self, voltage):
        rng = np.random.default_rng(seed=self._seed)
        phi = rng.uniform(0, 2*np.pi, size=voltage.size)  # Randomising phi from 0 to 2pi
        cplx = np.zeros(voltage.size, dtype=complex)
        i = np.arange(1, voltage.size//2)
        cplx.real[i] = voltage[i]*np.cos(phi[i])
        cplx.imag[i] = -voltage[i]*np.sin(phi[i])
        cplx.real[-i] = voltage[i]*np.cos(phi[i])
        cplx.imag[-i] = voltage[i]*np.sin(phi[i])
        return cplx

    @staticmethod
    def get_noise(frequency_spectrum, n_samples):
        return np.fft.ifft(frequency_spectrum) * n_samples * 1e-9  # Convert to Volts

    @staticmethod
    def get_noise_envelope(noise, sample_len):
        """
        Return back to the noise envelope from the simulated noise
        Parameters
        ----------
        noise : ndarray
            Noise component of the waveform
        sample_len : int
            Number of samples in the readout
        Returns
        -------
        ndarray
        """
        spectrum = np.fft.fft(noise*1e9 / sample_len)  # Convert to nV and rescale for FFT
        return np.abs(spectrum)

    def add_to_readout(self, readout):
        voltage = self.get_interpolated_voltage(self._n_samples, self._sample_width)
        frequency_spectrum = self.get_frequency_spectrum(voltage)
        noise = self.get_noise(frequency_spectrum, self._n_samples)
        return readout + noise * self.scale

#def excess_noise_factor():

#def gain():

def afterpulsing(ap_prob=1e-4):
    """
    ap_prob: probability of finding more than 1 pe per ns
    """
    pe_per_cam = ap_prob*camera_description.trace_length()*4*camera_description.n_pixels()*camera_description.sampling_rate()  #number of times we find more than 1 pe afterpulses

    charge_per_pix = np.random.choice(np.arange(1, 4, 0.1), round(pe_per_cam+0.5))  #not great, better a probability model 
    pixel = np.random.choice(np.arange(camera_description.n_pixels()), round(pe_per_cam+0.5))

    ap_charge = np.zeros(camera_description.n_pixels())
    ap_charge[pixel] = charge_per_pix

    time_per_pix = np.random.choice(np.arange(10, 200), round(pe_per_cam+0.5))
    ap_time = np.zeros(camera_description.n_pixels())
    ap_time[pixel] = time_per_pix

    return ap_charge, ap_time

def time_jitter(trigger_time, time_jitter):
    """
    trigger_time: 1D array of length 1764
    time_jitter: float/int
    """
    t = np.random.normal(0, time_jitter, len(trigger_time))
    
    return trigger_time + t
    



