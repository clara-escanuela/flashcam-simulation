import numpy as np
from astropy.coordinates import Angle
from scipy.stats import multivariate_normal, skewnorm, norm
import astropy.units as u
from ctapipe.image.toymodel import Gaussian, WaveformModel, ImageModel
from ctapipe.image.hillas import camera_to_shower_coordinates


class fake_shower(self)
    @u.quantity_input(x=u.m, y=u.m, length=u.m, width=u.m)
    def __init__(self, x, y, width, length, psi):
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.psi = psi
        self.intensity = intensity
        self.nsb = nsb_level_pe
        self.geometry = geometry

    def image(self, intensity, nsb_level_pe, geometry, noise=False):
        gaussian = Gaussian(self.x, self.y, self.width, self.length, self.psi)

        rng = default_rng(0)
        expected_signal = intensity * geometry.pix_area.value * gaussian(geometry.pix_x, .geometry.pix_y)

        signal = rng.poisson(expected_signal)
        if noise == False:
            noise = rng.poisson(nsb_level_pe, size=signal.shape)

        image = (signal + noise)

        return image, signal, noise

    def time(self, geometry, time_gradient, time_intercept):

        longitudinal, _ = camera_to_shower_coordinates(geometry.x, geometry.y, self.x, self.y, self.psi)
        longitudinal_m = longitudinal.to_value(u.m)
        time_gradient_ns_m = time_gradient.to_value(u.ns / u.m)
        time_intercept_ns = time_intercept.to_value(u.ns)

        return longitudinal_m * time_gradient_ns_m + time_intercept_ns


#class fake_pulse(self):


