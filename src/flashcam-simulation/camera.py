import numpy as np
from os.path import join, dirname
from ctapipe.visualization import CameraDisplay, CameraReadout, CameraGeometry
from ctapipe.coordinates import EngineeringCameraFrame
import astropy.units as u

def camera_plot(signal):
    geom = CameraGeometry.from_name("FlashCam")
    camera_geometry = geom.transform_to(EngineeringCameraFrame())

    display = CameraDisplay(
        camera_geometry,
        image=signal,
    )

    display.add_colorbar()

class camera_description(self):
    """
    Container for properties which define the camera
    """

    self.readout = CameraReadout.from_name("FlashCam")
    self.geometry = CameraGeometry.from_name("FlashCam")

    def reference_pulse(self):
        """
        Reference pulse shape for the FlashCam
        """
        readout = CameraReadout.from_name("FlashCam")
        time = self.readout.reference_pulse_sample_time.to_value(u.ns)
        signal = self.readout.reference_pulse_shape[0]

        return time, signal

    def trace_length(self):
        self.readout.n_samples = 128

        return self.readout.n_samples

    def continuous_readout_duration(self):
        return 1000  #in units of ns

    def sampling_rate(self):
        return self.readout.sampling_rate.to_value(u.GHz)

    def pixel_pos(self):
        return self.geometry.pix_id, self.geometry.pix_x, self.geometry.pix_y

    def pixel_r(self):
        return self.geometry.pix_id, (self.geometry.pix_x.to_value(u.m)**2 + self.geometry.pix_y.to_value(u.m)**2)*u.m

    def pixel_area(self):
        return self.geometry.pix_area

    def ref_sample_width_nsec(self):
        return self.readout.reference_pulse_sample_width.to_value(u.ns)

    def camera_sample_width_nsec(self):
        return 1.0 / readout.sampling_rate.to_value(u.GHz)


