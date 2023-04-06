import numpy as np
from os.path import join, dirname
from ctapipe.coordinates import EngineeringCameraFrame
import astropy.units as u
from ctapipe.io import EventSource
from ctapipe.instrument import TelescopeDescription, SubarrayDescription, CameraGeometry, CameraReadout, CameraDescription, OpticsDescription

def ctapipe_subarray(tel_id, url="dataset://gamma_prod5.simtel.zst"):
    source = EventSource(url)
    subarray = source.subarray
    subarray = subarray.select_subarray([tel_id])

    return subarray

class camera_description:
    """
    Container for properties which define the camera
    """
    geometry = CameraGeometry.from_name("FlashCam")
    readout = CameraReadout.from_name("FlashCam")

    def reference_pulse():
        """
        Reference pulse shape for the FlashCam
        """
        time = CameraReadout.from_name("FlashCam").reference_pulse_sample_time.to_value(u.ns)
        signal = CameraReadout.from_name("FlashCam").reference_pulse_shape[0]

        return time, signal

    def n_pixels():
        return CameraReadout.from_name("FlashCam").n_pixels

    def trace_length():
        return 128

    def n_samples():
        return CameraReadout.from_name("FlashCam").n_samples

    def continuous_readout_duration():
        return 1000  #in units of ns

    def sampling_rate():
        return CameraReadout.from_name("FlashCam").sampling_rate.to_value(u.GHz)

    def pixel_pos():
        return CameraGeometry.from_name("FlashCam").pix_id, CameraGeometry.from_name("FlashCam").pix_x, CameraGeometry.from_name("FlashCam").pix_y

    def pixel_r():
        return CameraGeometry.from_name("FlashCam").pix_id, (CameraGeometry.from_name("FlashCam").pix_x.to_value(u.m)**2 + CameraGeometry.from_name("FlashCam").pix_y.to_value(u.m)**2)*u.m

    def pixel_area():
        return CameraGeometry.from_name("FlashCam").pix_area

    def ref_sample_width_nsec():
        return CameraReadout.from_name("FlashCam").reference_pulse_sample_width.to_value(u.ns)

    def camera_sample_width_nsec():
        return 1.0 / CameraReadout.from_name("FlashCam").sampling_rate.to_value(u.GHz)
