import numpy as np
import os, argparse

sim_path = "/d1/hin/nieves/Data"

def sim_laser(NLaser_events, photons, trace_length, gain, nsb, tel_id=10, path_out = sim_path):
    """
    Laser simulations
    """
    sim_telarray_path = "/d1/hin/nieves/Sim_new/sim_telarray/LightEmission/"

    NLaser_photons = photons

    path_file = path_out + "/data/flatfield_Nevents{}_simul_CT{}_trace{}_gain{}_nsb{}GHz.gz".format(NLaser_events, tel_id, trace_length, gain, nsb)
   
    # Provide preprocessor setup for the configuration
    arguments = [
    '-c cfg/CTA/CTA-PROD6-MST-FlashCam.cfg', # Main configuration file

    '-DCTA_SOUTH',
     
    '-C maximum_telescopes=1',  # Safety feature to allow only one telescope in output
    '-C Altitude=2147',         # Altitud of telescope
    '-C atmospheric_transmission=atm_trans_2147_1_10_0_0_2147.dat', # Atmospheric data for Paranal
    '-C telescope_theta=0',  # Telescope pointing
    '-C telescope_phi=0',

    '-C photon_delay=0',
    '-C nightsky_background="all:0.{}"'.format(nsb),
    '-C nsb_scaling_factor=1.0',
    '-C nsb_autoscale_airmass=0.0,0.0',

    '-C mirror_degraded_reflection=1',  #1 for flatfield simulations, no reflection with mirror. Camera degradation in CAMERA_DEGRADED_EFFICIENCY.
    "-C bypass_optics=1",  # Light does not go to primary mirror 
    "-C dsum_prescale=0,0",  # No scaling
    #"-C dsum_clipping=0,"
    "-C dsum_threshold=1",  # Minimum amplitude level above pedestal to trigger
    "-C power_law=0.00",

    # Accept all events (else events with less photons get filtered out)
    '-C min_photoelectrons=0',
    '-C min_photons=0',
    #"-C force_fake_trigger=1",  #If there are photons, use median time

    "-C iobuf_maximum=16000000000",  # 16 GBYte
    "-C iobuf_output_max=200000000",  # 200 million single-photon bunches

    # The following options disable all runwise channel-to-channel variations
    '-C laser_var_photons=all:0',
    #'-C qe_variation=0',
    '-C gain_variation=0',
    #'-C fadc_var_sensitivity=0',
    '-C fadc_var_pedestal=0',
    '-C fadc_err_pedestal=0',
    '-C fadc_sysvar_pedestal=0',
    '-C transit_time_compensate_error=0',
    '-C transit_time_calib_error=0',
    '-C transit_time_compensate_step=4',
    '-C pm_voltage_variation=0',
    '-C pm_gain_index=0',
    '-C transit_time_error=4',
    '-C flatfielding=0',
    '-C fadc_amplitude={}'.format(gain),
    #'-C laser_pulse_sigtime=0',  # If pulse shpae of emitted light is a gaussian with this sigma in ns
    #'-C fadc_pulse_shape=0',

     #Output FADC traces with 'nsamples' bins; dark pedestal to 200 LSB; do not clip at 4095.

    '-C pulse_analysis=0',
    '-C sampled_output=1',
    '-C fadc_bins={}'.format(trace_length),
    '-C fadc_sum_bins=0',
    '-C fadc_pedestal=200',
    '-C fadc_noise=1.0',
    '-C fadc_max_signal=4095',

    #Configuration for Laser
    '-C laser_external_trigger=1',

    #'-C histogram_file=.hdata.gz',
    '-C show=all',  # Show entire output of sim_telarray in console
    '-C output_format=1', # 0 = Only ADC sums are written instead of waveforms, 1 = full waveforms for each pixel
    '-o {}'.format(path_file), # output file

    'le1.iact' # Dummy file. Best match.
    ]

    ff_arguments = [
    '--distance 1600',
    '--camera-radius 125',
    '--altitude 2147',
    '--atmosphere 26',
    '--run 1',
    '--spectrum 400',
    '--bunchsize 1',
    '--angular-distribution isotropic',
    #'lightpulse 2',
    '-o ../le1.iact',
    ]

    # Build the command string
    cmd = ''
    for argument in arguments:
        cmd += argument + ' '

    cnd = ' '
    for argument in ff_arguments:
        cnd += argument + ' '
  
    os.chdir(sim_telarray_path)
    os.system("rm {}".format(path_file))
    os.system("ln -s ../cfg/CTA/atmprof26.dat")  # Atmospheric data
    os.system("./ff-1m --events {} --photons {}".format(NLaser_events, NLaser_photons) + cnd)
    os.chdir("/d1/hin/nieves/Sim_new/sim_telarray/")
    os.system("./bin/sim_telarray "+ cmd + "> " + path_out+"/log/flatfield_output_Nevents{}_simul_CT{}_trace{}_gain{}_nsb{}GHz.txt".format(NLaser_events, tel_id, trace_length, gain, nsb))
                                                                                                                                                
#Generating file

NLaser_events = "10000"
pe2ph = 2.7869342315
distance = 1600  #distance of the laser to the camera in cm
pix_area = 21.79 #entrance area of one pixel in cm2
ps = 100

photons = 100*pe2ph*4*np.pi*1600**2/21.49

nph = [photons]  #photons * 4 * np.pi * distance**2 / pix_area
CTA_tel_list = [10]

ph_list = nph
for tel in CTA_tel_list:
    for ph in ph_list:
        sim_laser(NLaser_events  = NLaser_events, photons=ph, tel_id = tel, trace_length = 100, gain = 10, nsb=0)

    
