import os, argparse

parser = argparse.ArgumentParser()

parser.add_argument('--nevents', type=int, required=True)
parser.add_argument('--run', type=int, required=True)
parser.add_argument('--beta', type=float, required=False, default=1.0)
parser.add_argument('--theta', type=float, required=False, default=0.0)
parser.add_argument('--phi', type=float, required=False, default=0.0)
parser.add_argument('--x', type=float, required=False, default=0.0)
parser.add_argument('--y', type=float, required=False, default=0.0)
parser.add_argument('--nsb', type=int, required=False, default=1)
args = parser.parse_args()
dicts = vars(args)

x = dicts['x']
y = dicts['y']
nevents = dicts['nevents']
nsb = dicts['nsb']
theta = dicts['theta']
phi = dicts['phi']
beta = dicts['beta']
run = dicts['run']

sim_path = "/d1/hin/nieves/Data/fake-muons"
sim_telarray_path = "/d1/hin/nieves/Sim_new/sim_telarray/"
path_file = sim_path + "/data/fake_muons_run{}_Nevents={}_simul_MST_beta={}_theta={}_phi={}_xy={}_nsb={}GHz.simtel.gz".format(run, nevents, beta, theta, phi, str(x)+str(y), nsb)
corsika_file = sim_path + "/data/fake_muons_run{}_Nevents={}_simul_MST_beta={}_theta={}_phi={}_xy={}_nsb={}GHz.corsika.gz".format(run, nevents, beta, theta, phi, str(x)+str(y), nsb)
os.chdir(sim_telarray_path)
os.system("rm {}".format(path_file))

# Run the fake-muon, with output in sim_telarray directory (generally not a good idea but keeping the example simple):
os.system("LightEmission/fake-muon -N {} --run {} --theta {} --phi {} --beta {} -x {} -y {}"
          " --atmosphere cfg/CTA/atmprof_ecmwf_south_winter_fixed.dat -o "
          "{}".format(nevents, run, theta, phi, beta, x, y, corsika_file))
os.system('bin/sim_telarray -c cfg/CTA/CTA-PROD6-MST-FlashCam.cfg -DCTA_SOUTH -C nightsky_background="all:0.{}" -C telescope_theta=0 -C telescope_phi=0 '
          '-o {} {}'.format(nsb, path_file, corsika_file))

