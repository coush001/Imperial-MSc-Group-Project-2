import argparse
import numpy


parser = argparse.ArgumentParser()

parser.add_argument('v', help="Initial velcoity", type=float)
parser.add_argument('m', help="Initial mass", type=float)
parser.add_argument('theta', help="Initial Asteroid trajectory angle", type=float)
parser.add_argument('r', help="Initial Asteroid radius", type=float)
parser.add_argument('-p', '--density', help="Asteroid Density", type=float, default='3300')
parser.add_argument('-z', '--altitude', help="Initial Altitude", type=float, default='100E3')
parser.add_argument('-g', '--gravity', help="Gravity", type=float, default='9.81')
parser.add_argument('-o', '--rho', help="planetry Radius", type=float, default='6400E3')
parser.add_argument('-s', '--sigma0', help="Asteroid Material strength", type=float, default='64E3')
parser.add_argument('-t', '--pa_tab', help="Tabulated atmospheric data, filename", default=False)
parser.add_argument('-l', '--planet', help="Specified Planet", default='Earth')
parser.add_argument('-x', '--plot', help="Plot the results for specified parameters, default=True", default=False,
                    action='store_true')
parser.add_argument('-y', '--results', help="Print key results to terminal and save all model results to 'Data.csv'",
                    default=False, action='store_true')


args = parser.parse_args()
#print(args)

##########################

# create class instance with command line inputs
meteor2 = meteor.meteor(args.v, args.m, args.theta, args.r, args.density, args.altitude, args.gravity, args.rho,
                       args.sigma0, args.pa_tab, args.planet)

data = meteor2.fire()

# sort data into individual arrays to save to .csv
time = data[0]
v = data[1][0]
m = data[1][1]
theta = data[1][2]
r = data[1][3]



