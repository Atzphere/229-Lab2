import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
import datasets
import fitsystem
from cylinder import Cylinder

# self, filename, ydelta, y_conversion, offset_loc=0, offset_amt=0

plt.style.use('seaborn')

y_delta = 0.2


def to_kelvin(C):
    '''
    Converts celsius to kelvin
    '''
    return C + 273.15

print("loading data")

rough = datasets.Trial("../thermistor_data/cut_data/rough_cooling_thermistor_2023-03-21-144151.csv", y_delta, to_kelvin)
lacquer = datasets.Trial("../thermistor_data/cut_data/lacquered_cooling_thermistor_2023-03-15-143233.csv", y_delta, to_kelvin)
smooth = datasets.Trial("../thermistor_data/cut_data/polished_cooling_thermistor_2023-03-14-155156.csv", y_delta, to_kelvin)
maddie = datasets.Trial("../data/maddie.csv", y_delta, to_kelvin)

print("...done")
# print(smooth.times, smooth.y)

# plt.errorbar(y, t, yerr=d, fmt=".")
# plt.show()


# MODEL CYLINDERS:

rough_cyl = Cylinder(30.7 / 100, 25 / 1000, 409 / 1000)
lacquer_cyl = Cylinder(30.65 / 100, 25.4 / 1000, 421 / 1000)
smooth_cyl = Cylinder(30.55 / 100, 25 / 1000, 427 / 1000)
maddie_cyl = Cylinder(30.52 / 100, 25.5 / 1000, 890 *
                      (30.52E-2 * (np.pi * 25.5E-3**2 / 2)))

rough_cyl = Cylinder(30.45 / 100, 25.2 / 1000, 409 / 1000)
lacquer_cyl = Cylinder(30.45 / 100, 25.2 / 1000, 421 / 1000)
smooth_cyl = Cylinder(27.25 / 100, 25.2 / 1000, 427 / 1000)

# print(smooth_cyl.volume)
maddie_cyl.mass = maddie_cyl.volume * 2710
# print(maddie_cyl.volume * 890)


dat = lacquer
cyl = lacquer_cyl
trialname = "lacquered"

y, d, t = dat.chunked(30)
cyl.mass = cyl.volume * 2710
print(cyl.mass)
print(dat.y, dat.times)
dat.initial_y = y[0]
dat.time_interval = (np.min(dat.times), np.max(dat.times))
dat.y = y
dat.times = t
dat.ydelta = d
y_delta = d
print(len(y) == len(t))

Atemp = to_kelvin(21.5)  # K

# INTENSIVE PROPERTIES / PHYSICAL CONSTANTS

C = 890  # heat capacity of aluminum in kg / m^3
sigma = 5.67E-8


def get_timestep(times, stepdown=25):
    '''
    Returns the largest time step that when plugged into np.arange
    will ensure there are equivalent model time values to the given data
    in times.

    Arguments

        times : np.ndarray, or iterable
            The data to find the time step for.

        stepdown : positive number
            The factor by which to reduce this time step by.

    '''
    return np.min(np.diff(times)) / stepdown


def q_to_T(q, m, C):
    return q / (m * C)


def diffeq(t, yarr, *args):
    # return dT

    # get dQ

    # args go: epsilon, T_amb, Cylinder

    # print(args)

    T = yarr[0]
    epsilon = np.abs(args[0])
    T_amb = Atemp
    cylinder = args[1]

    D = cylinder.diameter
    A = cylinder.area
    mass = cylinder.mass

    delta = (T - T_amb)
    h = 1.32 * np.abs(delta / D)**(1 / 4)

    convect = -h * A * delta
    rad_out = -A * epsilon * sigma * T**4
    # print("rad1", q_to_T(rad_out))
    rad_in = A * epsilon * sigma * T_amb**4
    # print("rad2", q_to_T(rad_in))

    dQ = convect + rad_out + rad_in

    dT = q_to_T(dQ, mass, C)
    # print("i finished being called")
    return dT


sys = fitsystem.DiffEqFitSystem(diffeq, (999,), dat.initial_y,
                                      get_timestep(
                                          dat.times), dat.time_interval,
                                      constant_params=cyl)

p = []

'''
for i in np.linspace(20, 25, 100):
    print(i)
    T_amb = to_kelvin(i)  # K
    params, pcov = opt.curve_fit(rough_sys.get_values, rough.times,
                             rough.y,
                             sigma=y_delta * np.ones(len(rough.y)), p0=(np.random.random(),))
    p.append(params)

print(p)
'''

params, pcov = opt.curve_fit(sys.get_values, dat.times,
                             dat.y,
                             sigma=y_delta * np.ones(len(dat.y)), p0=(0.231,))


perr = np.sqrt(np.diag(pcov))[0]

sys.get_system_stats()

# PLOTTING


fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax[0].errorbar(dat.times, dat.y, yerr=y_delta, label="Data", fmt='.')
final_epsilon = params[0]
# final_t = params[1]
# print("Final temperature: {}".format(final_t))
model = sys.get_values(dat.times, (final_epsilon,))
ax[0].plot(dat.times, model, label="Model")
ax[0].set_title(
    "Cooling curve of {trial} object over time ($\\epsilon = {val:.4f}\\pm {unc:.4f}$)".
    format(trial=trialname, val=final_epsilon, unc=perr))
ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Temperature (K)")
ax[0].legend()

residuals = dat.y - model
in_sig = 100 * len(residuals[np.abs(residuals) <= y_delta]) / len(residuals)

ax[1].set_title(
    "Residuals ({per:.2f} % of points within 1 sigma of fit)"
    .format(per=in_sig))
ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Temperature (K)")
ax[1].errorbar(dat.times, dat.y - model, yerr=y_delta, fmt=".",
               elinewidth=1, markersize=5, ecolor="orange")
ax[1].axhline(0, color="gray")

plt.show()
