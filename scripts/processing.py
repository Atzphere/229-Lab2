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


rough = datasets.Trial("../data/rough.csv", y_delta, to_kelvin)
lacquer = datasets.Trial("../data/lacquer.csv", y_delta, to_kelvin)
smooth = datasets.Trial("../data/smooth.csv", y_delta, to_kelvin)
print(smooth.times, smooth.y)

T_amb = to_kelvin(23.5)  # K

# MODEL CYLINDERS:

rough_cyl = Cylinder(30.7 / 100, 25 / 1000, 409 / 1000)
lacquer_cyl = Cylinder(30.65 / 100, 25.4 / 1000, 421 / 1000)
smooth_cyl = Cylinder(30.55 / 100, 25 / 1000, 427 / 1000)

rough_cyl = smooth_cyl
rough = smooth

# INTENSIVE PROPERTIES / PHYSICAL CONSTANTS

C = 890  # density of aluminum in kg / m^3
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
    # print("i was called in the first place")
    # return dT

    # get dQ

    # args go: epsilon, Cylinder

    T = yarr[0]

    epsilon = np.abs(args[0])
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


rough_sys = fitsystem.DiffEqFitSystem(diffeq, 999, rough.initial_y,
                                      get_timestep(
                                          rough.times), rough.time_interval,
                                      constant_params=tuple([rough_cyl]))
# print(rough_sys.get_values((900), 0.5))
# print(np.ndim(rough.y))
params, pcov = opt.curve_fit(rough_sys.get_values, rough.times,
                             rough.y, sigma=y_delta * np.ones(len(rough.y)), p0=(0.5,))

perr = np.sqrt(np.diag(pcov))[0]

rough_sys.get_system_stats()

# PLOTTING

fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax[0].errorbar(rough.times, rough.y, yerr=y_delta, label="Data", fmt='.')
print(params)
final_epsilon = params[0]
model = rough_sys.get_values(rough.times, final_epsilon)
ax[0].plot(rough.times, model, label="Model")
ax[0].set_title(
    "Cooling curve of rough object over time ($\\epsilon = {val:.4f}\\pm {unc:.4f}$)".
    format(val=final_epsilon, unc=perr))
ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Temperature (K)")
ax[0].legend()

ax[1].scatter(rough.times, rough.y - model)

plt.show()
