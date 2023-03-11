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
maddie = datasets.Trial("../data/maddie.csv", y_delta, to_kelvin)
# print(smooth.times, smooth.y)

T_amb = to_kelvin(24.2)  # K

# MODEL CYLINDERS:

rough_cyl = Cylinder(30.7 / 100, 25 / 1000, 409 / 1000)
lacquer_cyl = Cylinder(30.65 / 100, 25.4 / 1000, 421 / 1000)
smooth_cyl = Cylinder(30.55 / 100, 25 / 1000, 427 / 1000)
maddie_cyl = Cylinder(30.52 / 100, 25.5 / 1000, 890 *
                      (30.52E-2 * (np.pi * 25.5E-3**2 / 2)))
# print(smooth_cyl.volume)
maddie_cyl.mass = maddie_cyl.volume * 890
# print(maddie_cyl.volume * 890)

rough_cyl = rough_cyl
rough = rough
trialname = "rough"

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
    # return dT

    # get dQ

    # args go: epsilon, T_amb, Cylinder

    # print(args)

    T = yarr[0]
    epsilon = np.abs(args[0])
    T_amb = args[1]
    cylinder = args[2]

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


rough_sys = fitsystem.DiffEqFitSystem(diffeq, (999, 298), rough.initial_y,
                                      get_timestep(
                                          rough.times), rough.time_interval,
                                      constant_params=rough_cyl)

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

params, pcov = opt.curve_fit(rough_sys.get_values, rough.times,
                             rough.y,
                             sigma=y_delta * np.ones(len(rough.y)), p0=(0.5, 298))


perr = np.sqrt(np.diag(pcov))[0]

rough_sys.get_system_stats()

# PLOTTING


fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax[0].errorbar(rough.times, rough.y, yerr=y_delta, label="Data", fmt='.')
final_epsilon = params[0]
final_t = params[1]
print("Final temperature: {}".format(final_t))
model = rough_sys.get_values(rough.times, (final_epsilon, final_t))
ax[0].plot(rough.times, model, label="Model")
ax[0].set_title(
    "Cooling curve of {trial} object over time ($\\epsilon = {val:.4f}\\pm {unc:.4f}$)".
    format(trial=trialname,val=final_epsilon, unc=perr))
ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Temperature (K)")
ax[0].legend()

residuals = rough.y - model
in_sig = 100 * len(residuals[np.abs(residuals) <= y_delta]) / len(residuals)

ax[1].set_title(
    "Residuals ({per:.2f} of points % within 1 sigma of fit)"
    .format(per=in_sig))
ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Temperature (K)")
ax[1].errorbar(rough.times, rough.y - model, yerr=y_delta, fmt=".",
               elinewidth=1, markersize=5, ecolor="orange")
ax[1].axhline(0, color="gray")

plt.show()
