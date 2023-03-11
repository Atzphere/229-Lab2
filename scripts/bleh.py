import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import optimize as opt
import datasets
from cylinder import Cylinder

y_delta = 0.2

def to_kelvin(C):
    '''
    Converts celsius to kelvin
    '''
    return C + 273.15

roughf = datasets.Trial("../data/smooth.csv", y_delta, to_kelvin)

plt.style.use('seaborn')

tmin = np.min(roughf.times)
tmax = np.max(roughf.times)
interval = (tmin, tmax)
samples = 40000
timestep = (tmax - tmin) / samples

times = np.linspace(tmin, tmax, samples)

# data = np.loadtxt("229exp1data3.csv", delimiter=",", comments="#", skiprows=1)
data_time = roughf.times
rough = roughf.y
# lacq = data[:, 2] + 273.15
# polished = data[:, 3] + 273.15

rough_cyl = Cylinder(30.7 / 100, 25 / 1000, 409 / 1000)
print(rough_cyl.area)
print(rough_cyl.volume)

A = .0245
T_amb = 23.9 + 273.15  # K

M = 0.368997
C = 890
rho = 2700
V = 0.00013666
D = 0.0255
sigma = 5.67E-8

last_epsilon = 999
y = []


def q_to_T(q):
    return q / (rho * V * C)


def diffeq(t, yarr, *args):
    # return dT

    # get dQ

    T = yarr[0]
    epsilon = np.abs(args[0])

    delta = (T - T_amb)
    # print(delta, t)

    h = 1.32 * np.abs(delta / D)**(1 / 4)
    # print(h)

    convect = -h * A * delta
    # print("convect:", q_to_T(convect))
    rad_out = -A * epsilon * sigma * T**4
    # print("rad1", q_to_T(rad_out))
    rad_in = A * epsilon * sigma * T_amb**4
    # print("rad2", q_to_T(rad_in))

    dQ = convect + rad_out + rad_in

    dT = q_to_T(dQ)
    print(dT)
    return dT


'''

sol = integrate.solve_ivp(
    diffeq,
    interval,
    initials,
    t_eval=times,
    method='LSODA',
    args=(0.5, 1))

y = sol["y"].flatten()
print(y)
t = sol["t"].flatten()
plt.plot(t, y)
plt.show()

'''


def wrap(t, epsilon):
    global last_epsilon
    global y
    print("I was called")
    if last_epsilon != epsilon:
        print("recalc")
        last_epsilon = epsilon
        sol = integrate.solve_ivp(
            diffeq,
            interval,
            initials,
            t_eval=times,
            method='LSODA',
            args=(epsilon, 1))
        y = sol["y"].flatten()
        # print("ivp solved with epsilon {}".format(epsilon))

    index = tuple(np.where(np.abs(times - t) <= (timestep / 1.5)))
    return y[index][0]


f = np.vectorize(wrap)


initials = [rough[0]]
print(type(initials))
print(initials)

params, pcov = opt.curve_fit(
    f, data_time, rough, sigma=np.ones(len(data_time)), p0=(0.5,))

perr = np.sqrt(np.diag(pcov))[0]
fig, ax = plt.subplots()
ax.errorbar(data_time, rough, yerr=1, label="Data", fmt='.')
final_epsilon = params[0]
ax.plot(data_time, f(data_time, final_epsilon), label="Model")
ax.set_title(
    "Cooling curve of rough object over time ($\\epsilon = {val:.4f}\\pm {unc:.4f}$)".
    format(val=final_epsilon, unc=perr))
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Temperature (K)")
ax.legend()

plt.show()
