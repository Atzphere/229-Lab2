import numpy as np
from scipy import integrate

def handle_ambiguous_tuple(tup):
    if hasattr(tup, '__iter__'):
        return tuple(tup)
    else:
        return tuple([tup])


class DiffEqFitSystem:
    def __init__(self, diff_eq, parameters, initials,
                 timestep, interval, constant_params=None,
                 buffer_ratio=1.2, metadata={}):
        # buffer_ratio is how many times an out-of-bounds value
        # becomes the new interval limit when auto-adjusting range.
        self.parameters = handle_ambiguous_tuple(parameters)
        self.constant_params = handle_ambiguous_tuple(constant_params)
        self.DE = diff_eq
        self.y = []
        self.initials = handle_ambiguous_tuple(initials)
        self.timestep = timestep
        self.interval = (interval[0], interval[1] + timestep)
        self.times = np.arange(*self.interval, timestep)
        self.buffer_ratio = buffer_ratio
        self.metadata = metadata

        self.accesses = 0
        self.recalculations = 0

        # print(len(self.parameters + self.constant_params))

    def recalculate_values(self):
        self.recalculations += 1
        sol = integrate.solve_ivp(
            self.DE,
            self.interval,
            self.initials,
            t_eval=self.times,
            method='LSODA',
            args=(self.parameters + self.constant_params))  # + tuple(1))
        self.y = sol["y"].flatten()

    def get_values(self, tarr_in, *parameters):
        # print("i was called")
        '''
        Return the appropriate yarr out given tarr_in and parameters
        Recalculate data array if parameters have changed.
        '''
        self.accesses += 1
        try:
            if (handle_ambiguous_tuple(parameters) != tuple(self.parameters)
                    or np.max(tarr_in) > self.interval[1]):
                self.parameters = handle_ambiguous_tuple(parameters)
                self.interval = (self.interval[0],
                                 self.buffer_ratio * np.max(tarr_in))
                self.times = np.arange(*self.interval, self.timestep)
                self.recalculate_values()
        except:
            pass

        def get_value(t):
            index = tuple(np.where(np.abs(self.times - t)
                                   <= (self.timestep / 1.5)))
            self.accesses += 1
            return self.y[index]
# ERROR HAPPENING BECAUSE WE'RE RETURNING MULTIPLE - EASY TO FIX JUST PICK FIRST ONE
        if hasattr(tarr_in, '__iter__'):
            return np.vectorize(get_value)(tarr_in)
        else:
            return float(get_value(tarr_in))

    def get_system_stats(self):
        print("Array accessed {} times with {} recalculations"
              " (~{:.2f} acceses/calculation.)".format(
            self.accesses, self.recalculations,
            (self.accesses / self.recalculations)))
