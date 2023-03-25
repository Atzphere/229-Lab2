import numpy as np
from cylinder import Cylinder


class Trial:
    def __init__(self, filename, ydelta, y_conversion=False,
                 offset_loc=0, offset_amt=0):
        self.ydelta = ydelta
        self.offset_loc = offset_loc
        self.offset_amt = offset_amt

        data = np.loadtxt(filename, delimiter=",",
                          comments="#", skiprows=1)
        self.times = data[:, 0]

        if y_conversion:
            self.y = y_conversion(data[:, 1])
        else:
            self.y = data[:, 1]

        self.time_interval = (np.min(self.times), np.max(self.times))

        self.offset_times = np.copy(self.times)
        # print(self.offset_times)
        self.offset_times[
            self.offset_loc:len(self.offset_times)] += self.offset_amt
        self.initial_y = [self.y[0]]
        # print(type(self.y[0].tolist()))


    def set_offset(self, loc, offset):
        self.offset_loc = loc
        self.offset_amt = offset
        self.offset_times = np.copy(self.times)
        self.offset_times[loc:len(self.offset_times)] += offset

    def chunked(self, chunksize):
        chunks = [(self.y[i:i + chunksize], self.times[i:i + chunksize])
                  for i in range(0, len(self.y), chunksize)]
        y, deltas, t = [], [], []
        for ys, times in chunks:
            y.append(np.mean(ys))
            deltas.append(np.std(ys))
            t.append(np.mean(times))
        return np.array(y), np.array(deltas), np.array(t)
