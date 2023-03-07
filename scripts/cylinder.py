import numpy as np

class Cylinder:
    def __init__(self, length, diameter, mass, onecap=True):
        self.length = length
        self.diameter = diameter
        self.mass = mass
        self.volume = (np.pi * (diameter / 2)**2) * length

        self.area =  np.pi * (2 * ((diameter / 2)**2) + (diameter * length))

        if onecap:
            self.area -= (np.pi * (diameter / 2)**2)
