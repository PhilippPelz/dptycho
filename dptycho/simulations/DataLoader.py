'''
Created on Feb 1, 2016

@author: philipp
'''
from ..io import h5rw

class DataLoader(object):
    '''
    classdocs
    '''
    

    def __init__(self, filename):
        '''
        Constructor
        '''
        f = h5rw.h5read(filename)
        self.measurements = f['measurements']
        self.positions = f['positions']
        self.probe = f['probe']
        self.propagator = f['propagator']
        self.Znums = f['Znums']
        
        