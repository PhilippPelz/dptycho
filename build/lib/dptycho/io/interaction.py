# -*- coding: utf-8 -*-
"""
Interaction module

Provides the server and a basic client to interact with it.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from __future__ import print_function
import time
import string
import random
import sys
from threading import Thread, Event
import Queue
import numpy as np
import re
import json
from .. import utils as u
from ..utils.verbose import logger

__all__ = ['Server', 'Client']

DEBUG = lambda x: None
#DEBUG = print

# Default parameters for networking
network_DEFAULT = u.Param(
    primary_address = "tcp://127.0.0.1",   # Default address for primary connection
    primary_port = 5560,            # Default port for primary connection
    port_range = range(5561,5571)   # Port range for secondary connections
)

# Default parameters for the server
Server_DEFAULT = u.Param(network_DEFAULT,
    poll_timeout = 10,   # Network polling interval (in milliseconds!)
    pinginterval = 2, # Interval to check pings (in seconds)
    pingtimeout = 10  # Ping time out: a client is disconnected if no news is sent within this period (in seconds)
)

# Default parameters for the client
Client_DEFAULT = u.Param(network_DEFAULT,
    poll_timeout = 100,   # Network polling interval (in milliseconds!)
    pinginterval = 1 # Interval to check pings (in seconds)
)


def ID_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """\
    Generate a random ID string made of capital letters and digits.
    size [default=6] is the length of the string.
    """
    return ''.join(random.choice(chars) for x in range(size))


def is_str(s):
    """\
    Test if s behaves like a string.
    """
    try:
        s + ''
        return True
    except:
        pass
    return False


class NumpyEncoder(json.JSONEncoder):
    """\
    Custom JSON Encoder class that take out numpy arrays from a structure
    and replace them with a code string.
    """
    def encode(self, obj):  
        # Prepare the array list
        self.npy_arrays = []
        
        # Encode as usual
        s = json.JSONEncoder.encode(self, obj)
        
        # Return the list along with the encoded object
        npy_arrays = self.npy_arrays
        del self.npy_arrays 
        return s, npy_arrays

    def default(self, obj):    
        if hasattr(obj, '__array_interface__'):
            # obj is "array-like". Add it to the list
            self.npy_arrays.append(obj)
            
            # Replace obj by a key string giving the index of obj in the list
            return u'NPYARRAY[%03d]' % (len(self.npy_arrays)-1)
            
        return json.JSONEncoder.default(self, obj)

NE = NumpyEncoder()

# This is the string to match against when decoding
NPYARRAYmatch = re.compile("NPYARRAY\[([0-9]{3})\]")


