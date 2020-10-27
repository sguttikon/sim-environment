#!/usr/bin/env python3
import numpy as np

def wrap_angle(angle) -> float:
    """
    Wrap the angle to [-np.pi, np.pi]
    """
    return ( (angle-np.pi) % (2*np.pi) ) - np.pi
