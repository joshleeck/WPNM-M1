# -*- coding: utf-8 -*-
"""
Author: Joshua Lee (MSS/CCRS)
Updated: 03/05/2019 - Coded skeleton
         04/09/2019 - Added metric for total mass

Extended based on a practical by Dr Hilary Weller
"""

import numpy as np

def L2ErrorNorm(phiNumerical, phiAnalytical):
    """
    Calculates the L2 error norm (RMS error)
    """

    # calculate the error and the error norms
    phiError = phiNumerical - phiAnalytical
    L2 = np.sqrt(sum(phiError**2)/sum(phiAnalytical**2))

    return L2, phiError

def compute_mass(phi, label):
    """
    Calculates the total mass of phi and prints it
    """
    phitotal=sum(phi)
    print('%s Total Mass: %s' % (label, phitotal))

# Potential to code other metrics for assessment of solution here