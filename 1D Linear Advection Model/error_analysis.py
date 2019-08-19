# -*- coding: utf-8 -*-
"""
Author: Joshua Lee (MSS/CCRS)
Updated: 03/05/2019 - Coded skeleton

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

# Potential to code other metrics for assessment of solution here