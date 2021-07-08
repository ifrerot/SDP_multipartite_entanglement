from ncpol2sdpa import generate_variables
from math import *
from scipy import integrate
from numpy import pi,real,conj,imag


def generate_commuting_measurements(party, label):
    """Genrates the list of symbolic variables representing the measurements
    for a given party. The variables are treated as commuting.

    :param party: configuration indicating the configuration of number m
                  of measurements and outcomes d for each measurement. It is a
                  list with m integers, each of them representing the number of
                  outcomes of the corresponding  measurement.
    :type party: list of int
    :param label: label to represent the given party
    :type label: str

    :returns: list of sympy.core.symbol.Symbol
    """
    measurements = []
    for i, p in enumerate(party):
        measurements.append(generate_variables(label + '%s' % i, p - 1,
                                               hermitian=True))
    return measurements


def get_phi(N,r,t):
    """Computes the value phi_r(t) needed for the estimation of correlation on the
    1D quench, as for Eq. (13) in arxiv.????
    
    :param N: number of parties for the global state
    :type N: int
    :param r: distance r = |i-j| between the particles i,j in the lattice
    :type r: int
    :param t: time after the quench at t = 0
    :type t: float
    
    :returns: the value phi_r(t) as complex
    """
    
    ex_com_re = lambda k: cos(k*r + t*cos(k))
    ex_com_im = lambda k: sin(k*r + t*cos(k))
    
    res = sum([ex_com_re(2*pi*k/N) for k in range(N)])+ 1j*sum([ex_com_im(2*pi*k/N) for k in range(N)])
    
    return res/N


def get_moment_quench(N,t,measurements,lambda_=None,full_info = False):
    """Generates the list of moment equalities substitution for the sdp
    relaxation for the considered 1D quench expereriment,
    that is, the values of the one- and two-body correlations from Eq. (13) 
    in arxiv.????. Optional: adding the noise parameter lambda_ 
    
    :param N: number of particles.
    :type N: int
    :param t: time after the quench at t = 0
    :type t: float
    :param measurements: list of commuting variables representing the local
                         spin operators in the relaxation.
    :type measurements: list of sympy.core.symbol.Symbol
    :param lambda: symbolic variable corresponding to the amount of white noise
                   added to the state. Optional, in case one is interested in
                   computing the constraints for the noiseless case.
    :type lambda_: sympy.core.symbol.Symbol
    
    :param full_info: option for selecting which kind of correlations is
                      is observed. If set to True, it adds the XY,XZ,YZ correlation
                      to the list of Eq. (13)
    :type full_info: bool
    
    :returns: moments: list of sympy.core.add.Add
    """
    
    moments = {}
    
    # setting the noise parameter to zero unless specified
    if lambda_ is None:

        lambda_ = 0

    # generating all the one-body expectation values
    for k1, party1 in enumerate(measurements):
                
                z = party1[0][0]
                x = party1[1][0]
                y = party1[2][0]
                moments[z] =  (1 - lambda_)*(1 - 2*abs(get_phi(N,k1,t))**2)
                
                moments[x] = 0
                moments[y] = 0

    # generating all the two-body correlators

    for k1, party1 in enumerate(measurements):
        z1 = party1[0][0]
        x1 = party1[1][0]
        y1 = party1[2][0]
        moments[x1*y1] = 0
        moments[y1*x1] = 0 
        moments[z1*x1] = 0
        moments[x1*z1] = 0
        moments[z1*y1] = 0
        moments[y1*z1] = 0
        for k2, party2 in enumerate(measurements[k1 + 1:], start=k1 + 1):
                
            z2 = party2[0][0]
            x2 = party2[1][0]
            y2 = party2[2][0]

            moments[z1*z2] =  (1 - lambda_)*(1 - 2*(abs(get_phi(N,k1,t))**2 + abs(get_phi(N,k2,t))**2))
            moments[x1*x2] =  (1 - lambda_)*2*real(get_phi(N,k1,t)*conj(get_phi(N,k2,t)))
            moments[y1*y2] =  (1 - lambda_)*2*real(get_phi(N,k1,t)*conj(get_phi(N,k2,t)))
            
            moments[x1*y2] = 0
            moments[y1*x2] = 0 
            moments[z1*x2] = 0
            moments[x1*z2] = 0
            moments[z1*y2] = 0
            moments[y1*z2] = 0

            
            if full_info is True:
                
                moments[x1*y2] = -(1 - lambda_)*2*imag(get_phi(N,k1,t)*conj(get_phi(N,k2,t)))
                moments[y1*x2] =  (1 - lambda_)*2*imag(get_phi(N,k1,t)*conj(get_phi(N,k2,t)))
                moments[z1*x2] = 0
                moments[x1*z2] = 0
                moments[z1*y2] = 0
                moments[y1*z2] = 0
                                
    return moments