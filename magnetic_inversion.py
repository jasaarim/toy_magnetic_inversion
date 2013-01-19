#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Jarno Saarimäki
"""
.. _magnetic_inversion

This script implements a simple adaptive Gaussian inversion algorithm
to fit an infitely extended rectangular body to one dimensional
magnetic field profile.

The inversion algorithm is adapted from William Menke's (1988)
'Geophysical data analysis: discrete inverse theory' and the forward
model computation from Richand Blakely's (1995) 'Potential theory in
gravity and magnetic applications'.
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
matplotlib.interactive(True)
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from scipy.integrate import dblquad
import functools, time
# Import the integrand from the compiled Fortran code
try:    import integrand as integr
except: print('Compile the fortran code with f2py to "integrand"')

#: Dimensions of the cross-section of the elongated body that is
#: fitted to the data
body_dim = (20,50)
#: Initial models parameters (susceptibility,horizontal distance, depth)
inv_param = [0.3,175,20]

#: How much model parameters are perturbed in each dimension to
#: compute the derivatives
perturb = 1e-5*np.ones(3)
#: Maximum number of iterations
maxiter = 15
#: When the norm of the model parameters changes less than this
#: stop the iteration
delta = 1e-1


def response_point(F_hat,F_abs,x,y):
    """
    Function that is integrated over the cross-section of the fitted body.
    
    :param F_hat: unit vector of the ambient magnetic field.
    :param F_abs: magnitude of the ambient magnetic field    
    :param x: horizontal distance
    :param y: depth    
    
    .. note:: It is preferred to use a compiled version of this
              function because otherwise the computations get very
              slow. The included Fortran code can be compiled with
              f2py (command: ``f2py -c -m integrand integrand.f``).
    """
    r_abs = np.sqrt(x**2 + y**2)
    r_hat = np.array([x,y]) / r_abs
    response =  F_abs / (2*np.pi) * \
                (2*(np.dot(r_hat,F_hat))**2 -1) / r_abs**2 
    return response

def response_body(body_dim, points, F_hat, F_abs, inv_param):
    """
    Integrate the response function over the cross-section. The response
    is derived, e.g., in Blakely (1995).
    
    :param block_dim: (width, height)
    :param F_hat: unit vector of the ambient magnetic field.
    :param F_abs: magnitude of the ambient magnetic field
    :param inv_param: inversion parameters (susc,x,y)
    """
    # Geometrical coefficients for each measurements point
    G = []
    integrand = functools.partial(integr.response_point,F_hat,F_abs)
    for point in points:
       G.append( dblquad(integrand,
                         inv_param[1]-point,
                         inv_param[1]-point+body_dim[0] ,
                         lambda x: inv_param[2],
                         lambda x: inv_param[2]+body_dim[1] )[0] )

    # Convert the geometrical factors to magnetic field values
    # according to the susceptibility of the block
    return np.array(G)*inv_param[0]


def init_plot(distance, mag_data, inv_param, block_dim):
    """Make the initial plot"""
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(distance,mag_data,'g',lw=3,label='measurements')
    ax1.plot(distance,mag_data,'--r',lw=2, label='model response')
    ax1.set_ylabel('Anomaly [nT]')
    ax1.legend()
    ax1.autoscale(enable=False)
    
    ax2 = fig.add_subplot(212)
    # Plot the rectangle
    x = [ inv_param[1] ,
          inv_param[1] ,
          inv_param[1]+block_dim[0],
          inv_param[1]+block_dim[0] ]
    y = [ inv_param[2],
          inv_param[2]+block_dim[1],
          inv_param[2]+block_dim[1],
          inv_param[2] ]
    c = np.array([0,0,0])
    ax2.fill(x,y,color=c)
    ax2.axis([distance[0],distance[-1],150,0])
    ax2.set_ylabel('Depth [m]')
    ax2.set_xlabel('Horizontal distance on the profile')    
    ax2.autoscale(enable=False)
    
    ax2.text(250,100,'Iteration %i\nSusc: %.3f ' % (0,inv_param[0]) )    
    
    fig.suptitle('Measured magnetic field and a fitted elongated body\n' +
                 'Estimating 3 model parameters')
    plt.draw()

    return fig,ax1,ax2

def update_plots(ax1, ax2, new_resp, inv_param, block_dim):
    """Update the plots"""
    ax1.lines[1].set_ydata(new_resp)
    # Plot the rectangle
    x = [ inv_param[1],
          inv_param[1],
          inv_param[1]+block_dim[0],
          inv_param[1]+block_dim[0] ]
    y = [ inv_param[2],
          inv_param[2]+block_dim[1],
          inv_param[2]+block_dim[1],
          inv_param[2] ]
    # c = np.array([1,1,1])*( 1 - m[0]/max_susc)
    c = np.array([0,0,0])
    ax2.patches[0].remove()
    ax2.fill(x,y,color=c)[0]
    ax2.texts[0].set_text('Iteration %i\nSusc: %.3f ' % (i+1,inv_param[0]) )
    plt.draw()

def optimize_params(response,mag_data,inv_param,perturb):
    """
    Optimize the inversion paramters with the current state value
    and assuming a linear response.
    """
    # Response with current model parameters
    cur_resp = response(inv_param)
    # Matrix of the partial derivatives w.r.t. all model
    # parameters
    J = np.zeros( (len(cur_resp), 3) )
    # Estimate the partial derivatives 
    for j in range(3):
        param_perturb = np.zeros(3)
        param_perturb[j] = perturb[j]
        new_resp = response(inv_param+param_perturb)
        J[:,j] = ( new_resp - cur_resp ) / param_perturb[j]
    # Assuming that the response is linear (i.e., the derivatives
    # define the response), estimate the optimal model parameters
    inverted =  np.linalg.inv( np.dot( J.T , J ) ) 
    change_mat = np.dot( inverted , J.T )
    m_change = np.dot( change_mat , (mag_data - cur_resp) )
    inv_param += m_change

    # perform some checks for the new parameters and change them
    # accordingly
    if inv_param[0] < 1e-5:  inv_param[0] = 1e-5
    
    if inv_param[1] < 130:   inv_param[1] = 130
    elif inv_param[1] > 180: inv_param[1] = 180
    
    if inv_param[2] < 5:     inv_param[2] = 5
    elif inv_param[2] > 50:  inv_param[2]= 50

    return cur_resp, new_resp, inv_param, m_change

def load_data(filename):
    """Load the total magnetic field data."""
    data = np.loadtxt(filename)
    # Estimate the ambient field value as the mean of the measurements
    F_abs = np.mean(data[:,1])
    # Fluctuation around the mean
    mag_data = data[:,1]- F_abs
    # Measurement positions along the profile
    distance = data[:,0]

    # remove the outlier at 165 meters
    mag_data = mag_data[distance != 165]
    distance = distance[distance != 165]

    # Use inclication 75 degrees (no need for declination because the
    # profile is from S-N) The ambient field vector.
    # FIXME: this isn't correct! Probably bug in the point_response. 
    I = 0
    #I = 75/180*np.pi
    F = F_abs*np.array([np.cos(I), np.sin(I)])
    F_hat = F / F_abs

    return distance,mag_data,F_abs, F_hat

if __name__ == '__main__':

    # Load the data
    distance, mag_data, F_abs, F_hat = load_data('measurements')
    # Initialize the plot
    fig,ax1,ax2 = init_plot(distance, mag_data, inv_param, body_dim)
    print('starting the iteration to find suitable model parameters')
    # Give the response as a function our model parameters
    response = functools.partial(response_body,body_dim,distance,F_hat,F_abs)
    for i in range(maxiter):
        print('Iteration step %d/%d' % (i+1,maxiter))
        # Optimize the inversion parameters
        cur_resp, new_resp, inv_param, d_inv_param = \
            optimize_params(response,mag_data,inv_param,perturb)
        # Update the plots
        update_plots(ax1, ax2, new_resp, inv_param, body_dim)
        # Stop the iteration if the change is smaller than delta
        if np.linalg.norm(d_inv_param) < delta:
            break
        # Wait for a while
        time.sleep(1)
