from __future__ import print_function

import sys
import time
import numpy as np
import galsim
import astropy.wcs as ap

#****************SUBROUTINES*****************

def Read_DC2_Tree_Ring_Model(Rx, Ry, Sx, Sy):
    """
    This function finds the tree ring parameters for a given sensor
    and assigns a tree ring model to that sensor.
    """

    def tree_ring_radial_function(r):
        """
        This function defines the tree ring distortion of the pixels as
        a radial function.

        Parameters
        ----------
        r: float
            Radial coordinate from the center of the tree ring structure
            in units of pixels.
        """
        centroid_shift = 0.0
        for j, fval in enumerate(cfreqs):
            centroid_shift += np.sin(2*np.pi*(r/fval)+cphases[j]) * fval / (2.0*np.pi)
        for j, fval in enumerate(sfreqs):
            centroid_shift += -np.cos(2*np.pi*(r/fval)+sphases[j]) * fval / (2.0*np.pi)
        centroid_shift *= (A + B * r**4) * .01 # 0.01 factor is because data is in percent
        return centroid_shift

    
    filename = 'csl_tree_ring_flats/tree_ring_parameters_2018-04-26.txt'
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    numfreqs = 20
    r_max = 8000.0
    dr = 3.0
    npoints = int(r_max - dr)
    cfreqs = np.zeros([numfreqs])
    cphases = np.zeros([numfreqs])    
    sfreqs = np.zeros([numfreqs])
    sphases = np.zeros([numfreqs])    
    for i, line in enumerate(lines):
        if line.split()[0] == 'Rx':
            items = lines[i+1].split()
            if int(items[0]) == Rx and int(items[1]) == Ry and int(items[2]) == Sx and int(items[3]) == Sy:
                Cx = float(items[4])
                Cy = float(items[5])
                A = float(items[6])
                B = float(items[7])                    
                for j in range(numfreqs):
                    freqitems = lines[i + 3 + j].split()
                    cfreqs[j] = float(freqitems[0])
                    cphases[j] = float(freqitems[1])                        
                    sfreqs[j] = float(freqitems[2])
                    sphases[j] = float(freqitems[3])
                tr_function = galsim.LookupTable.from_func(tree_ring_radial_function, x_min=0.0,\
                                                                   x_max=r_max, npoints=npoints)
                tr_center = galsim.PositionD(Cx, Cy)
                return (tr_center, tr_function)
            else:
                continue
        else:
            continue



# Put the salient numbers up here so they are easy to adjust.

nx = 509
ny = 2000
x_amp = 8
y_amp = 2

#(Rx, Ry) = (1,0)
#(Sx,Sy) = (2,0)
(Rx, Ry) = (2,2)
(Sx,Sy) = (1,0)

(treering_center, treering_func) = Read_DC2_Tree_Ring_Model(Rx, Ry, Sx, Sy)

seed = 31415


counts_total = 5000.0    #  80.e3    # 80K flats
counts_per_iter = 5.e3  # a few thousand is probably fine.  (bigger is faster of course.)
t0 = time.time()

rng = galsim.UniformDeviate(seed)

niter = 1#int(counts_total / counts_per_iter + 0.5)
counts_per_iter = counts_total / niter  # Recalculate in case not even multiple.
print('Total counts = {} = {} * {}'.format(counts_total,niter,counts_per_iter))

# Not an LSST wcs, but just make sure this works properly with a non-trivial wcs.
#wcs = galsim.FitsWCS('../../tests/fits_files/tnx.fits')

base_image = galsim.ImageF(nx*x_amp, ny*y_amp, scale = 0.2)#wcs=wcs)
base_image.setOrigin(-nx * x_amp / 2, -ny * y_amp / 2)
print('image bounds = ',base_image.bounds)

# nrecalc is actually irrelevant here, since a recalculation will be forced on each iteration.
# Which is really the point.  We need to set coundsPerIter appropriately so that the B/F effect
# doesn't change *too* much between iterations.
sensor = galsim.SiliconSensor(rng=rng,
                              treering_func=treering_func, treering_center=treering_center)

# We also need to account for the distortion of the wcs across the image.  
# This expects sky_level in ADU/arcsec^2, not ADU/pixel.
base_image.wcs.makeSkyImage(base_image, sky_level=1.)
base_image.write('wcs_area.fits')

# Rescale so that the mean sky level per pixel is skyCounts
mean_pixel_area = base_image.array.mean()

sky_level_per_iter = counts_per_iter / mean_pixel_area  # in ADU/arcsec^2 now.
base_image *= sky_level_per_iter

# The base_image has the right level to account for the WCS distortion, but not any sensor effects.
# This is the noise-free level that we want to add each iteration modulated by the sensor.

noise = galsim.PoissonNoise(rng)

t1 = time.time()
print('Initial setup time = ',t1-t0)

# Make flats
for i in range(x_amp):
    for j in range(y_amp):
        t1 = time.time()
        # image is the image that we will build up in steps.

        b = galsim.BoundsI(base_image.bounds.xmin + nx * i, base_image.bounds.xmin + nx * (i+1) - 1, base_image.bounds.ymin + ny * j, base_image.bounds.ymin + ny * (j+1) - 1)
        image = base_image[b]
        for k in range(niter):
            t2 = time.time()
            # temp is the additional flux we will add to the image in this iteration.
            # Start with the right area due to the sensor effects.
            temp = sensor.calculate_pixel_areas(image)
            #temp.write('sensor_area.fits')

            # Multiply by the base image to get the right mean level and wcs effects
            temp *= image 
            #temp.write('nonoise.fits')

            # Finally, add noise.  What we have here so far is the expectation value in each pixel.
            # We need to realize this according to Poisson statistics with these means.
            temp.addNoise(noise)
            #temp.write('withnoise.fits')

            # Add this to the image we are building up.
            image += temp
            t3 = time.time()
            print('Iter {}: time = {}'.format(k,t3-t2))

        t4 = time.time()
        print('Total time to make flat image with level {} = {}'.format(counts_total, t4-t1))

base_image.write('csl_tree_ring_flats/flat_full_%d_%d_%d_%d.fits'%(Rx,Ry,Sx,Sy))

t5 = time.time()
print('Total time to make flat image = {}'.format(t5-t0))


