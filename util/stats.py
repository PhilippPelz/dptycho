import scipy.stats as stats
import numpy as np

def truncnorm(shape,a,b,mu,sigma):
    T = stats.truncnorm(
        (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    return T.rvs(shape)

def percentile(a,q):
    return np.percentile(a,q)

# a, b = 3.5, 6
# mu, sigma = 5, 0.7
# a= truncnorm([1024,1024],a,b,mu,sigma)
# print(a.shape)
# print(np.mean(a))

def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)

    A function to reduce an image to a radial cross-section.

    :INPUT:
      data   - whatever data you are radially averaging.  Data is
              binned into a series of annuli of width 'annulus_width'
              pixels.

      annulus_width - width of each annulus.  Default is 1.

      working_mask - array of same size as 'data', with zeros at
                        whichever 'data' points you don't want included
                        in the radial data computations.

      x,y - coordinate system in which the data exists (used to set
               the center of the data).  By default, these are set to
               integer meshgrids

      rmax -- maximum radial value over which to compute statistics

    :OUTPUT:
        r - a data structure containing the following
                   statistics, computed across each annulus:

          .r      - the radial coordinate used (outer edge of annulus)

          .mean   - mean of the data in the annulus

          .sum    - the sum of all enclosed values at the given radius

          .std    - standard deviation of the data in the annulus

          .median - median value in the annulus

          .max    - maximum value in the annulus

          .min    - minimum value in the annulus

          .numel  - number of elements in the annulus

    :EXAMPLE:
      ::

        import numpy as np
        import pylab as py
        import radial_data as rad

        # Create coordinate grid
        npix = 50.
        x = np.arange(npix) - npix/2.
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2)
        fake_psf = np.exp(-(r/5.)**2)
        noise = 0.1 * np.random.normal(0, 1, r.size).reshape(r.shape)
        simulation = fake_psf + noise

        rad_stats = rad.radial_data(simulation, x=xx, y=yy)

        py.figure()
        py.plot(rad_stats.r, rad_stats.mean / rad_stats.std)
        py.xlabel('Radial coordinate')
        py.ylabel('Signal to Noise')
    """

# 2012-02-25 20:40 IJMC: Empty bins now have numel=0, not nan.
# 2012-02-04 17:41 IJMC: Added "SUM" flag
# 2010-11-19 16:36 IJC: Updated documentation for Sphinx
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory

    import numpy as ny

    class radialDat:
        """Empty object container.
        """
        def __init__(self):
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    #---------------------
    # Set up input parameters
    #---------------------
    data = ny.array(data)

    if working_mask==None:
        working_mask = ny.ones(data.shape,bool)

    npix, npiy = data.shape
    if x==None or y==None:
        x1 = ny.arange(-npix/2.,npix/2.)
        y1 = ny.arange(-npiy/2.,npiy/2.)
        x,y = ny.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = ny.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = ny.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = ny.zeros(nrad)
    radialdata.sum = ny.zeros(nrad)
    radialdata.std = ny.zeros(nrad)
    radialdata.median = ny.zeros(nrad)
    radialdata.numel = ny.zeros(nrad, dtype=int)
    radialdata.max = ny.zeros(nrad)
    radialdata.min = ny.zeros(nrad)
    radialdata.r = radial

    #---------------------
    # Loop through the bins
    #---------------------
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      #import pylab as py
      #pdb.set_trace()
      if not thisindex.ravel().any():
        radialdata.mean[irad] = ny.nan
        radialdata.sum[irad] = ny.nan
        radialdata.std[irad]  = ny.nan
        radialdata.median[irad] = ny.nan
        radialdata.numel[irad] = 0
        radialdata.max[irad] = ny.nan
        radialdata.min[irad] = ny.nan
      else:
        radialdata.mean[irad] = data[thisindex].mean()
        radialdata.sum[irad] = data[r<maxrad].sum()
        radialdata.std[irad]  = data[thisindex].std()
        radialdata.median[irad] = ny.median(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size
        radialdata.max[irad] = data[thisindex].max()
        radialdata.min[irad] = data[thisindex].min()

    #---------------------
    # Return with data
    #---------------------

    return radialdata
