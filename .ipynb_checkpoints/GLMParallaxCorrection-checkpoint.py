#############################################################################################
#Routine to correct GLM Level 3 Gridded Data Parallax inherent to satellite data.
#GLM L3 data do not account for this correction as it's primary use is to be overlaid
#onto other satellite datasets. However, to align these data to ground-based systems
#such as radar data, one must account for the cloud-top height parallax issues when
#presented with satallite imagery. This script does this correction, based on the
#notebook for parallax correction in glmtools:
#https://github.com/deeplycloudy/glmtools/blob/master/examples/parallax-corrected-latlon.ipynb
#
# Original code author: Eric Bruning
#############################################################################################

#----------------
#Required Imports
#----------------
import pyproj as proj4
from lmatools.coordinateSystems import CoordinateSystem
from lmatools.grid.fixed import get_GOESR_coordsys
import numpy as np


#------------------
#Lightning Ellipse:
#------------------
def lightning_ellipse(key):
    # equatorial, polar radii
    lightning_ellipse_rev = {
        # Values at launch
        0: (6.394140e6, 6.362755e6),

        # DO.07, late 2018. First Virts revision.
        # The GRS80 altitude + 6 km differs by about 3 m from the value above
        # which is the exact that was provided at the time of launch. Use the
        # original value instead of doing the math.
        # 6.35675231414e6+6.0e3
        1: (6.378137e6 + 14.0e3, 6.362755e6),
    }
    return(lightning_ellipse_rev[key])
#---------
#Routines:
#---------
def semiaxes_to_invflattening(semimajor, semiminor):
    """ Calculate the inverse flattening from the semi-major
        and semi-minor axes of an ellipse"""
    rf = semimajor/(semimajor-semiminor)
    return rf

class GeostationaryFixedGridSystemAltEllipse(CoordinateSystem):

    def __init__(self, subsat_lon=0.0, subsat_lat=0.0, sweep_axis='y',
                 sat_ecef_height=35785831.0,
                 semimajor_axis=None,
                 semiminor_axis=None,
                 datum='WGS84'):
        """
        Satellite height is with respect to an arbitray ellipsoid whose
        shape is given by semimajor_axis (equatorial) and semiminor_axis(polar)

        Fixed grid coordinates are in radians.
        """
        rf = semiaxes_to_invflattening(semimajor_axis, semiminor_axis)
        # print("Defining alt ellipse for Geostationary with rf=", rf)
        self.ECEFxyz = proj4.Proj(proj='geocent',
            a=semimajor_axis, rf=rf)
        self.fixedgrid = proj4.Proj(proj='geos', lon_0=subsat_lon,
            lat_0=subsat_lat, h=sat_ecef_height, x_0=0.0, y_0=0.0,
            units='m', sweep=sweep_axis,
            a=semimajor_axis, rf=rf)
        self.h=sat_ecef_height

    def toECEF(self, x, y, z):
        X, Y, Z = x*self.h, y*self.h, z*self.h
        return proj4.transform(self.fixedgrid, self.ECEFxyz, X, Y, Z)

    def fromECEF(self, x, y, z):
        X, Y, Z = proj4.transform(self.ECEFxyz, self.fixedgrid, x, y, z)
        return X/self.h, Y/self.h, Z/self.h

class GeographicSystemAltEllps(CoordinateSystem):
    """
    Coordinate system defined on the surface of the earth using latitude,
    longitude, and altitude, referenced by default to the WGS84 ellipse.

    Alternately, specify the ellipse shape using an ellipse known
    to pyproj, or [NOT IMPLEMENTED] specify r_equator and r_pole directly.
    """
    def __init__(self, ellipse='WGS84', datum='WGS84',
                 r_equator=None, r_pole=None):
        if (r_equator is not None) | (r_pole is not None):
            rf = semiaxes_to_invflattening(r_equator, r_pole)
            # print("Defining alt ellipse for Geographic with rf", rf)
            self.ERSlla = proj4.Proj(proj='latlong', #datum=datum,
                                     a=r_equator, rf=rf)
            self.ERSxyz = proj4.Proj(proj='geocent', #datum=datum,
                                     a=r_equator, rf=rf)
        else:
            # lat lon alt in some earth reference system
            self.ERSlla = proj4.Proj(proj='latlong', ellps=ellipse, datum=datum)
            self.ERSxyz = proj4.Proj(proj='geocent', ellps=ellipse, datum=datum)
    def toECEF(self, lon, lat, alt):
        projectedData = np.array(proj4.transform(self.ERSlla, self.ERSxyz, lon, lat, alt ))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0,:], projectedData[1,:], projectedData[2,:]

    def fromECEF(self, x, y, z):
        projectedData = np.array(proj4.transform(self.ERSxyz, self.ERSlla, x, y, z ))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0,:], projectedData[1,:], projectedData[2,:]

def get_GOESR_coordsys_alt_ellps(sat_lon_nadir,ltg_ellps_re,ltg_ellps_rp):
    #nadir = -75
    goes_sweep = 'x' # Meteosat is 'y'
    datum = 'WGS84'
    sat_ecef_height=35786023.0
    geofixcs = GeostationaryFixedGridSystemAltEllipse(subsat_lon=sat_lon_nadir,
                    semimajor_axis=ltg_ellps_re, semiminor_axis=ltg_ellps_rp,
                    datum=datum, sweep_axis=goes_sweep,
                    sat_ecef_height=sat_ecef_height)
    grs80lla = GeographicSystemAltEllps(r_equator=ltg_ellps_re, r_pole=ltg_ellps_rp,
                                datum='WGS84')
    return geofixcs, grs80lla

#-----------------
#Centers to Edges:
#-----------------
def centers_to_edges_2d(x):
    """
    Create a (N+1, M+1) array of edge locations from a
    (N, M) array of grid center locations.

    In the interior, the edge positions set to the midpoints
    of the values in x. For the outermost edges, half the
    closest dx is assumed to apply. This matters for polar
    meshes, where one edge of the grid becomes a point at the
    polar coordinate origin; dx/2 is a half-hearted way of
    trying to prevent negative ranges.
    Useful when plotting with pcolor, which requires
    X, Y of shape (N+1) and grid center values of shape (N).
    Otherwise, pcolor silently discards the last row and column
    of grid center values.

    Parameters
    ----------
    x : array, shape (N,M)
        Locations of the centers

    Returns
    -------
    xedge : array, shape (N+1,M+1)

    """
    xedge = np.zeros((x.shape[0]+1,x.shape[1]+1))
    # interior is a simple average of four adjacent centers
    xedge[1:-1,1:-1] = (x[:-1,:-1] + x[:-1,1:] + x[1:,:-1] + x[1:,1:])/4.0

    #         /\
    #        /\/\
    #       / /\ \
    #      /\/  \/\
    #     / /\  /\ \
    #    /\/  \/  \/\
    #   / /\  /\  /\ \
    #  /\/  \/  \/  \/\
    #4 \/\  /\  /\  /\/ 4
    # 3 \ \/  \/  \/ / 3
    #    \/\  /\  /\/
    #   2 \ \/  \/ / 2
    #      \/\  /\/
    #     1 \ \/ / 1
    #        \/\/
    #       0 \/ 0 = center ID of 0th dimension
    #

    # calculate the deltas along each edge, excluding corners
    xedge[1:-1,0] = xedge[1:-1, 1] - (xedge[1:-1, 2] - xedge[1:-1, 1])/2.0
    xedge[1:-1,-1]= xedge[1:-1,-2] - (xedge[1:-1,-3] - xedge[1:-1,-2])/2.0
    xedge[0,1:-1] = xedge[1,1:-1]  - (xedge[2,1:-1]  - xedge[1,1:-1])/2.0
    xedge[-1,1:-1]= xedge[-2,1:-1] - (xedge[-3,1:-1] - xedge[-2,1:-1])/2.0

    # now do the corners
    xedge[0,0]  = xedge[1, 1] - (xedge[2, 2] - xedge[1, 1])/2.0
    xedge[0,-1] = xedge[1,-2] - (xedge[2,-3] - xedge[1,-2])/2.0
    xedge[-1,0] = xedge[-2,1] - (xedge[-3,2] - xedge[-2,1])/2.0
    xedge[-1,-1]= xedge[-2,-2]- (xedge[-3,-3]- xedge[-2,-2])/2.0

    return xedge

def glm_non_corrected(nadir,x,y,z):
    '''Get the lon,lat coordinates of the GLM L3 data
       that do not correct for parallax.
    '''
    geofixCS,grs80lla = get_GOESR_coordsys(nadir)
    lon,lat,alt       = grs80lla.fromECEF(*geofixCS.toECEF(x,y,z))
    return(lon,lat,alt,geofixCS,grs80lla)

def glm_parallax_corrected(nadir,x,y,z,ltg_ellps_re,ltg_ellps_rp,grs80lla):
    '''Get the lon,lat coordinates of the GLM L3 datasets
       with an applied parallax corretion.

       Makes use of a lightning ellipsoid to define the coordinates
       where a pixel intersects the MSL and Lighting ellipsoids.
    '''
    #Now new Earth Ellipsoid whos surface is that of the Lightning Ellipsoid
    #------------------------
    geofix_ltg, lla_ltg = get_GOESR_coordsys_alt_ellps(nadir,ltg_ellps_re,ltg_ellps_rp)

    lon_ltg0,lat_ltg0,alt_ltg0=lla_ltg.fromECEF(*geofix_ltg.toECEF(x,y,z))
    lon_ltg0.shape = x.shape
    lat_ltg0.shape = y.shape

    x_ltg, y_ltg, z_ltg = geofix_ltg.fromECEF(*lla_ltg.toECEF(lon_ltg0,lat_ltg0,alt_ltg0))

    #Find lat,lon,alt of intersection point wrt to MSL ellipsoid
    #--------------------------
    lon_ltg,lat_ltg,alt_ltg=grs80lla.fromECEF(*geofix_ltg.toECEF(x,y,z))
    lon_ltg.shape = x.shape
    lat_ltg.shape = y.shape
    return(lon_ltg,lat_ltg)

def coordinate_correction(glm_data,geo,proj,ltg_ellps_re,ltg_ellps_rp):
    '''Takes the coordinate (radian) positions of glm l3
       pixels and corrects them for parallax.

       glm_data is an xarray dataset object.

    '''
    #Get GLM Non-corrected Coordinates:
    x_1d  = glm_data.x
    y_1d  = glm_data.y
    x,y   = np.meshgrid(x_1d, y_1d) # Two 2D arrays of fixed grid coordinates
    z     = np.zeros_like(x)
    nadir = glm_data.nominal_satellite_subpoint_lon.values[0]
    #Get lon,lat un-corrected coordinates
    lon,lat,alt,geofixCS,grs80lla = glm_non_corrected(nadir,x,y,z)
    lon_ltg,lat_ltg               = glm_parallax_corrected(nadir,x,y,z,ltg_ellps_re,ltg_ellps_rp,grs80lla)
    lon_ltg_edge                  = centers_to_edges_2d(lon_ltg)
    lat_ltg_edge                  = centers_to_edges_2d(lat_ltg)
    lon_edge                      = centers_to_edges_2d(lon)
    lat_edge                      = centers_to_edges_2d(lat)

    #Test transformation onto OKLMA domain system:
    gx,gy,_   = proj.fromECEF(*geo.toECEF(lon_ltg,lat_ltg,lat_ltg*0))
    gx0,gy0,_ = proj.fromECEF(*geo.toECEF(lon    ,lat    ,lat    *0))
    return(gx,gy)
