###########################################
# Runs xlma-python (pyxlma) to grid flash
# sorted LMA data.
#
# Note: Grid Specifications (i.e., spacing,
# domain size, and time frame duration) are
# defined in the FlashGriddingParams.txt file
# and should match that used to grid the GLM
# Level 2 files via glmtools.
###########################################
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import glob
import datetime as dt
import sys
sys.path.append('../../')
import backend as bk

from scipy.interpolate import griddata as gd
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter as gf

from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import cartopy as ct

from lmatools import coordinateSystems as cs
from pyxlma.lmalib.grid import create_regular_grid, assign_regular_bins, events_to_grid
from pyxlma.lmalib import grid


def grid_setup(lon_ranges,lat_ranges,alt_ranges,spacing,case,start,end,grid_dt,xover,grid_3d):
    '''
    Set up LMA gridding from parameter file.
    Arguments:
        -) lon_range = longitude extents for gridding,
        -) lat_range = latitude extents for gridding,
        -) spacing   = lon,lat grid spacing in degrees,
        -) case      = selected case to create datetime time frame objects,
        -) start     = desired start time string (in HHMMSS format),
        -) end       = desired end time string (in HHMMSS format),
        -) grid_dt   = time frame time spacing (in seconds)
    '''
    #GRID SETUP:
    # Change the dictionaries below to a consistent set of coordinates
    # and adjust grid_spatial_coords in the call to events_to_grid to
    # change what is gridded (various time series of 1D, 2D, 3D grids)
    if grid_3d == False:
        event_coord_names = {
            'event_latitude':'grid_latitude_edge',
            'event_longitude':'grid_longitude_edge',
            'event_time':'grid_time_edge',
            }
        flash_ctr_names = {
            'flash_init_latitude':'grid_latitude_edge',
            'flash_init_longitude':'grid_longitude_edge',
            'flash_time_start':'grid_time_edge',
            }
        flash_init_names = {
            'flash_center_latitude':'grid_latitude_edge',
            'flash_center_longitude':'grid_longitude_edge',
            'flash_time_start':'grid_time_edge',
            }
    else:
        event_coord_names = {
            'event_altitude':'grid_altitude_edge',
            'event_latitude':'grid_latitude_edge',
            'event_longitude':'grid_longitude_edge',
            'event_time':'grid_time_edge',
            }
        flash_ctr_names = {
            'flash_init_altitude':'grid_altitude_edge',
            'flash_init_latitude':'grid_latitude_edge',
            'flash_init_longitude':'grid_longitude_edge',
            'flash_time_start':'grid_time_edge',
            }
        flash_init_names = {
            'flash_center_altitude':'grid_altitude_edge',
            'flash_center_latitude':'grid_latitude_edge',
            'flash_center_longitude':'grid_longitude_edge',
            'flash_time_start':'grid_time_edge',
            }

    # Oklahoma -- Can define any grid spacing -- chose a grid spacing of 0.05 deg
    lat_range = (lat_ranges[0],lat_ranges[1],spacing)
    lon_range = (lon_ranges[0],lon_ranges[1],spacing)
    if grid_3d == True:
        alt_range = (alt_ranges[0],alt_ranges[1],alt_ranges[2])
        print('Grid Alt dimension size: {0}'.format(np.arange(alt_range[0],alt_range[1],alt_range[2]).shape))
    print('Grid Lat dimension size: {0}'.format(np.arange(lat_range[0],lat_range[1],lat_range[2]).shape))
    print('Grid Lon dimension size: {0}'.format(np.arange(lon_range[0],lon_range[1],lon_range[2]).shape))

    grid_dt = np.asarray(dtime, dtype='m8[s]') #300 s windows
    # grid_t0 = flash_itime
    # grid_t1 = flash_ftime

    #Set up start and end times
    #added 03/28/22--shift day over if day crossover exists (set in param file)
    if xover == True:
        shift_day = 1
    else:
        shift_day = 0
    f_itime = np.datetime64(dt.datetime(int(case[:4]),int(case[4:6]),int(case[6:]),
                                        int(start[:2]),int(start[2:4]),int(start[4:])))
    f_ftime = np.datetime64(dt.datetime(int(case[:4]),int(case[4:6]),int(case[6:])+shift_day,
                                        int(end[:2]),int(end[2:4]),int(end[4:])))
    time_range = (f_itime, f_ftime+grid_dt, grid_dt)
    print('Number of time frames to grid: {0}'.format(np.arange(f_itime,f_ftime+grid_dt,grid_dt).shape))

    if grid_3d == False:
        grid_edge_ranges ={
            'grid_latitude_edge' :lat_range,
            'grid_longitude_edge':lon_range,
            'grid_time_edge'     :time_range,
            }
        grid_center_names ={
            'grid_latitude_edge':'grid_latitude',
            'grid_longitude_edge':'grid_longitude',
            'grid_time_edge':'grid_time',
            }
    else:
        grid_edge_ranges ={
            'grid_altitude_edge' :alt_range,
            'grid_latitude_edge' :lat_range,
            'grid_longitude_edge':lon_range,
            'grid_time_edge'     :time_range,
            }
        grid_center_names ={
            'grid_altitude_edge' :'grid_altitude',
            'grid_latitude_edge':'grid_latitude',
            'grid_longitude_edge':'grid_longitude',
            'grid_time_edge':'grid_time',
            }
    return(grid_edge_ranges,grid_center_names,event_coord_names)



if __name__ == '__main__':
    desired_dt  = '5min' # other options are 5min and 10min
    #Read in gridding parameters--shared across GLM and LMA gridding routines
    target      =f'../../CaseParams/{desired_dt}/FlashGriddingParams_20230303.txt'
    params      = eval(open(target, 'r').read())
    case        = params['Case']
    network     = params['network']
    network_lon = params['network_lon']
    network_lat = params['network_lat']
    lat_ranges  = params['lat_range']
    lon_ranges  = params['lon_range']
    spacing     = params['spacing']
    dtime       = params['dt']
    start       = params['start_time']
    end         = params['end_time']
    grid_bnds   = params['grid_extent']
    grid_dx     = params['grid_spacing']
    xover       = params['xover']
    grid_3d     = False
    alt_ranges  = (0,30e3,500) #may have too m any points, change dz if needed
    regrid      = False
    #Start map projection
    #Updated 03/28/22 -- added exception if network not in network center file (mostly for mobile deployments as they change)
    #in the case of a mobile deployment (MLMA), new projection instances will be created based off the network center in the param file
    print(xover)
    try:
        print('Using archived network center location.')
        geo,proj,(nlon,nlat) = bk.LMA_projection(network)
    except:
        print('No archive network center location found--using mobile network location for selected case.')
        from lmatools import coordinateSystems as cs
        lat = network_lat
        lon = network_lon
        geo = cs.GeographicSystem()
        proj= cs.MapProjection(projection='eqc',ctrLat=lat,ctrLon=lon,ellipse='WGS84',datum='WGS84')

    print('Setting up gridding for Case {0}'.format(case))
    #For reference
    year = case[:4]
    month= case[4:6]
    day  = case[6:]
    print('LMA case: {0}-{1}-{2}'.format(year,month,day))

    #Here, we assume flash sorting has already been done,
    #and so the flash sorted files are opened.
    print('Opening flash sorted file')
    try:
        lma_file = f'sorted/{year}-{month}-{day}-{network}_QC.nc'
        lma_data = xr.open_dataset(lma_file)
    except:
        print('Switching to local file path for data in separate location.')
        '''CHANGE THE PATH!!!'''
        lma_file = f'/Users/admin/Desktop/PERiLS-v1/LightningObs/LMA/sorted/Filtered/{year}-{month}-{day}-{network}_QC.nc'
        lma_data = xr.open_dataset(lma_file)
    ds0      = lma_data.copy() #make a copy to use for populating grids

    #Start Gridding initialization:
    print('Setting up grid')
    if grid_3d == True:
        grid_edge_ranges, grid_center_names, event_coord_names = grid_setup(lon_ranges,lat_ranges,alt_ranges,spacing,case,start,end,dtime,xover,grid_3d)
    else:
        grid_edge_ranges, grid_center_names, event_coord_names = grid_setup(lon_ranges,lat_ranges,None,spacing,case,start,end,dtime,xover,grid_3d)
    #Create grid
    grid_ds = create_regular_grid(grid_edge_ranges, grid_center_names)
    #Generate event grids
    ds      = assign_regular_bins(grid_ds, ds0, event_coord_names, pixel_id_var='event_pixel_id', append_indices=True)
    #If file already exists check for it, else run gridding to get gridded data--We will NOT save the data yet!
    try:
        if grid_3d == True:
            grid_ds = xr.open_dataset(f'gridded/{desired_dt}/OKLMA_GRIDS_{case[2:]}_{desired_dt}_3d.nc')
        else:
            grid_ds = xr.open_dataset(f'gridded/{desired_dt}/OKLMA_GRIDS_{case[2:]}_{desired_dt}.nc')

    except:
        print('No file exists...running gridding.')
        start   = dt.datetime.now()
        if grid_3d == True:
            grid_ds = events_to_grid(ds, grid_ds, min_points_per_flash=10,
                                    grid_spatial_coords=['grid_time', 'grid_altitude', 'grid_latitude', 'grid_longitude'],
                                    )
        else:
            grid_ds = events_to_grid(ds, grid_ds, min_points_per_flash=10,
                                    grid_spatial_coords=['grid_time', None, 'grid_latitude', 'grid_longitude'],
                                    )
        end     = dt.datetime.now()
        print('Total Time:',(end-start).total_seconds())

    #Set up domain for regridding--want everything on a uniform grid:
    dx      = float(grid_dx)#1e3
    dom_max = float(grid_bnds)
    dom_min = -float(grid_bnds)
    xbins   = np.arange(dom_min,dom_max,dx)
    ybins   = np.arange(dom_min,dom_max,dx)
    xm,ym   = np.meshgrid(xbins,ybins)

    #Transform and project LMA grid coordinates in grid_ds to cartesian coordinates centered at LMA network:
    grid_lon = grid_ds.grid_longitude.values
    grid_lat = grid_ds.grid_latitude.values
    if grid_3d == True:
        grid_alt = np.arange(alt_ranges[0],alt_ranges[1],alt_ranges[2])#grid_ds.grid_altitude.values
        grid_alt = np.atleast_1d(grid_alt)
    else:
        grid_alt = np.repeat(0,grid_lon.shape[0])
    lonm,latm = np.meshgrid(grid_lon,grid_lat)
    #Staggered method below is because grid_lon and grid_lat may not be the same size
    grid_x,_,_ = proj.fromECEF(*geo.toECEF(grid_lon,grid_lon*0,grid_lon*0))
    _,grid_y,_ = proj.fromECEF(*geo.toECEF(grid_lat*0,grid_lat,grid_lat*0))
    gx,gy = np.meshgrid(grid_x,grid_y)
    print('Check original grid shape: lon={0}, lat={1}'.format(grid_x.shape,grid_y.shape))

    grid_ds0 = grid_ds.copy()
    if regrid == True:
        #Set up new grid dataset, with original gridded data, for writing the regridded data and new
        #spatial coordinates
        grid_ds0 = grid_ds0.assign_coords({"grid_x":xbins,"grid_y":ybins}) #assigns new projection coordinates

        #Prepare for regridding:
        print('Setting up gridded data for regridding onto analysis domain.')
        iters = grid_ds.grid_time.shape[0] #number of time frames
        xform_type = 'nearest' #method of interpolation onto new grid
        print('Starting gridded variable re-gridding/interpolation onto analys domain.')
        #Regrid using griddata
        fed_grids=np.array([gd(np.vstack((gx.flatten(),gy.flatten())).T,
                    grid_ds.flash_extent_density.values[i].flatten(),
                    np.vstack((xm.flatten(),ym.flatten())).T,method=xform_type).reshape(xm.shape) for i in range(iters)]
                  )

        mfa_grids=np.array([gd(np.vstack((gx.flatten(),gy.flatten())).T,
                        grid_ds.average_flash_area.values[i].flatten(),
                        np.vstack((xm.flatten(),ym.flatten())).T,method=xform_type).reshape(xm.shape) for i in range(iters)]
                      )

        std_grids=np.array([gd(np.vstack((gx.flatten(),gy.flatten())).T,
                        grid_ds.stdev_flash_area.values[i].flatten(),
                        np.vstack((xm.flatten(),ym.flatten())).T,method=xform_type).reshape(xm.shape) for i in range(iters)]
                      )

        min_grids=np.array([gd(np.vstack((gx.flatten(),gy.flatten())).T,
                        grid_ds.minimum_flash_area.values[i].flatten(),
                        np.vstack((xm.flatten(),ym.flatten())).T,method=xform_type).reshape(xm.shape) for i in range(iters)]
                      )

        mfe_grids=np.array([gd(np.vstack((gx.flatten(),gy.flatten())).T,
                        grid_ds.average_flash_energy.values[i].flatten(),
                        np.vstack((xm.flatten(),ym.flatten())).T,method=xform_type).reshape(xm.shape) for i in range(iters)]
                      )

        ec_grids=np.array([gd(np.vstack((gx.flatten(),gy.flatten())).T,
                        grid_ds.event_count.values[i].flatten(),
                        np.vstack((xm.flatten(),ym.flatten())).T,method=xform_type).reshape(xm.shape) for i in range(iters)]
                      )

        ep_grids=np.array([gd(np.vstack((gx.flatten(),gy.flatten())).T,
                        grid_ds.event_total_power.values[i].flatten(),
                        np.vstack((xm.flatten(),ym.flatten())).T,method=xform_type).reshape(xm.shape) for i in range(iters)]
                      )

        #Assign new grid variables to add to orignal variables:
        print('Writing re-gridded data into dataset')
        grid_ds0 = grid_ds0.assign({'flash_extent_density_regrid':(('grid_time','grid_x','grid_y'),fed_grids),
                                'average_flash_area_regrid'  :(('grid_time','grid_x','grid_y'),mfa_grids),
                                'stdev_flash_area_regrid'    :(('grid_time','grid_x','grid_y'),std_grids),
                                'minimum_flash_area_regrid'  :(('grid_time','grid_x','grid_y'),min_grids),
                                'average_flash_energy_regrid':(('grid_time','grid_x','grid_y'),mfe_grids),
                                'event_count_regrid'         :(('grid_time','grid_x','grid_y'),ec_grids ),
                                'event_total_power_regrid'   :(('grid_time','grid_x','grid_y'),ep_grids )})

    #Save grids:
    print('Saving gridded data.')
    if grid_3d == True:
        grid_ds0.to_netcdf(f'gridded/{desired_dt}/OKLMA_GRIDS_{case[2:]}_{desired_dt}_3d.nc')

    else:
        grid_ds0.to_netcdf(f'gridded/{desired_dt}/OKLMA_GRIDS_{case[2:]}_{desired_dt}.nc')
