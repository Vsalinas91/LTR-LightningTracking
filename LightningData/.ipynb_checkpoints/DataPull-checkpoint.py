from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets
import matplotlib.pyplot as plt

import urllib.request
from datetime import datetime, timedelta
import glob
import numpy as np
import pandas as pd
import shapely.geometry as sgeom


from pyxlma.lmalib.io import read as lma_read
from pyxlma.lmalib.flash.cluster import cluster_flashes
from pyxlma.lmalib.flash.properties import flash_stats, filter_flashes
from pyxlma.lmalib.grid import  create_regular_grid, assign_regular_bins, events_to_grid
from pyxlma.plot.xlma_plot_feature import color_by_time, plot_points, setup_hist, plot_3d_grid, subset
from pyxlma.plot.xlma_base_plot import subplot_labels, inset_view, BlankPlot
from lmatools import coordinateSystems as cs
from lmatools.density_tools import extent_density, unique_vectors




class data_selector(object):
    def __init__(self,pull):
        self.case          = '19990101'
        self.start_hour    = '0'
        self.end_hour      = '23'
        self.analysis_hour = '0'
        self.pull          = pull
    
        
        self.chi2max    = 0
        self.minstations= 0
        self.min_events = 0
        
        self.center_lon = 0
        self.center_lat = 0
        self.point_buffer = 0
        self.line_buffer = 0
        
        
        #Widgets for case selection
        #------------------------
        self.start_date = widgets.DatePicker(
                            description='Start Date',
                            disabled=False
                            )
    
        self.start_hour_select = widgets.Dropdown(value=int(self.start_hour),options=np.arange(0,24).tolist(),description='Starting Hour (UTC)')
        self.end_hour_select   = widgets.Dropdown(value=int(self.end_hour),options=np.arange(0,24).tolist(),description='Ending Hour (UTC)')
        self.selected_chunk    = widgets.Dropdown(value=int(self.analysis_hour),options=np.arange(0,23).tolist(),description='Analysis Hour (UTC)')

    
        self.date_widgets = widgets.VBox([widgets.HBox([self.start_date]),
                                          widgets.HBox([self.start_hour_select]),
                                          widgets.HBox([self.end_hour_select]),
                                          widgets.HBox([self.selected_chunk])]
                                        )
        
        #Widgets for lma filtering:
        #----------------------------
        self.chi2         = widgets.Dropdown(value=1,options=np.arange(0,5,1),description='Reduced Chi2Max')
        self.min_stations = widgets.Dropdown(value=5,options=np.arange(0,6,1),description='Minimum Number of Stations')
        self.min_events_flash= widgets.Dropdown(value=10,options=np.arange(0,11),description='Minimum Number of Events')
        
        self.filter_widgets = widgets.VBox([self.chi2,
                                            self.min_stations,
                                            self.min_events_flash])
        
        #Eye center location:
        self.eye_lon = widgets.FloatText(value=-80.0,description='Approximate Eye Logitude')
        self.eye_lat = widgets.FloatText(value=27.0,description='Approximate Eye Latitude')
        self.eye_width= widgets.FloatText(value=46,description='Buffer from Center (km)')
        self.wall_width= widgets.FloatText(value=50,description='Eyewall Buffer (km)')
        self.zres_set = widgets.FloatText(value=.25,description='Vertical Grid dz (km)')
        self.dx_set   = widgets.FloatText(value=1.0,description='Horizontal Grid dx (km)')

        self.fed_widgets = widgets.VBox([self.eye_lon,
                                         self.eye_lat,
                                         self.eye_width,
                                         self.wall_width,
                                         self.zres_set,
                                         self.dx_set])
        
    def update_selections(self):
        self.case         = self.start_date.value.strftime('%Y%m%d')
        self.start_hour   = self.start_hour_select.value
        self.end_hour     = self.end_hour_select.value
        self.analysis_hour= self.selected_chunk.value
        
    def update_filters(self):
        self.chi2max = self.chi2.value
        self.minstations= self.min_stations.value
        self.min_events = self.min_events_flash.value
        
    def update_center(self):
        self.center_lon = self.eye_lon.value
        self.center_lat = self.eye_lat.value
        self.point_buffer= self.eye_width.value
        self.line_buffer= self.wall_width.value
        self.zres       = self.zres_set.value
        self.dxres      = self.dx_set.value

    def lma_thredds(self):
        '''
        Short script to pull LMA data for any time period from the NSSL thredds server.
        Args:
            case = string of the desired case date (e.g., 20220928). Must be in YYYYMMDD format
            start_hour = first file requested to download
            end_hour   = final file data requested to download.
            pull       = If True, data are downloaded, else assume data have already been collected.

        All data downloaded into the directory in which the function is executed.
        '''
        #Download data if you don't have it.
        #----------
        self.update_selections()
        if self.pull == True:
            try:
                base_url = f'https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/{case}/LYLOUT_{case}_%H%M%S_0600.dat.gz'

                first_time = datetime(int(self.case[:4]),int(self.case[4:6]),int(self.case[6:]),self.start_hour,0,0)
                last_time  = datetime(int(self.case[:4]),int(self.case[4:6]),int(self.case[6:]),self.end_hour  ,0,0)
                file_time_step = timedelta(0, 600)
                n_files = (last_time-first_time)/file_time_step

                all_times = (first_time + file_time_step*i for i in range(int(n_files)))
                filenames = [t.strftime(base_url) for t in all_times]
                for fn in filenames[:]:
                    base_fn = fn.split('/')[-1]
                    print("Downloading", base_fn)
                    urllib.request.urlretrieve(fn, filename=base_fn)
            except:
                print('Base url is incorrect for selected date...Verify the correct path to Thredds data directory via a browser and edit path here.')
        else:
            print('No data requested to download.')
            
            
    def lma_load(self):
        self.update_selections()
        #Read files:
        #-----------
        print("Reading files")
        files         = sorted(glob.glob(f'processed/LYLOUT_{self.case[2:]}_{str(self.analysis_hour)}*_0600.dat.gz'))
        self.lma_data, self.starttime = lma_read.dataset(files)
        
        
    def lma_filter(self):
        print('Filtering L2 data.')
        self.update_filters()
        #Filter L2 data and set up time range for analysis
        #-----------
        good_events = (self.lma_data.event_stations >= self.minstations) & (self.lma_data.event_chi2 <= self.chi2max)
        self.lma_data = self.lma_data[{'number_of_events':good_events}]

        self.dttuple = [self.starttime, self.starttime+timedelta(minutes=60)]#starttime+timedelta(minutes=duration_min)]
        self.tstring = 'LMA {}-{}'.format(self.dttuple[0].strftime('%H%M'),
                                          self.dttuple[1].strftime('%H%M UTC %d %B %Y '))
        
        #Set analysis time limits and duration (60*1) = 10 minute chunk
        self.tlim_sub = [pd.to_datetime(self.starttime), pd.to_datetime(pd.to_datetime(self.starttime) + np.asarray(60*1, 'timedelta64[m]'))]
        print(self.tstring)
        print(self.dttuple)
        
    def flash_sort(self):
        #Flash Sorting:
        #------------
        print("Clustering flashes")
        self.ds = cluster_flashes(self.lma_data)
        print("Calculating flash stats")
        self.ds = filter_flashes(self.ds, flash_event_count=(self.min_events, None))
        self.ds = flash_stats(self.ds)
        self.ds0 = self.ds.copy() #make a copy for using flash sorted data later
        
        
    def plot_overview(self):
        #Set some default variables
        #--------------
        alt_data     = self.ds.event_altitude.values/1000.0
        lon_data     = self.ds.event_longitude.values
        lat_data     = self.ds.event_latitude.values
        time_data    = pd.Series(self.ds.event_time) # because time comparisons
        chi_data     = self.ds.event_chi2.values
        station_data = self.ds.event_stations.values


        # Plot color map and marker size
        plot_cmap = 'plasma'
        plot_s = 5
        
        
        tstring = 'LMA {}-{}'.format(self.tlim_sub[0].strftime('%H%M'),
                                     self.tlim_sub[1].strftime('%H%M UTC %d %B %Y '))

        clat, clon = float(self.lma_data.network_center_latitude), float(self.lma_data.network_center_longitude)
        shift = 1 #degree
        xlim = [clon-(1.5+shift), clon+1.5]
        ylim = [clat-(1.5+shift), clat+1.5]
        zlim = [0, 20]
        xchi = 1.0
        stationmin = 5.0

        # END OF CONFIG

        lon_set, lat_set, alt_set, time_set, selection = subset(
                   lon_data, lat_data, alt_data, time_data, chi_data, station_data,
                   xlim, ylim, zlim, self.tlim_sub, xchi, stationmin)

        bk_plot = BlankPlot(pd.to_datetime(self.tlim_sub[0]), bkgmap=True, 
                      xlim=xlim, ylim=ylim, zlim=zlim, tlim=self.tlim_sub, title=tstring)

        # Add a view of where the subset is
        xdiv = ydiv = 0.1
        inset_view(bk_plot, lon_data, lat_data, xlim, ylim, xdiv, ydiv,
                  buffer=0.5, inset_size=0.15, plot_cmap = 'plasma', bkgmap = True)
        # Add some subplot labels
        subplot_labels(bk_plot)
        # Add a range ring
        #bk_plot.ax_plan.tissot(rad_km=40.0, lons=clon, lats=clat, n_samples=80,
        #                  facecolor='none',edgecolor='k')
        # Add the station locations
        stn_art = bk_plot.ax_plan.plot(self.lma_data['station_longitude'], 
                                       self.lma_data['station_latitude'], 'wD', mec='k', ms=5)

        if len(lon_set)==0:
            bk_plot.ax_hist.text(0.02,1,'No Sources',fontsize=12)
        else:
            plot_vmin, plot_vmax, plot_c = color_by_time(time_set, self.tlim_sub)
            plot_points(bk_plot, lon_set, lat_set, alt_set, time_set,
                              plot_cmap, plot_s, plot_vmin, plot_vmax, plot_c)

        plt.show()
        
    def source_selection(self):
        #get data for lma event and initiation source locations and convert them to cartesian map projected coordinates:
        self.geo = cs.GeographicSystem()
        filter_range_ctr_lon, filter_range_ctr_lat = self.ds.network_center_longitude.data, self.ds.network_center_latitude.data
        self.proj= cs.MapProjection(projection='aeqd',ctrLat=filter_range_ctr_lat,ctrLon=filter_range_ctr_lon)

        self.lmax,self.lmay,self.lmaz                = self.proj.fromECEF(*self.geo.toECEF(self.ds0.event_longitude.data       ,self.ds0.event_latitude.data       ,
                                                                                           self.ds0.event_altitude.data       ))
        self.lma_initx,self.lma_inity,self.lma_initz = self.proj.fromECEF(*self.geo.toECEF(self.ds0.flash_init_longitude.data  ,self.ds0.flash_init_latitude.data  ,
                                                                                           self.ds0.flash_init_altitude.data  ))
        self.lma_ctrx,self.lma_ctry,self.lma_ctrz    = self.proj.fromECEF(*self.geo.toECEF(self.ds0.flash_center_longitude.data,self.ds0.flash_center_latitude.data,
                                                                                           self.ds0.flash_center_altitude.data))
        
        
        
        
    def eyewall_sample(self,latlon):
        self.update_center()
        if latlon==False:
            approx_x,approx_y = -140,-140 #17000 
        else:
            #using lon lats
            approx_lon,approx_lat = self.center_lon,self.center_lat #-82.6,26.3
            approx_x,approx_y,_ = self.proj.fromECEF(*self.geo.toECEF(approx_lon,approx_lat,0))
            approx_x,approx_y = approx_x*1e-3,approx_y*1e-3

        pt = sgeom.asPoint((approx_x,approx_y))
        buffer = pt.buffer(self.point_buffer,resolution=60)
        boundary_coords = buffer.boundary.xy
        circ = np.vstack((np.hstack(boundary_coords[0]),np.hstack(boundary_coords[1]))).T
        self.center_x = approx_x
        self.center_y = approx_y
        
        #Get good points around line buffer
        buff = self.line_buffer #50#10
        self.good_pts = [buffer.boundary.buffer(buff).contains(sgeom.Point(x,y)) for x,y in zip(self.lma_initx*1e-3,self.lma_inity*1e-3)]
        self.valid_x = self.lma_initx[self.good_pts]
        self.valid_y = self.lma_inity[self.good_pts]
        self.valid_z = self.lma_initz[self.good_pts]
        self.valid_id= self.ds.flash_id.values[self.good_pts]
        
        
        self.src_ids = self.ds0.event_parent_flash_id.values
        
        
        #now get good sources:
        self.valid_src_ids = []
        self.valid_src_alt = []
        self.valid_src_x   = []
        self.valid_src_y   = []
        self.valid_src_time = []
        for f in self.valid_id:
            fmask = (f==self.src_ids)
            self.valid_src_alt.append( self.lmaz [fmask])
            self.valid_src_x.append(   self.lmax[fmask])
            self.valid_src_y.append(   self.lmay[fmask])
            self.valid_src_ids.append( self.src_ids[fmask])

        self.valid_src_x   = np.hstack(self.valid_src_x)
        self.valid_src_alt = np.hstack(self.valid_src_alt)
        self.valid_src_y   = np.hstack(self.valid_src_y)
        self.valid_src_ids = np.hstack(self.valid_src_ids)
        self.valid_src_alt = self.valid_src_alt * 1e-3
        
        
        #Chunk over altitude to build 3D FED
        zres = self.zres #km
        self.zleft = np.arange(0,20,zres)
        self.zright= np.arange(0,20,zres) + zres
        self.xchunk = [self.valid_src_x  [(self.valid_src_alt>=zl) & (self.valid_src_alt<zr)] for zl,zr in zip(self.zleft,self.zright)]
        self.ychunk = [self.valid_src_y  [(self.valid_src_alt>=zl) & (self.valid_src_alt<zr)] for zl,zr in zip(self.zleft,self.zright)]
        self.zchunk = [self.valid_src_alt[(self.valid_src_alt>=zl) & (self.valid_src_alt<zr)] for zl,zr in zip(self.zleft,self.zright)]
        self.idchunk =[self.valid_src_ids[(self.valid_src_alt>=zl) & (self.valid_src_alt<zr)] for zl,zr in zip(self.zleft,self.zright)]


        self.xichunk = [self.valid_x [(self.valid_z*1e-3 >=zl) & (self.valid_z*1e-3 <zr)] for zl,zr in zip(self.zleft,self.zright)]
        self.yichunk = [self.valid_y [(self.valid_z*1e-3 >=zl) & (self.valid_z*1e-3 <zr)] for zl,zr in zip(self.zleft,self.zright)]
        self.iidchunk= [self.valid_id[(self.valid_z*1e-3 >=zl) & (self.valid_z*1e-3 <zr)] for zl,zr in zip(self.zleft,self.zright)]
        
        #Future feature is to also control bounds
        dx = self.dxres
        self.xrange = np.arange(-290,-10,dx)
        self.yrange = np.arange(-290,-10,dx)
        
        #Produce cylindrical coordinates for unwrapping: #ADD CONFIGURATION WIDGETS HERE!!
        self.rrange = np.arange(0,300,5)
        self.thrange= np.arange(-180,180,1)# -149,149. np.arange(0,360,5)

        self.rchunk  = [np.sqrt((np.array(self.xchunk)[i]-approx_x*1e3)**2. + (np.array(self.ychunk)[i]-approx_y*1e3)**2) for i in range(len(self.xchunk))]
        self.thchunk = [np.arctan2((np.array(self.ychunk)[i]-approx_y*1e3),(np.array(self.xchunk)[i]-approx_x*1e3)) for i in range(len(self.xchunk))]


        self.richunk  = [np.sqrt(   (np.array(self.xichunk)[i]-approx_x*1e3)**2. + (np.array(self.yichunk)[i]-approx_y*1e3**2)) for i in range(len(self.xichunk))]
        self.thichunk = [np.arctan2((np.array(self.yichunk)[i]-approx_y*1e3),      (np.array(self.xichunk)[i]-approx_x*1e3)) for i in range(len(self.xichunk))]
        
    def eye_wall_fed(self,view):
        self.dens_polar = []
        for (xi,yi,xc,yc,vid_s,vid_i) in zip(self.richunk,self.thichunk,self.rchunk,self.thchunk,self.idchunk,self.iidchunk):
            angs  = np.rad2deg(yi)
            angsc = np.rad2deg(yc)

            view=view
            fdens_p = self.extend_density_2d(xi*1e-3,angs-(view),
                                          self.rrange,self.thrange,
                                          5,1,
                                          xc*1e-3,angsc-(view),
                                          vid_s,vid_i)
            self.dens_polar.append(fdens_p)
            
        self.redge,self.thedge= np.array(self.dens_polar)[:,0][0][0],np.array(self.dens_polar)[:,0][0][1]
        self.densitiesP  = np.array(self.dens_polar)[:,1]
        self.stacked_densP = np.vstack(self.densitiesP).reshape(self.densitiesP.shape[0],self.rrange.shape[0]-1,self.thrange.shape[0]-1)
        
        
    def flash_prop_dir(self,adjust):
        '''
        Calculate the direcation of propagation for individual flashes. 
        Assumes flashes propagation is euclidean--ie., tortuosity of branched development ignored.
        '''
        self.ptheta = []
        for f in self.valid_id:
            fmask = np.where(f==self.src_ids)
            x_v,y_v   = (self.lmax[fmask]-self.center_x*1e3), (self.lmay[fmask]-self.center_y*1e3)
            t = self.ds0.event_time.values[fmask]
            try:
                prop_ang  = np.arctan2(y_v[t==t.max()]-y_v[t==t.min()],x_v[t==t.max()]-x_v[t==t.min()])
            except:
                prop_ang  = np.repeat(np.nan,len(fmask))
            self.ptheta.append(np.rad2deg(prop_ang))

        self.ptheta = np.hstack(self.ptheta)
        if adjust == True:
            print('Adjusting angle to range from 0 to 2pi.')
            self.ptheta[(self.ptheta<=0)] = 360+self.ptheta[(self.ptheta<=0)]
        else:
            print('Retaining angles in range of -pi to pi for unwrapped cross-section.')
            
    def mean_prop_dir(self,adjust,dom_min,dom_max,dx):
        self.flash_prop_dir(adjust)
        self.angle_lines()
        #eyewall dir
        xranges = np.arange(dom_min,dom_max,dx)#-300,-10,2)
        yranges = np.arange(dom_min,dom_max,dx)#-300,-10,2)

        self.Hdir,self.xdir,self.ydir = np.histogram2d( self.valid_x*1e-3,self.valid_y*1e-3,bins=(xranges,yranges))
        self.Hdirw,self.xdir,self.ydir = np.histogram2d(self.valid_x*1e-3,self.valid_y*1e-3,bins=(xranges,yranges),weights=(self.ptheta))
        
    def angle_lines(self):
        '''
        Generate lines from 0-360 degress at 45 degree increments.
        '''
        self.newx,self.newy = [],[]
        inter = 45 #increments in degrees
        for ang in np.arange(0,360,inter):
            linex,liney = np.arange(0,100,5),np.arange(0,100,5)
            for (lx,ly) in zip(linex,liney):
                rotx,roty = self.rotate((0,0),(lx-self.center_x*1e-3,ly-self.center_y*1e-3),np.deg2rad(ang-45))
                self.newx.append(rotx),self.newy.append(roty)
        self.newx = np.array(self.newx).reshape(np.arange(0,360,inter).shape[0],np.arange(0,100,5).shape[0])
        self.newy = np.array(self.newy).reshape(np.arange(0,360,inter).shape[0],np.arange(0,100,5).shape[0])
        
    def rotate(self,origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians. Used to rotate tracked ellipses
        to draw line from focal-point to focal-point, then rotates back to
        original orientation.

        Args:
            origin = centroid of sample line
            point  = (x,y) point(s) to be rotated
            angle  = orientation angle to rotate ellipse back onto x-axis

        Returns:
            qx,qy = rotated points
        """
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return(qx,qy)
    
    
    def extend_density_2d(self,x,y,xbins,ybins,dx,dy,x_doubled,y_doubled,ids_doubled,ids):
        x0, x1 = xbins[0],xbins[-1]
        y0, y1 = ybins[0],ybins[-1]
        x_cover, y_cover = np.meshgrid(x,y)
        xedge = np.arange(x0, x1+dx, dx)
        yedge = np.arange(y0, y1+dy, dy)
        density, edges = extent_density(x_doubled, y_doubled, ids_doubled, 
                                        x0, y0, dx, dy, xedge, yedge)
        return(edges,density)
