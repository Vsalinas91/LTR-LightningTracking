################################################
#
# Serves as the basis for most tracking routines
# used to Identify regions of lightning activity to
# track the path and orientation of linear storms.
#
# 12/29/22:
# At this time, could improve how smoothing is done
# by minimzing the value that give one single large object.
# Implement in future?
################################################

#Basics
import pandas as pd
import datetime as dt
import numpy as np
from scipy import ndimage
from lmatools import coordinateSystems as cs

#Tracking Imports
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from scipy import ndimage
from scipy.ndimage import gaussian_filter as gf

#Tracking Ellipse and plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
sns.reset_orig()

#Geometric bounds and vertex recovery
import shapely as sh
from shapely.geometry import Polygon,Point

#--------------------------
#Coordinate transformations
#--------------------------
def LMA_projection(network,ctrlat,ctrlon):
    '''Get map projection for LMA domain to center all data onto a
       newtork's domain. Makes gridding, or analysis, of multiple
       data assets easier to compare when they are aligned on a common
       map projection.

       04/21/22:
       ---------
       To add: if network is mobile, pass through mobile center lat lons to get projection, else use station_centers.csv file
    '''
    try: #For non-mobile cases
        #Set network center:
        #-------------------
        center    = pd.read_csv('/Users/admin/Desktop/VSE-TestCases/station_centers.csv')
        networks  = center.stid.values
        valid_idx = [i for i,n in enumerate(networks) if network in str(n)]
        if len(valid_idx) > 1:
            sub_nets = (networks[valid_idx])
            valid_net= sub_nets[0]
        else:
            valid_net= networks[valid_idx]

        cent_df = center.iloc[center.stid.values==valid_net]
        lon,lat = cent_df['ctrlon'].values,cent_df['ctrlat'].values
    except:#for mobile cases
        lon,lat = ctrlon,ctrlat
    #Set up coordinate transformation for distance from center calculations:
    geo = cs.GeographicSystem()
    proj= cs.MapProjection(projection='eqc',ctrLat=lat,ctrLon=lon,ellipse='WGS84',datum='WGS84')
    return(geo,proj,(lon,lat))

#-----------------------------------
#Operations for tracking geometries
#-----------------------------------
def img_var(img,pix):
    '''Generate 2D Variance grids for any field if desired, simply
       import this method and provide the data field "img" and the number of
       adjacent grid points "pix" to compute the variance from.
    '''
    rows, cols = pix,pix
    win_rows, win_cols = 2,2

    win_mean = ndimage.uniform_filter(img, (win_rows, win_cols))
    win_sqr_mean = ndimage.uniform_filter(img**2, (win_rows, win_cols))
    win_var = win_sqr_mean - win_mean**2
    return(win_var)

def shift_ellipse(angle,x,y,distance):
    '''For tracking lightning data, ellipses are drawn around
       each tracked object. This routine alows one to shift the centroid of the
       ellipses so as to sample different regions of the storm.

       Note: It is recommended that this shift be done simply by
       shifting the points instead unless it is desired to shift the
       sampling region early on.

       Args:
           angle = orientation angle
           x,y = centroid coordinates of ellipse
           distance = distance to offset the centroids (can be +/-)
    '''
    angle = np.deg2rad(angle + 90)
    nx = x + distance*np.cos(angle)
    ny = y + distance*np.sin(angle)
    return nx,ny

def smooth_angle(i,onow,opast,n,counter,avg,t_frames):
    '''solution based on: https://stackoverflow.com/questions/68496749/smoothing-2d-points-scattered-in-real-time-online-smoothing
       The tracking of lightning objects can "wobble" about the tracked object's centroid making smoothed
       sampling difficult. This routine smooths the orientation of the identified lightning object by
       taking the current time step "i" and orientation "onow" and averaging to a new smoothed orientation for the next time
       step to obtain a smoothed "next" object orientation.

       Args:
           i = current iteration
           onow = current orientation of object
           opast = previous orientation of object (skipped for 0th iteration)
           n = Window size to smooth n previous orinetations for current iteration.
           counter = number of iterations accumulated over time
           avg = first pass at the average is set to 0.
    '''
    #Get difference in angle from previous time step:
    if (i+1) < t_frames.shape[0]:
        dthetas = np.abs(onow-opast)
    elif (i+1) == t_frames.shape[0]:
        dthetas = np.abs(onow - opast)
    val = onow
    #if angle is > 170 degress it's probably in a new quadrant and we dont want to refine it
    if dthetas >= 90:
        avg = val
    elif dthetas < 90:
#         counter += 1
        coeff = 1. / min(counter, n)
        # update using moving average
        avg = coeff * val + (1. - coeff) * avg
    return(avg)

def rotate(origin, point, angle):
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


#------------------------------------------
#Storm Tracking with Lighting Gridded Data
#------------------------------------------
def tracking(case,base_date,lx,ly,led,grid_bnds,dx,constant_feature,set_width,set_height,gauss_smooth,offset,t_frames,frame_dt,skip_times=0,save_fig=True):
    '''Routine to track storm motion and orientation using either GLM (current) or LMA data--specifically group extent (GLM) or
       event extent (LMA) densities. Tracking parameters are specified before calling this method.

       Args:
           base_date  = case base date
           lx,ly      = glm/lma x and y coordinates (after projection and parallax correction)
           led        = lightning extent density (GLM GED [group event] or LMA EED [event extent])
           grind_bnds = bounds by which to constrain analysis within, also used to set domain extent
                        of lma/glm grid products.
           dx         = grid domain spacing for converting ellipse semi-major/minor lengths back into cartesian coordinates
                        (only used if constant feature == False)
           t_frames   = the time frames used to make analysis (parameter is set in the param file and constructed in
                        whatever is being used to call this method.)
           skip_times = number of initial frames that are skipped prior to tracking, default is 0.
           save_fig   = boolean to decided wether to save tracked frames to provide example of how the feature tracking is working.

       Tracking params:
           constant_feature = set if constant feature (TRUE), or adaptive tracking (FALSE), is desired.
                              adaptive tracking allows ellipse to expand contract with storm size.
           set_width        = set ellipse width (semi-minor axis) if constant feature is set to TRUE, 0 if not
           set_height       = set ellipse height (semi-major axis) if constant feature is set to True, 0 if not
           gauss_smooth     = smoothing factor by which to identify solid objects of lightning activity, factor
                              is number of pixels (grid points) away from any one grid point that is averaged to smooth
                              lightning data. (e.g. 10 = 10 pixels surrounding any one point to smooth GED or EED)
           offset           = number in km to offset the ellipse from original tracked location (forwards or backwards in direction
                              of storm motion)

        Returns:
            get_eccentricity = identified/tracked feature eccentricity (0,1, with 0 meaning ellipsoidal and 1 circular)
            get_centroidx    = tracked feature centroid x coordinate -- these may be shifted with the offset parameter
            get_centroidy    = tracked feature centroid y coordinate -- these may be shifted with the offset parameter
            get_ocentroidx   = tracked feature centroid x coordinate -- original, no offset regardless of parameter
            get_ocentroidy   = tracked feature centroid y coordinate -- original, no offset regardless of parameter
            patches          = Ellipse geometry patch data (vertices, centroid, other matplotlib properties)
            get_ftime        = current frame time
            get_path_coords  = Ellipse vertices converted into path coordinates for plotting over other spatial figures
            get_patch_geom   = Get ellipse path as a ring geometry from shapely, replaces use of get_path_coords
            get_semi_major_len = object semi-major axis (constant if constant_feature==True)
            get_semi_minor_len = object semi-minor axis (constant if constant_feature==True)
            get_orientation  = object orientation prior to smoothing
            get_norientation = smoothed object orientation
            get_valid_idx    = if skip_times !=0, this gets the adjusted indices to index the proper GLM/LMA time frames

           '''
    get_eccentricity    = []
    get_centroidx       = []
    get_centroidy       = []
    get_ocentroidx      = []
    get_ocentroidy      = []
    patches             = []
    get_ftime           = []
    get_path_coords     = []
    get_patch_geom      = []
    get_semi_major_len  = []
    get_semi_minor_len  = []
    get_orientation     = []
    get_norientation    = []
    #Get the indices to index the correct time frames later
    get_valid_tidx      = []

    #For smoothing the object orientation:
    n       = 2  # the average "window size"
    counter = 0  # count how many steps so far
    avg     = 0.  # the average
    opast   = 0

    print('Starting Tracking Routine')
    print('Tracking Field: GLM GED')
    print('Smoothing Field by sigma={0}'.format(gauss_smooth))
    print('Total Time Frames: {0}'.format(len(t_frames)))
    for m,(nt,i) in enumerate(zip(t_frames[:],np.arange(t_frames.shape[0])[:])):
        #NEW ADDED
        #Set new index for glm data to get correct index if time frames are skipped
        # j is the correct index relative to the entire glm data array, if skipping is desired then
        # j != i and will be at a value of the skipped frame onwards, else j==1.
        j = i+skip_times
        #Get frame title:
        frame_title = (base_date + dt.timedelta(seconds=t_frames[i])).strftime('%Y-%m-%d %H:%M:%S')
        #Get image to track linear storm--use GCD Standard Deviation
        image = np.log10(led[j])
        image[~np.isfinite(image)] = 0 #filter out all inf values due to log scaling of 0s
        if len(lx.shape) > 2:
            box_filter = ((lx[j]*1e-3 < grid_bnds*1e-3) & (lx[j]*1e-3 > -grid_bnds*1e-3) &
                          (ly[j]*1e-3 < grid_bnds*1e-3) & (ly[j]*1e-3 > -grid_bnds*1e-3))
        else:
            box_filter = ((lx*1e-3 < grid_bnds*1e-3) & (lx*1e-3 > -grid_bnds*1e-3) &
                          (ly*1e-3 < grid_bnds*1e-3) & (ly*1e-3 > -grid_bnds*1e-3))

        image = gf(image,gauss_smooth) #smooth the image so that one large feature is detected

        #Apply thresholding
        #Otsu's method performs automatic image thresholding by searching
        #for the threshold that minimizes the intra-class variance, defined as a weighted sum of variances of the two classes
        thresh = threshold_otsu(image) #cut-off treshold defined by Otsu's Method and is adapted per iteration
        bw     = closing((image >= thresh) & (box_filter), square(1)) #make image into black and white blobs--essetially boolean array

        #Label image regions based on if blob exists or not--more than one label detected if more than one blob found
        label_image = label(bw)
        labels      = label(bw,return_num=True)[1] #get number of labels in image; lowest label is largest feature
        #To make the background transparent, pass the value of `bg_label`,
        #and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

        #Set axis frames to save and animate later
        fig, ax = plt.subplots(1,2,figsize=(8, 3))

        #Iterate over detected blobs, or features in image but first check for if more than one blob appears
        if len(lx.shape) > 2:
            gmx,gmy = lx[j],ly[j]
        else:
            gmx,gmy = lx,ly
        #Iterate over detected blobs, or features in image
        x0,y0 = 0,0
        for region in regionprops(label_image):
            # take regions with largest areas:
            main_object_size   = np.where(label_image==1)[0].shape[0] #for convective line--this should be largest always
            second_object_size = np.where(label_image==2)[0].shape[0]
            #Get all centroids to ensure tracked object is always within analysis domain area
            all_centroid_x = gmx[int(region.centroid[0]),int(region.centroid[1])]
            all_centroid_y = gmy[int(region.centroid[0]),int(region.centroid[1])]
            
            #########
            #Currently, only handles detecting largest linear object. Future work may want to exapand on this
            #by providing means to track multiple features leading up to linear organization.
            d = np.sqrt((all_centroid_x*1e-3 - x0)**2. + (all_centroid_y*1e-3-y0)) #get distance from current to previous point
            x0 = all_centroid_x*1e-3 #previous x
            y0 = all_centroid_y*1e-3 #previous y
            #########
            
            #Here, we check to make sure that we only ID one single region for tracking
            if main_object_size > second_object_size:
                object_size = main_object_size
            else:
                object_size = second_object_size
                
            if (region.area == object_size):
                get_region = region
                get_xcent = all_centroid_x*1e-3
                get_ycent = all_centroid_y*1e-3
                
                #Get feature orientation, or
                #angle between the 0th axis (rows) and the major axis of the ellipse
                #that has the same second moments as the region, ranging from
                #-pi/2 to pi/2 counter-clockwise.

                #Update 04/21/22: For some reason orientation is reversed for IOP cases? Remove -90 if true:
                #need to account for sign of angle: If adjusted is negative (angle -90) then make a positive angle
                if (np.rad2deg(get_region.orientation)-90) < 0:
                    adjust_theta = -1
                else:
                    adjust_theta = 1

                orientation = -(adjust_theta/adjust_theta)*(np.rad2deg(get_region.orientation)-90) #needs minus out font - 12/29/22
                #orientation = np.rad2deg(get_region.orientation) + 90 #only use this line if tracking geometry is not constant!

                #Angle smoothing---------------
                onow = orientation
                avg = smooth_angle(i,onow,opast,n,i+1,avg,t_frames)
                norientation = avg
                get_norientation.append(norientation)
                opast = onow
                #------------------------------

                #Get feature centroid(s)
                centroid_x = get_xcent
                centroid_y = get_ycent

                #shift ellipse towards front of line:
                fcenter_x,fcenter_y = shift_ellipse(norientation,centroid_x,centroid_y,offset)

                #Rotate patch object
                if constant_feature == True:
                    width = set_width
                    height= set_height
                else:
                    width = get_region.major_axis_length*(dx*1e-3)
                    height= get_region.minor_axis_length*(dx*1e-3)

                ellipse = Ellipse((fcenter_x, #(gmx[int(region.centroid[0]),int(region.centroid[1])]*1e-3,
                                   fcenter_y), #gmy[int(region.centroid[0]),int(region.centroid[1])]*1e-3),
                                   width =width,
                                   height=height,
                                   facecolor='None',edgecolor='red',linewidth=3,
                                   angle = norientation)

                ax[1].add_patch(ellipse)
                patches.append(ellipse)
                #End plot matter for region features----

                #Save region props to list
                get_eccentricity.append((i,region.convex_area,region.eccentricity)) #Save both the frame, area, and eccentricity
                get_centroidx.append(fcenter_x*1e3);get_centroidy.append(fcenter_y*1e3)
                get_semi_major_len.append(region.major_axis_length*(dx*1e-3));get_semi_minor_len.append(region.minor_axis_length*(dx*1e-3))
                get_orientation.append(orientation)
                #this is how you recover the transformed path of the ellipses
                pcoords = ellipse.get_patch_transform().transform(ellipse.get_path().vertices[:-1])
                Ring    = sh.geometry.LinearRing(pcoords)
                get_patch_geom.append(Ring)
                #save frame time
                get_ftime.append(t_frames[i])
                get_valid_tidx.append(i)
                #non offset centroids:
                get_ocentroidx.append(get_xcent*1e3);get_ocentroidy.append(get_ycent*1e3)


        for axs in ax.flatten():
            circle1 = plt.Circle((0, 0), 100, fill=False,edgecolor='k',linewidth=1,linestyle='--')
            circle2 = plt.Circle((0, 0), 200, fill=False,edgecolor='k',linewidth=1,linestyle='--')
            axs.add_patch(circle1)
            axs.add_patch(circle2)

        cba = ax[1].pcolormesh(gmx[:]*1e-3,gmy[:]*1e-3,np.log10(led[j]),vmin=0,vmax=2,alpha=.8) #was lx[j],ly[j]
        cbar = plt.colorbar(cba,ax=ax[1])
        cbar.ax.set_ylabel(r'$log_{10}(GED)$')

        cbas= ax[0].pcolormesh(gmx[:]*1e-3,gmy[:]*1e-3,label_image,alpha=0.8,cmap=plt.cm.Reds)
        cbars=plt.colorbar(cbas,ax=ax[0])
        cbars.ax.set_ylabel(r'Identified Feature(s) (Value$\geq$1)')


        [axs.set_xlim(-300,300) for axs in ax.flatten()]
        [axs.set_ylim(-300,300) for axs in ax.flatten()]

        [axs.set_xlabel('X Distance (km)') for axs in ax.flatten()]
        [axs.set_ylabel('Y Distance (km)') for axs in ax.flatten()]
        [axs.set_title(f'{frame_title}')   for axs in ax.flatten()]

        ax[0].text(-250,-250,r'$\sigma$'+'={0} cells'.format(gauss_smooth),fontsize=13,color='k')

        plt.tight_layout()
        [axs.grid(alpha=0.2) for axs in ax.flatten()]
        if save_fig == True:
            plt.savefig(f'Output/{str(frame_dt)}/TrackedFrames/{case[2:]}/FRAME_ID-{str(i).zfill(3)}_TRACK.png',dpi=180)
        plt.close()

    print('Tracking Complete.')
    return(get_eccentricity,get_centroidx,get_centroidy,get_ocentroidx,get_ocentroidy,patches,
           get_ftime,get_path_coords,get_patch_geom,get_semi_major_len,get_semi_minor_len,
           get_orientation,get_norientation,get_valid_tidx)

#-----------------------------
#Plotting overview of tracking
#-----------------------------
def plot_tracking_overview(case,lx,ly,led,good_centx,good_centy,good_patch,good_ids,
                           good_eccs,good_areas,dx,t_frames,skip_times,sensor,frame_dt):

    '''Plot case overview: Includes all tracking ellipses, tracked centroid, cumulative
       lightning contours for mapping out its progression through time, storm reports for
       selected report type, and time series of object eccentricity, area, and flash density
       totals per time frame (verticle lines for storm report times).

       Args:
           lx,ly     = lightning data x and y coordinates projected on LMA domain center
           led       = lightning data (e.g., FED, GED)
           repx,repy = report x and y coordinates projected onto LMA domain
           reptime   = report time(s) in seconds since analysis start date
           good_centx,good_centy = tracked feature centroids (no offset)
           good_patch= tracked feature geometry object to pull vertices for plotting
           good_ids  = all t_frame indices with tracked data (should always be complete without skipping unless
                       constant_feature is set to False)
           good_eccs = tracked feature eccentricities for time series plotting
           good_areas= tracked feature areas for time series plotting
           dx        = lightning grid spacing
           t_frames  = analysis time_frames
           skip_times= index number of time frames skipped from start of analysis (default is 0)

        Plots figure.
        '''
    print(sensor)
    fig,ax = plt.subplots(1,2,figsize=(12,5))
    #Draw concentric range circles:
    #------------------------------
    circle1 = plt.Circle((0, 0), 100, fill=False,edgecolor='k',linewidth=0.5,linestyle='-')
    circle2 = plt.Circle((0, 0), 200, fill=False,edgecolor='k',linewidth=0.5,linestyle='-')
    ax[0].add_patch(circle1)
    ax[0].add_patch(circle2)

    #Plot Time Average Grid of FCD:
    #-------------------------------
    if len(lx.shape)>2:
        lx = lx[0]
        ly = ly[0]
    else:
        lx,ly = lx,ly

    cmap = plt.cm.Reds_r
    ctime = ax[0].contourf(lx*1e-3,ly*1e-3,np.log10(np.nancumsum(led,axis=0)[-1]),15,cmap=plt.cm.gray_r,alpha=0.6)
    ax[0].contour(lx*1e-3,ly*1e-3,np.log10(np.nancumsum(led,axis=0)[-1]),15,colors='k',linewidths=0.3,alpha=0.8)
    cax2 = fig.add_axes([.25, 0.33, 0.15, 0.02])
    tbar = plt.colorbar(ctime,cax=cax2,orientation='horizontal')
    tbar.ax.tick_params(labelsize=9,rotation=35)
    if sensor == 'GLM':
        tbar.ax.set_xlabel(r'$log_{10}$(GED Cumulative Sum)',fontsize=9)
    else:
        tbar.ax.set_xlabel(r'$log_{10}$(FED Cumulative Sum)',fontsize=9)


    #Scatter the Tracked Feature Centroids: Just plot the points instead of a fit for now
    order = 1 #2
    # storm_track = sns.regplot(good_centx*1e-3,good_centy*1e-3,ax=ax[0],order=order,color='k',scatter=False,ci=None,line_kws={'linewidth':4})
    storm_track = ax[0].plot(good_centx*1e-3,good_centy*1e-3,'.k',ms=5)

    #Set up plot to draw rectangles that indicate tracked feature:
    #-------------------------------
    colors = cmap(np.linspace(0,1,len(good_patch)))
    alphas = np.linspace(0.7,1,len(good_patch))

    for j,(p) in enumerate(zip(good_patch)):
        path = p[0].get_path()
        # Get the list of path vertices
        vertices = path.vertices.copy()
        # Transform the vertices so that they have the correct coordinates
        vertices = p[0].get_patch_transform().transform(vertices)
        get_x,get_y = [],[]
        for vx,vy in vertices:
            get_x.append(vx);get_y.append(vy)

        linestyle='-'
        width=.2
        ax[0].plot(get_x,get_y,color=colors[j],linestyle=linestyle,linewidth=width)

    #Plot Time Series
    #--------------------------------
    lw = 2
    axb = ax[1].twinx() #get second y-axis
    #make eccentricity data share shape as time frames
    set_ecc = np.zeros_like(t_frames)
    set_ecc[good_ids] = good_eccs
    #Plot eccentricity and FCD Totals in time:
    p1, = ax[1].plot(t_frames[good_ids],good_eccs,'k-',linewidth=lw,alpha=0.8)
    p2, = axb.plot(t_frames[skip_times:],np.nansum(led[skip_times:],axis=(1,2)),'C0-',linewidth=lw,alpha=0.8)

    #Plot thrid axis with feature area size--is convective region growing or shrinking?
    axc = ax[1].twinx()
    p3, = axc.plot(t_frames[good_ids],good_areas*(dx*dx),'C3-',linewidth=lw,alpha=0.8)
    # right, left, top, bottom
    axc.spines['right'].set_position(('outward', 60))


    #Axis Labels:
    #----------------
    ax[0].set_xlabel('X Distance [km]',fontsize=13)
    ax[0].set_ylabel('Y Distance [km]',fontsize=13)
    ax[0].tick_params(labelsize=13)

    ax[1].set_ylabel('Feature Eccentricity',fontsize=13)
    ax[1].set_xlabel('Time (s)',fontsize=13)
    ax[1].tick_params(labelsize=13,rotation=35)
    if sensor == 'GLM':
        axb.set_ylabel(f'GED ({frame_dt}' + ' '+ r'$\rm s^{-1}$)',fontsize=13)
    else:
        axb.set_ylabel(f'FED ({frame_dt}' + ' '+ r'$\rm s^{-1}$)',fontsize=13)

    axc.set_ylabel(r'Feature Area ($\rm km^2$)',fontsize=13)
    axb.yaxis.label.set_color(p2.get_color())
    axc.yaxis.label.set_color(p3.get_color())

    #Axis formatters:
    #----------------
    ax[0].set_xlim(-300,300)
    ax[0].set_ylim(-300,300)
    axb.set_xlim(t_frames.min(),t_frames.max())
    axb.set_ylim(np.nansum(led,axis=(1,2)).min()-50,np.nansum(led,axis=(1,2)).max()+50)

    ax[0].grid(alpha=0.2)
    ax[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'Output/{str(frame_dt)}/Overview/{case[2:]}/Track_Summary.png',dpi=200,bbox_inches='tight')

#---------------------------------------------
#Generate sample points, or sample point axis
#---------------------------------------------
def gen_sample_points(case,px,py,led,constant_feature,smooth_track,compress_ellipse,compression_dist,d,good_centx,good_centy,t_frames,get_valid_tidx,
                  get_centroidx,get_centroidy,set_width,set_height,get_norientation,get_semi_major_len,get_semi_minor_len,
                  xoffset,yoffset,frame_dt):

    '''Generate sampling points through the tracked ellipse semi-major axis from focal point to focal point. Note: an option to
       compress the distance from start to end sample points is available to ensure sample points stay within a more confined region of the tracked
       ellipse/tracked feature.

       Args:
           px,py = lightning gridded data x and y coords (changed from lx,ly since variables are used elsewhere in this method)
           led   = lightning gridded data
           constant_feature = boolean whether the tracked feature was of constant size, or adaptive of lightning signature size
           smooth_track = option to smooth the tracked centroids to avoid sample "jumping"
           compress_ellipse = option to reduce the span of the sampling line inwards from focal-point to focal-point
           compress_dist    = if compress_ellipse is True, set how much to compress the sample line inward (in km)
           d                = number of samples in line (can be any number, default is 10)
           good_centx       = tracked object centroid x-coordinates
           good_centy       = tracked object centroid y-coordinates
           t_frames         = analysis time frames
           get_valid_tidx   = ensure the correct tracked feature time frames are being used if any are skipped in the tracking routine
           get_centroidx    = tracked object centroid x-coord (may include tracking offset if choice is made in tracking routine)
           get_centroidy    = tracked object centroid y-coord (may include tracking offset if choice is made in tracking routine)
           set_width        = ellipse semi-minor axis length
           set_height       = ellipse semi_major axis length
           get_norientation = smooth tracked feature orientations
           get_semi_major_len= semi-major axis length of tracked object (!= set_width as these are determined by the identified region in
                               the sampling routine)
           get_semi_minor_len= see get_semi_major len
           xoffset,yoffset   = For iterating and generating (n) number of sampliing lines, can offset the centroids to move sampling line forwards
                               or backwards along storm path.

     Returns:
           line_rot = sampling line object (n_frames,n_points) [t_frames.shape,d]
           center_x = un/smoothed centroids x coords
           center_y = un/smoothed centroids ycoords
    '''

    #Define track center coordinates:
    #--------
    if smooth_track == True:
        print('Smoothing tracked storm centroids')
        #using a poly fit: (not good for smaller time frames)
        #-----
        #center_x = np.linspace(good_centx.min(),good_centx.max(),len(t_frames[get_valid_tidx]))*1e-3
        #poly_fit = np.poly1d(np.polyfit(good_centx*1e-3,good_centy*1e-3,2)) #2nd order polynomial fit
        #center_y = poly_fit(center_x)
        #center_r = np.sqrt(center_x**2. + center_y**2.) #for approxmate path

        #Smooth using gaussian filter -- works better even for smaller time frame dt:
        #-----
        #rule of thumb, use 100 for 1 min, 10 for 10 min, 20 for 5 min
        if frame_dt == '1min':
            alpha = 100
        elif frame_dt == '5min':
            alpha = 20
        elif frame_dt == '10min':
            alpha = 10
        center_x = gf(good_centx*1e-3,alpha) #eg 60 points on either side to really smooth it out (60*60 = 60 min window)
        center_y = gf(good_centy*1e-3,alpha) #Set to some number of points before and after
    else:
        center_x = np.array(get_centroidx)*1e-3
        center_y = np.array(get_centroidy)*1e-3

    #Shift points by some distance: (can iterate over a list of points to generate mulitple lines)
    center_x = center_x + xoffset
    center_y = center_y + yoffset

    #From the storm track, define a sampling line with points from the center of the track spaced a x km both directions.
    #First start with new ellipse:
    #-----
    if compress_ellipse == True:
        print('Compressing tracking ellipse to ensure points stay within storm')
        reduce = compression_dist #km - reduce semi-major axis by 100 km (or another value) to ensure points stay in storm at all times.
        print('Reducing semi-major axis length by {0} km'.format(reduce))
    else:
        reduce = 0 #km

    if constant_feature == True:
        print('Setting a constant ellipse to sample storm (no expansion/compression with storm growth/decay)',set_height)
        track_ellipse = np.array([Ellipse((center_x[i],center_y[i]),
                                          width =set_width,#get_semi_major_len[i],
                                          height=set_height,#get_semi_minor_len[i],
                                          angle=get_norientation[i]) for i in range(len(t_frames[get_valid_tidx]))])
        major_axis    = np.repeat(set_width ,len(get_semi_major_len))
        minor_axis    = np.repeat(set_height,len(get_semi_minor_len))
    else:
        track_ellipse = np.array([Ellipse((center_x[i],center_y[i]),
                                          width =get_semi_major_len[i]-reduce,
                                          height=get_semi_minor_len[i],
                                          angle=get_norientation[i]) for i in range(len(t_frames[get_valid_tidx]))])
        major_axis    = np.array(get_semi_major_len)
        minor_axis    = np.array(get_semi_minor_len)
    #Ellipse Coordinates for plotting:
    #-----
    ecoords       = np.array([track_ellipse[i].get_patch_transform().transform(track_ellipse[i].get_path().vertices[:-1]) for i in range(len(t_frames[get_valid_tidx]))])

    #Get ellipse focal points (or specifc vertices) for defining sample line:
    #-----
    focal_pta = np.sqrt((major_axis-reduce)**2. - minor_axis**2.)

    ####################################
    # UPDATES BELOW: (06/22/23)        #
    ####################################
    #NOTE HERE: dividing the focal_pta by 2 will push the sample positions inwards, set to 1 if that is not desired (won't affect constant widths and lengths)

    #For OKLMA CASES use the following: #!!!!!!!!!!!!!!!!!CHANGE
    pxa0,pxb0 = center_x + focal_pta/2, center_x - focal_pta/2
    pya0,pyb0 = center_y, center_y

    #pya0,pyb0 = center_y + focal_pta/2, center_y - focal_pta/2
    #pxa0,pxb0 = center_x, center_x

    #For PERiLS CASES use the following (change if E-W oriented)
    #pya0,pyb0 = center_y + focal_pta/2, center_y - focal_pta/2
    #pxa0,pxb0 = center_x, center_x

    # #rotate coords wrt ellipse: #!!!!!!!!!!!!!!!!!!!!!!CHANGE--Use these points only for when orientation is N-->S
    el_th = np.deg2rad(np.array(get_norientation)) #Had to rotate by -90 when changing the adjusted focal point length along y, remove -90 for OKLMA cases
    p_rot = [rotate((center_x[i],center_y[i]),(pxa0[i],pya0[i]),el_th[i]) for i in range(len(t_frames[get_valid_tidx]))]
    pxa   = np.array([p[0] for p in p_rot])
    pya   = np.array([p[1] for p in p_rot])
    p_rotb = [rotate((center_x[i],center_y[i]),(pxb0[i],pyb0[i]),el_th[i]) for i in range(len(t_frames[get_valid_tidx]))]
    pxb   = np.array([p[0] for p in p_rotb])
    pyb   = np.array([p[1] for p in p_rotb])


    #Get points along line to sample fields:
    line_rot = []
    for i in range(len(pxa0)):
        if pxb0[i] < 0:
            x1 = pxa0[i] #If OKLMA-210612, use pxa,pxb,pya,pyb only!!!! else use pxa0 and pxb0
            x2 = pxb0[i] #If OKLMA-210612 use pxa,pxb,pya,pyb only!!!!
        else:
            x1 = pxa0[i] #If OKLMA, use pxa,pxb,pya,pyb only!!!!
            x2 = pxb0[i] #If OKLMA, use pxa,pxb,pya,pyb only!!!!
        lx = np.linspace(x2,x1,d) #<---USE!
#         lx = np.linspace(pxb0[i],pxa0[i],d)

        #Updated how lines are drawn from focal points, for OKLMA cases, just repeat lower coordinate, for PERiLS draw from south to north by 10 points
        ly = np.linspace(pya0[i],pyb0[i],len(lx)) #PERILS #UPDATE 08-17-22: revesred order of y locations from np.linspace(pyb0,pya0,len(lx)) to np.linspace(pya0,pyb0,len(lx))
        #ly = np.repeat(pyb0[i],len(lx)) #OKLMA--wouldn't recommend using this at all
        for j in range(d):
            center_pt = (center_x[i],center_y[i])
            line_x,line_y = lx[j],ly[j]
            line_rot.append(rotate(center_pt,(line_x,line_y),el_th[i]))
    line_rot = np.array(line_rot).reshape(len(t_frames[get_valid_tidx]),d,2)


    if len(px.shape)>2:
        px,py = px[0],py[0]
    else:
        px,py = px,py

    #PLOTTING:------------------------------------

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    # ax.plot(center_x,center_y,'+k')
    ax.plot(center_x,center_y,linestyle='--',color='k',ms=10)
    ax.contourf(px*1e-3,py*1e-3,np.log10(np.nancumsum(led,axis=0)[-1]),15,cmap=plt.cm.gray_r,alpha=0.4)
    #Save out ellipse verts:
    ellipse_x,ellipse_y = [],[] #Added 04-18-22
    for j in range(len(t_frames[get_valid_tidx])):
        vxs,vys = [],[]
        for vx,vy in zip(ecoords[j,:,0],ecoords[j,:,1]):
            vxs.append(vx);vys.append(vy)
        ax.plot(vxs,vys,color='k',linestyle='-',linewidth=0.7,alpha=0.5)
        ellipse_x.append(vxs);ellipse_y.append(vys)

    for (px1,py1,px2,py2) in zip(pxa,pya,pxb,pyb):
        ax.plot([px1,px2],[py1,py2],'C3-',linewidth=.5)

    for t in range(len(t_frames[get_valid_tidx])):
        colors = plt.cm.Reds(np.linspace(0,1,line_rot[0,:,0].shape[0]))

        for l in line_rot[t]:
            ax.scatter(l[0],l[1],marker='+',edgecolor='k',alpha=0.8,linewidth=2,c='k')

    circle1 = plt.Circle((0, 0), 100, fill=False,edgecolor='k',linewidth=.3,linestyle='-')
    circle2 = plt.Circle((0, 0), 200, fill=False,edgecolor='k',linewidth=.3,linestyle='-')
    circle3 = plt.Circle((0, 0), 300, fill=False,edgecolor='k',linewidth=.3,linestyle='-')

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    circle1 = plt.Circle((0, 0), 100, fill=False,edgecolor='k',linewidth=.3,linestyle=':')
    circle2 = plt.Circle((0, 0), 200, fill=False,edgecolor='k',linewidth=.3,linestyle=':')
    circle3 = plt.Circle((0, 0), 300, fill=False,edgecolor='k',linewidth=.3,linestyle=':')

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    ax.set_ylabel('Y-Distance (km)')
    ax.set_xlabel('X-Distance (km)')
    ax.set_xlim(-360,360)
    ax.set_ylim(-360,360)
    ax.grid(alpha=0.5,linewidth=.5)
    plt.savefig(f'Output/{str(frame_dt)}/Overview/{case[2:]}/SampleSegmentGen_{str(frame_dt)}.png',dpi=160,bbox_inches='tight')

    return(line_rot,center_x,center_y,ellipse_x,ellipse_y)
