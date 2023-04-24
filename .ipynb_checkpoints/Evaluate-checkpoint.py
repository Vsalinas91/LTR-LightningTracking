############################################################################################
#EVALUATION.PY
#--------------
#The contents of Evaluation.py is to provide the functions used to generate
#the LTR evaluation figures and statistics. This script handles the following:
#
#    -) LTR evaluation is first done by examining why it fails in certain instances
#       using fine time resolution FED datasets. This makes use of the OKLMA cases only,
#       excluding OK-0223.
#    -) Using all case data, evaluation metrics are computed and their averages through
#       time are used to compare with past studies using similar evaluation methods.
#
#The code is organized as follows:
#    1st Block) Data handler
#    2nd Block) Regression and RMSE functions
#    3rd Block) Plotting Functions
#
#Author: Vicente Salinas
#Contact: vicente.salinas@noaa.gov
#Updated: 04/18/23
############################################################################################
#--------
#Imports
#--------
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import gaussian_filter as gf
from scipy.stats import linregress as lr


#################################################################
#I/O: Data Read In and Variable Set                             #
#################################################################
#-------------------------------------
#Functions for Data Reading:
#-------------------------------------
def get_data(case,dt):
    '''
    Get tracked ellipse and FOB data for specified case and time-frame resolution.
    
    Arguments: 
        -) case = string of the selected case (YYYYMMDD)
        -) dt   = FED time resolution [e.g., 1min,5min,10min]
    Returns:
        -) ellipse_dat = data containing ellipse verticies and centroids
        -) tracked_dat = data containing FOB centroids, orientations, eccentricities and areas
    '''
    ellipse_dat = xr.open_dataset(f'Output/{dt}/TrackedData/{case[2:]}/{case}_Tracked_Ellipse_Verts.nc')
    tracked_dat = pd.read_csv(    f'Output/{dt}/TrackedData/{case[2:]}/TRACKED-OBJECT-PROPS.csv')
    return(ellipse_dat,tracked_dat)

def get_fcounts(dataset,start,end):
    '''
    Get flash rate using flash sorted chunked data.
    Arguments:
        -) dataset = dataset name for chuncked/sampled flash sorted data
    '''
    #set things up from the chunked dataset
    area_all= np.hstack(dataset.lma_area.values)
    time_all= np.hstack(dataset.lma_time.values)
    #mask out all padded nan values
    nanmask = np.isfinite(time_all)
    #get flattened arrays
    areas   = area_all[nanmask]
    times   = time_all[nanmask]
    #mask out small singletons
    amask   = areas[np.argsort(times)]>4
    #get counts of flashes within each time frame
    in_time=np.array([times[amask] [(times[amask]>=tstart) & (times[amask]<tend)].shape[0] for tstart,tend in zip(start,end)])
    return(np.array(in_time))


def data_to_compare(tracked_data):
    '''
    Get variables to generate figures used to compare geometric properties of tracked 
    ellipse and flash objects (FOB) across time-resolutions.
    
    Argument(s):
        -) tracked_data = data for tracked object (TRACKED-OBJECT-PROPS.csv)
    Returns:
        -) radial = vector magnitude of centroid positions relative to domain origin
        -) time   = time frame time in seconds since start of day
        -) area   = tracked FOB area in units km^2
        -) angle  = smoothed orientation of FOB
    '''
    radial= np.sqrt(tracked_data.cx**2. + tracked_data.cy**2.)
    time  = tracked_data.time
    area  = tracked_data.area
    ecc   = tracked_data.ecc
    angle = tracked_data.angle
    return(radial,time,area,ecc,angle)
        
#-----------------------------------------------
#Evaluation Figures and Functions for All Cases
#-----------------------------------------------
def all_cases(case):
    '''
    Get tracked FOB and Ellipse geometric data for all cases + PERiLS cases used to 
    evaluate the LTR. Here 5 minute time interval data are used as they fall between
    both 1 and 10 minutes and are in range of typical data used in real-time systems.
    
    Argument: case = case name in YYMMDD string format
    '''
    case_full    = '20'+case
    ellipse_file = f'Output/5min/TrackedData/{case}/{case_full}_Tracked_Ellipse_Verts.nc'
    case   = pd.read_csv(    f'Output/5min/TrackedData/{case}/TRACKED-OBJECT-PROPS.csv')
    ellipse= xr.open_dataset(ellipse_file)
    return(case,ellipse)


def case_durations_centroids(case,ellipse,istart,istop):
    '''
    This function provides the case specific LTR centroids and their tracked durations.
    Arguments:
        -) case = tracked FOB data
        -) istart= starting time frame
        -) istop = stopped time frame
        
    istop and istart are provided for PERiLS cases as there are times when the LTR failed, and so 
    to prevent issues with the evaluation statistics, certain time frames must be skipped--those
    values are set in the notebook.
    
    Returns:
        -) cx,cy = smoothed centroids
        -) dur   = tracked durations
    '''
    cx = np.array([np.nanmean(ellipse.ellipse_x[i]) for i in range(len(ellipse.ellipse_x))])[istart:istop]
    cy = np.array([np.nanmean(ellipse.ellipse_y[i]) for i in range(len(ellipse.ellipse_x))])[istart:istop]
    dur   = case.time[istart:istop].max() - case.time[istart:istop].min()
    return(cx,cy,dur)


#################################################################
#Statistics and Regression Analyses                             #
#################################################################
#---------------------------------
#Define Regression for LTR tracks
#---------------------------------
def fit_line(cx,cy):
    '''
    Define linear regression to assume linear storm track.
    Arguments: 
        -) cx,cy = smoothed FOB centroids
    Returns:
        -) lin_reg = list of regression information including slope, intercept, r-value and p-value
    '''
    lin_reg = lr(cx,cy)
    return(lin_reg)

def reg_eq(cx,cy):
    '''
    Define regression equation using x-coordinates of track centroids.
    Returns:
        -) x coordinates of centroids
        -) eq = equation, or predicted y coordinate values
    '''
    s = fit_line(cx*1e-3,cy*1e-3)
    slope    = s[0]
    intercept= s[1]
    eq= (slope*cx*1e-3)+intercept
    return(cx*1e-3,eq)

#------------------------------
#Define evaluation statistics
#------------------------------
def rmse(x,y,xp,yp):
    '''
    Calculate the root mean square error between smoothed and linear tracked centroids (Laksmnanan and Smith, 2010).
    Returns:
        rms = or rmse of the two tracks.
    '''
    r_actual = np.sqrt(x**2. + y**2)
    r_predict = np.sqrt(xp**2 + yp**2.)
    rms = np.sqrt(np.sum(np.abs(r_predict-r_actual)**2.) / len(r_predict))
    return(rms)

#################################################################
#FIGURES: Generate all Evaluation Figures                       #
#################################################################
#-----------------------------------------------------------------------
#Plot tracking geometric variables for 1, 5, and 10 min time resolutions
#-----------------------------------------------------------------------
def geo_compare(case,dts,times,dists,areas,eccs,angles,centroids,save):
    '''
    This function plots a collection of time series for the tracked FOB data to compare
    them across different FED time intervals. In addition, the tracked FOB centroids are 
    also plotted to examine the sensitivity of using FEDs generated using different time
    intervals for tracking consistency.
    
    Arguments:
        -) case = case name YYYYMMDD
        -) dts  = list of time resolutions [1min,5min,10min]
        -) times= tuple of all time frames from the dts list
        -) dists= tuple of all centroid distances from OKLMA network center
        -) areas= tuple of all FOB areas
        -) eccs = tuple of all FOB eccentricities
        -) angles= tuple of all FOB smoothed orientations
        -) centroids= dataframe with FOB centroids
        -) save = Boolean to save figure
    '''
    t1,t5,t10 = times
    r1,r5,r10 = dists
    a1,a5,a10 = areas
    e1,e5,e10 = eccs
    ag1,ag5,ag10= angles
    c1,c5,c10 = centroids
    
    
    fig,ax = plt.subplots(2,2,figsize=(12,10))
    #Plot time series of distances from LMA origin
    one, = ax[0,0].plot(t1 ,r1*1e-3 ,'C0')
    five,= ax[0,0].plot(t5 ,r5*1e-3 ,'C3')
    ten, = ax[0,0].plot(t10,r10*1e-3,'C1')
    ax[0,0].legend([one,five,ten],dts,loc='lower right')
    ax[0,0].set_ylabel('Object Distance from OKLMA (km)',fontsize=13)
    
    #Plot time series of FOB areas
    axb = ax[0,1].twinx()
    ax[0,1].plot(t1,a1  ,'C0' ,linewidth=2)
    ax[0,1].plot(t5,a5  ,'C3' ,linewidth=2)
    ax[0,1].plot(t10,a10,'C1', linewidth=2)
    if case=='20211027':
        ax[0,1].set_ylim(1e3,6.5e3)
        axb.set_ylim(0,1)
    else:
        axb.set_ylim(.6,1)
    #Plot time series of FOB Eccentricities with the areas
    oneb,  = axb.plot(t1,e1  ,color='C0',linestyle=(0, (7, .75 )),linewidth=.75,alpha=0.8)
    fiveb, = axb.plot(t5,e5  ,color='C3',linestyle=(0, (7, .75 )),linewidth=.75,alpha=0.8)
    tenb,  = axb.plot(t10,e10,color='C1',linestyle=(0, (7, .75 )),linewidth=.75,alpha=0.8)
    ax[0,1].legend([one,five,ten,oneb,fiveb,tenb],dts+dts,ncol=2,loc='lower center')
    ax[0,1].set_ylabel(r'Object Area ($\rm km^2$)',fontsize=13)
    axb.set_ylabel('Object Eccentricity',fontsize=13)
    
    
    #Plot time series of average FOB area
    ax[1,0].plot(t1,ag1,'C0')
    ax[1,0].plot(t5,ag5,'C3')
    ax[1,0].plot(t10,ag10,'C1')
    ax[1,0].legend([one,five,ten],dts)
    ax[1,0].set_ylabel(r'Object Orientation ($\circ$)',fontsize=13)
    
    #Plot smoothed FOB centroids
    sigmas = [20*5,20,10]
    ax[1,1].scatter(gf(c1.cx,sigmas[0]) *1e-3,gf(c1.cy,sigmas[0]) *1e-3,marker='.'  ,color='C0')
    ax[1,1].scatter(gf(c5.cx,sigmas[1]) *1e-3,gf(c5.cy,sigmas[1]) *1e-3,marker='.'  ,color='C3')
    ax[1,1].scatter(gf(c10.cx,sigmas[2])*1e-3,gf(c10.cy,sigmas[2])*1e-3,marker='.',color='C1')
    ax[1,1].legend([one,five,ten],[dts[i]+' Centroids' for i in range(len(dts))])
    ax[1,1].set_xlim(-150,150)
    ax[1,1].set_ylim(-150,150)
    
    #Axis formatting
    #---------------
    #Xlabel for time series
    [axs.set_xlabel('Time (s)',fontsize=13) for axs in ax.flatten()] 
    [axs.set_xlim(t10.min(),t10.max()) for i,axs in enumerate(ax.flatten()) if i < 3]
    #Draw grid for time series panels
    [axs.grid(axis='x') for i,axs in enumerate(ax.flatten()) if i < 3]
    #Draw circles to represent origin around OKLMA center
    circle1 = plt.Circle((0, 0), 50 , color='gray',fill=False,alpha=0.7)
    circle2 = plt.Circle((0, 0), 100, color='gray',fill=False,alpha=0.7)
    ax[1,1].add_patch(circle1)
    ax[1,1].add_patch(circle2)
    ax[1,1].grid()
    #Set axis labels for centroid panel
    ax[1,1].set_xlabel('X Distance (km)',fontsize=13)
    ax[1,1].set_ylabel('Y Distance (km)',fontsize=13)
    #Tick param text size
    [axs.tick_params(labelsize=13) for axs in ax.flatten()]
    axb.tick_params(labelsize=13)
    
    #Annotate
    labels = ['A)','B)','C)','D)']
    [axs.text(.01,.9,l,color='C3',weight='bold',transform=axs.transAxes,fontsize=15) for (axs,l) in zip(ax.flatten(),labels)]
    plt.tight_layout()
    if save==True:
        plt.savefig(f'PaperFigures/TRACK_COMPARE-OK{case[2:]}.pdf',dpi=160,bbox_inches='tight')
        
#------------------------------------------------------------------------------------------------------------
#Plot annotated figure for identifing errors with FOB orientation in high (1minute) time resolution FED data
#------------------------------------------------------------------------------------------------------------
def orientation_errors(c1,c1b,in_time1,in_time2,save):
    '''
    This function illustrates where errors in the FOB orientation can be found, in terms of times at which 
    their sizes are small, for both OK cases when using 1 minute time resolution FED datasets. All reasoning 
    are annotated in both panels A and B to demonstrate the sensitivity of using a 1 minute resolution.
    
    Arguments:
        -) c1,c1b   = dataframes with geometric data of the tracked FOBs
        -) in_time1 = time frames for case1
        -) in_time2 = time frames for case2
    '''
    fig,ax = plt.subplots(2,1,figsize=(5,11))


    color1 = ax[0].scatter(np.gradient(c1b.angle,60),c1b.area,marker='o',c=in_time1,#c1b.time,
                           edgecolor='k',linewidth=0.4,cmap=plt.cm.Spectral_r)
    cbar1 = plt.colorbar(color1,ax=ax[0])

    color2 = ax[1].scatter(np.gradient(c1.angle,60),c1.area,marker='o',c=in_time2,#c1.time,
                           edgecolor='k',linewidth=0.4,cmap=plt.cm.Spectral_r)
    cbar2 = plt.colorbar(color2,ax=ax[1])

    [axs.grid(alpha=.2,color='k') for axs in ax.flatten()]
    [axs.tick_params(labelsize=13) for axs in ax.flatten()]
    [axs.set_ylabel(r'Object Area $\rm (km^2)$' ,fontsize=16) for axs in ax.flatten()]
    [axs.set_xlabel(r'$\rm \partial_t \Theta \ (\circ/min)$',fontsize=16) for axs in ax.flatten()]
    ax[0].text(.02,.9,'A) OK-210612',transform=ax[0].transAxes,fontsize=15,color='C3',weight='bold')
    ax[1].text(.02,.9,'B) OK-211027',transform=ax[1].transAxes,fontsize=15,color='C3',weight='bold')

    ax[0].annotate('Quadrant Shift', 
                   xy=(-1.4,1500),  
                   xycoords='data',
                   xytext=(0.1, 0.4), 
                   textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05,width=3),
                   transform=ax[0].transAxes,weight='bold'
                )
    ax[0].annotate('', 
                   xy=(1.4,1500),  
                   xycoords='data',
                   xytext=(0.45, 0.4), 
                   textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05,width=3),
                   transform=ax[0].transAxes
                )


    ax[1].annotate('Object Size Change', 
                   xy=(-1.,2100),  
                   xycoords='data',
                   xytext=(0.02, 0.5), 
                   textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05,width=3),
                   transform=ax[0].transAxes,
                   weight='bold'
                )
    ax[1].annotate(' ', 
                   xy=(.35,2700),  
                   xycoords='data',
                   xytext=(0.5, 0.5), 
                   textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05,width=3),
                   transform=ax[0].transAxes
                )
    ax[1].annotate(' ', 
                   xy=(1.1,2500),  
                   xycoords='data',
                   xytext=(0.5, 0.5), 
                   textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05,width=3),
                   transform=ax[0].transAxes
                )


    ax[1].annotate(' ',#'Merging two objects', 
                   xy=(-.4,2500),  
                   xycoords='data',
    #                xytext=(0.01, 0.67), 
                   xytext=(0.02,0.5),
                   textcoords='axes fraction',
                   arrowprops=dict(facecolor='k', shrink=0.05,width=3),
                   transform=ax[0].transAxes,
                   color='k',
                   weight='bold'
                )


    cbar1.ax.set_ylabel(r'Flash Rate ($\rm s^{-1}$)',fontsize=15)
    cbar2.ax.set_ylabel(r'Flash Rate ($\rm s^{-1}$)',fontsize=15)
    ax[0].set_title(r'Time Interval $\Delta t$=1 minute',fontsize=13)
    if save==True:
        plt.savefig('PaperFigures/OBJECT_SHIFT_1min_BOTH.pdf',dpi=160,bbox_inches='tight')


#------------------------
#Tracked Centroid Figure
#------------------------
def centroid_plots(cases,all_x,all_y,all_reg,save):
    '''
    This function plots the smoothed LTR FOB tracks for all cases in panel A) for comparison to linear storm tracks
    as defined using a linear regression in panel B). The purpose of this comparison is to demonstrate how close
    to a line storm tracks follow, as suggested in Lakshmanan and Smith, (2010).
    
    Arguments:
        -) cases = list of all case names
        -) all_x = all x centroid coordinates
        -) all_y = all y centroid coordinates
        -) all_reg= all regression fits to predict the y coordinates.
    '''
    fig,ax = plt.subplots(1,2,figsize=(8.5,4))
    #Set array of colors to distinguish between cases
    colors = plt.cm.magma(np.linspace(0,.9,len(all_reg))[:])
    #Plot smoothed FOB centroids
    [ax[0].plot(cx,cy,color='k',linewidth=4,alpha=0.5) for cx,cy in zip(all_x,all_y)]
    #Plot linear tracks
    [ax[1].plot(x,reg,color='k', linewidth=4,alpha=0.5) for (x,reg) in zip(all_x,all_reg)]
    [ax[1].plot(x,reg,color=c,   linewidth=3,alpha=1.0) for (x,reg,c) in zip(all_x,all_reg,colors)]
    all_lines = []
    for (cx,cy,c) in zip(all_x,all_y,colors):
        lines, = ax[0].plot(cx,cy,color=c,linewidth=3)
        all_lines.append(lines)
    
    #Panel legends
    ax[0].legend(all_lines,cases)
    ax[1].legend(all_lines,cases)
    #panel titles
    ax[0].set_title('Tracked Centroids')
    ax[1].set_title('Linear Fit')
    
    #Axis formatting
    [axs.set_xlim(-210e0,210e0) for axs in ax.flatten()]
    [axs.set_ylim(-200e0,200e0) for axs in ax.flatten()]
    [axs.grid(alpha=0.3) for axs in ax.flatten()]
    [axs.scatter(0,0,marker='o',color='',edgecolor='black',s=12000,linewidth=.2) for axs in ax.flatten()]
    [axs.scatter(0,0,marker='+',color='k',edgecolor='black',s=100,linewidth=1) for axs in ax.flatten()]
    [axs.set_xlabel('X Distance (km)') for axs in ax.flatten()]
    [axs.set_ylabel('Y Distance (km)') for axs in ax.flatten()]
    
    #Set panel names
    labels = ['A)','B)']
    [axs.text(.03,.9,l,fontsize=18,color='C3',weight='bold',transform=axs.transAxes) for axs,l in zip(ax.flatten(),labels)]
    plt.tight_layout()
    
    if save==True:
        plt.savefig('PaperFigures/ALL_CASE_TRACKS_COMPARE.pdf',dpi=200,bbox_inches='tight')
        
#----------------------------------------------
#LTR Evaluation Figure from Evaluation Metrics
#----------------------------------------------
def ltr_eval(cases,all_dur,all_rmse,all_rmse_tri,all_std,all_diffs,save):
    '''
    This function is used to plot the LTR evaluation statistics from the following metrics:
        -) Mean track duration
        -) Mean RMSE (Smoothed vs Linear)
        -) Mean RMSE (Smoothed vs Polynomial)
        -) Mean FOB Area Standard Deviation
        -) Mean FOB track offset--difference between centroid locations in time
        
    The purpose of each of these comparisons is detailed in the paper, but are to:
        1) Evaluate how long the LTR can track a storm system relative to previous studies (minimum of 3 hours)
        2) Evaluate how linear the tracks are
        3) Confirm if track are best described by line, or curve
        4) Evaluate the consistency of the LTR in identifying an object describing the storm system's shape
        5) Evaluate the consistency of the LTR track itself; does it deviate little in time?
        
    Arguments:
        -) cases        = list of case names
        -) all_dur      = all case track total durations
        -) all_rmse     = all RMSE for smoothed vs linear storm tracks
        -) all_rmse_tri = all RMSE for smoothed vs polynomial storm tracks
        -) all_std      = all FOB area standard deviations
        -) all_diffs    = all centroid position differences
    '''
    fig,ax = plt.subplots(3,2,figsize=(6,7))
    ax[0,0].bar(x=np.arange(len(all_dur)),height=all_dur.T/60/60,#yerr=(all_dur.T/60/60) - np.nanmean(all_dur.T/60/60),
              color=plt.cm.gray(np.linspace(0.5,1,len(all_dur))),edgecolor='k',alpha=0.7,hatch='\\')
    ax[0,0].set_xticks(np.arange(0,len(all_dur)))
    ax[0,0].set_xticklabels(cases,rotation=45)
    ax[0,0].set_ylabel('Track Duration (hours)')
    
    ax[0,1].bar(x=np.arange(len(all_dur)),height=all_rmse.T*1e0,#yerr = all_rmse.T*1e-3 - np.nanmean(all_rmse*1e-3),
              color=plt.cm.gray(np.linspace(0.5,1,len(all_dur))),edgecolor='k',alpha=0.7,hatch='\\')
    ax[0,1].set_xticks(np.arange(0,len(all_dur)))
    ax[0,1].set_xticklabels(cases,rotation=45)
    ax[0,1].set_ylabel('Track Linearity (km)')
    ax[0,1].set_ylim(0,8)
    
    
    ax[1,0].bar(x=np.arange(len(all_dur)),height=all_rmse_tri.T*1e0,#yerr=all_rmse_tri.T*1e-3 - np.nanmean(all_rmse_tri*1e-3),
                color=plt.cm.gray(np.linspace(0.5,1,len(all_dur))),edgecolor='k',alpha=0.7,hatch='\\')
    ax[1,0].set_xticks(np.arange(0,len(all_dur)))
    ax[1,0].set_xticklabels(cases,rotation=45)
    ax[1,0].set_ylabel('Track Non-Linearity (km)')
    ax[1,0].set_ylim(0,8)
    
    
    ax[1,1].bar(x=np.arange(len(all_dur)),height=(all_rmse.T*1e0) - (all_rmse_tri.T*1e0),
                color=plt.cm.gray(np.linspace(0.5,1,len(all_dur))),edgecolor='k',alpha=0.7,hatch='\\')
    ax[1,1].set_xticks(np.arange(0,len(all_dur)))
    ax[1,1].set_xticklabels(cases,rotation=45)
    ax[1,1].set_ylabel('Linearity Error (km)')
    ax[1,1].set_ylim(0,None)
    
    
    ax[2,0].bar(x=np.arange(len(all_dur)),height=(all_std.T*1e-3),
                color=plt.cm.gray(np.linspace(0.5,1,len(all_dur))),edgecolor='k',alpha=0.7,hatch='\\')
    ax[2,0].set_xticks(np.arange(0,len(all_dur)))
    ax[2,0].set_xticklabels(cases,rotation=45)
    ax[2,0].set_ylabel(r'$\sigma_{A}$ ($\rm km^2$)')
    ax[2,0].set_ylim(0,10)
    
    
    ax[2,1].bar(x=np.arange(len(all_dur)),height=(all_diffs.T),
                color=plt.cm.gray(np.linspace(0.5,1,len(all_dur))),edgecolor='k',alpha=0.7,hatch='\\')
    ax[2,1].set_xticks(np.arange(0,len(all_dur)))
    ax[2,1].set_xticklabels(cases,rotation=45)
    ax[2,1].set_ylabel(r'$\rm d_{centroid}$ ($\rm km$)')
    ax[2,1].set_ylim(0,None)
    
    [axs.set_xlabel('Case') for axs in ax.flatten()]
    [axs.grid(alpha=0.4) for axs in ax.flatten()]
    labels = ['A)','B)','C)','D)','E)','F)']
    [axs.text(.83,.8,l,fontsize=15,color='C3',weight='bold',transform=axs.transAxes) for axs,l in zip(ax.flatten(),labels)]
    plt.tight_layout()
    if save == True:
        plt.savefig('PaperFigures/ALL_CASE_STATISTICAL_EVAL.pdf',dpi=200,bbox_inches='tight')