from pyxlma.lmalib.io import read
from pyxlma.lmalib.flash.cluster import cluster_flashes
from pyxlma.lmalib.flash.properties import flash_stats,filter_flashes
from pyxlma.lmalib.io import read
#from GLMDataPull import GLMRequest
import numpy as np
import gzip

import glob
import datetime as dt
import calendar
import os


#Note: if day cross-over exists, throw all data into a single raw data directory with the start_date as the name
class LMAGrabber(object):
    def __init__(self,network,base_date,end_time,reprocess,sort_params):
        self.network = network
        self.base_date = base_date
        self.end_time  = end_time
        self.reprocess = reprocess
        self.sort_params=sort_params

    def trim_valid(self):
        #only get files with events else pandas throws an error
        self.get_good = []
        for f in self.valid_files:
            with gzip.open(f) as valid_file:
                for line in valid_file:
                    if 'Number of events' in line.decode():
                        events = int(line.strip().split()[-1])
                        if events > 0:
                            good_file = f
                            self.get_good.append(good_file)


    def get_lma_files(self):
        year = self.base_date.year
        month= self.base_date.month
        day  = str(self.base_date.day).zfill(2)
        cmonth = calendar.month_abbr[month]
        '''CHANGE THE PATH!!!'''
        # self.lma_data = sorted(glob.glob(f'/Users/admin/Desktop/PERiLS_Analysis/LMA/processed/{self.network}/{year}/{cmonth}/{day}/*.dat.gz'))
        self.lma_data = sorted(glob.glob(f'/Users/admin/Desktop/PERiLS_Analysis/LMA/processed/{self.network}/{year}/{cmonth}/{day}/*.dat.gz'))

    def lma_times(self):
        self.get_lma_files()
        self.lma_file_times = np.array([self.lma_data[i][-18:-18+6] for i in range(len(self.lma_data))])
        self.lma_file_date  = np.array([self.lma_data[i][-25:-25+6] for i in range(len(self.lma_data))])
        lma_file_fmt = []
        for i in range(len(self.lma_file_times)):
            f_day = int(self.lma_file_date[i][4:])
            if f_day == self.base_date.day:
                date_fmt = dt.datetime(self.base_date.year,self.base_date.month,self.base_date.day,
                                      int(self.lma_file_times[i][:2]),
                                      int(self.lma_file_times[i][2:4]),
                                      int(self.lma_file_times[i][4:]))
            elif f_day == self.base_date.day+1:
                date_fmt = dt.datetime(self.base_date.year,self.base_date.month,self.base_date.day+1,
                                      int(self.lma_file_times[i][:2]),
                                      int(self.lma_file_times[i][2:4]),
                                      int(self.lma_file_times[i][4:]))
            lma_file_fmt.append(date_fmt)
        self.lma_file_fmt = np.array(lma_file_fmt)
        # self.lma_file_fmt   = np.array([dt.datetime(self.base_date.year,self.base_date.month,self.base_date.day+add_day,
        #                       int(self.lma_file_times[i][:2]),
        #                       int(self.lma_file_times[i][2:4]),
        #                       int(self.lma_file_times[i][4:]))
        #                       for i in range(len(self.lma_file_times))])
        self.lma_valid_times = np.array([i for i in self.lma_file_fmt if ((i>=self.base_date) & (i<=self.end_time))])
        self.lma_sec         = np.array([(self.lma_valid_times[i] - self.base_date).total_seconds() for i in range(len(self.lma_valid_times))])
        self.t_bnds          = (self.lma_valid_times[0],self.lma_valid_times[-1])
        self.valid_files     = []
        for n, (i,j) in enumerate(zip(self.lma_data,self.lma_file_fmt)):
            if (j>=self.base_date) & (j<=self.end_time):
                self.valid_files.append(self.lma_data[n])

    def flash_sort(self):
        self.lma_times()
        self.parse_date = self.base_date.strftime('%Y-%m-%d')
        if self.reprocess == True:
            self.sorted_path='sorted/'+self.parse_date+'-'+f'{network}_QC.nc'
        else:
            self.sorted_path='sorted/'+self.parse_date+'-'+f'{network}.nc'
        if os.path.exists(self.sorted_path):
            print('Flash file already exists.')
            # self.ds = xr.open_dataset(self.sorted_path)
            pass
        else:
            print('Starting flash sorting')
            self.trim_valid()
            #print(self.valid_files)
            self.lma_data,self.starttime = read.dataset(self.get_good)
            if self.reprocess == False:
                self.ds = cluster_flashes(self.lma_data)
                print('Flashes sorted, populating flash stats.')
                self.ds = flash_stats(self.ds)
                print('Saving flash sorted data.')
                self.ds.to_netcdf(self.sorted_path)
            else:
                print('Pruning Source Data with Flash Sorting Filters.')
                good_events = (self.lma_data.event_stations >= self.sort_params['stationsmin']) & (self.lma_data.event_chi2 <= self.sort_params['chi2max'])
                self.lma_data = self.lma_data[{'number_of_events':good_events}] #re-make Level2 data with applied filters
                print('Starting Flash sorting.')
                self.ds     = cluster_flashes(self.lma_data)
                print('Flashes Sorted -- Starting stats population.')
                self.ds     = flash_stats(self.ds)
                print('Filtering min events per flash.')
                self.ds     = filter_flashes(self.ds, flash_event_count=(self.sort_params['min_events_per_flash'],None))
                print('Saving flash sorted data.')
                self.ds.to_netcdf(self.sorted_path)

if __name__ == '__main__':
    #Read in gridding parameters--shared across GLM and LMA gridding/processing routines
    '''CHANGE THE PATH !!!'''
    target      ='/Users/admin/Desktop/PERiLS_Analysis/FlashGriddingParams_20230405.txt'
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

    #For reference
    year = case[:4]
    month= case[4:6]
    day  = case[6:]
    print('Setting up flash sorting routine.')
    print('----------')
    print('Setting up flash sorting for Case: {0}'.format(case))
    shh,ehh   = int(start[:2]), int(end[:2])
    smm,emm   = int(start[2:4]),int(end[2:4])
    sss,ess   = int(start[4:]), int(end[4:])
    #Set up datetime objects
    #added 03/28/22 to handle day crossovers
    if xover==True: #check to see if day overlap exists, boolean in param file based on when storms entered and exited domain
        shift_day = 1
    else:
        shift_day = 0
    base_date = dt.datetime(int(year),int(month),int(day),shh,smm,sss) #starting period
    try:
        end_time  = dt.datetime(int(year),int(month),int(day)+shift_day,ehh,emm,ess) #ending period--can be day +1 if day crossover exists
    except:
        #Start new month if day cross-over starts on last day of month:
        end_time  = dt.datetime(int(year),int(month)+shift_day,shift_day,ehh,emm,ess) #ending period--can be day +1 if day crossover exists

    print('Selected times for analysis period begin at {0} and end at {1} hours'.format(base_date,end_time))
    case_duration = ((end_time-base_date).total_seconds()/60) / 60
    print('Duration of analysis period {0} hours'.format(case_duration))
#     print('Duration of analysis period {0} hours'.format(ehh-shh))

    reprocess = True #reprocess/flash sort data with more stringent criteria defined below
    #sort_params = {'duration_min':.15,'chi2max':5.0,'stationsmin':6,'min_events_per_flash':10}
    sort_params = {'chi2max':1.0,'stationsmin':5,'min_events_per_flash':10}
    if reprocess == True:
        print('Setting up flash sorting with the following params:')
        # print('Min duration: {0}'.format(sort_params['duration_min']))
        print('Chi2max: {0}'.format(sort_params['chi2max']))
        print('Min Stations: {0}'.format(sort_params['stationsmin']))
        print('Min events per flash: {0}'.format(sort_params['min_events_per_flash']))
        print('--------')

    lma = LMAGrabber(network,base_date,end_time,reprocess,sort_params)
    lma.flash_sort()
#     print('Flash sorting complete.')
#     print('Saving flash sorting log file.')
#     #save datetime bounds, file list, and converted file times to dictionary for later referencing
#     tbnds     = lma.t_bnds
#     files     = lma.valid_files
#     file_tsec = lma.lma_sec
#     dict = {'time_bounds':tbnds, 'file_list':files, 'file_times_sec':file_tsec}
#     case_log = f'caselogs/{lma.parse_date}-{network}.npy'
#     if os.path.exists(case_log):
#         print('Log file already exists, disabled saving--if time bounds are different, remove or rename previous file.')
#         pass
#     else:
#         np.save(case_log,dict)
