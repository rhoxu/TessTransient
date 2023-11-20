#!/usr/bin/env python3

import os
from glob import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from time import time as t

from astropy.io import fits
from astropy import wcs

from astropy.visualization import (SqrtStretch, ImageNormalize)

import lightkurve as lk
import tessreduce as tr

def _Extract_fits(pixelfile):
    """
    Quickly extract fits
    """
    try:
        hdu = fits.open(pixelfile)
        return hdu
    except OSError:
        print('OSError ',pixelfile)
        return

def _Strip_units(data):
    """
    strips units from data.
    """
    if type(data) != np.ndarray:
        data = data.value
    return data

def _Match_events(Events, Eventtime, Eventmask, Seperation = 5):
    """
    Old Code ish, yikes. Matches flagged pixels that have coincident event times of +-5 cadences and are closer than 4 pix
    seperation.
    """

    i = 0
    eventmask2 = []
    
    ts = t()
    
    while len(Events) > i:
        
        if (t()-ts) > 90:
            print('Too much time taken!')
            return None,None,None
        
        coincident = (np.isclose(Eventtime[i, 0], Eventtime[i:, 0], atol = Seperation) + np.isclose(
            Eventtime[i, 1], Eventtime[i:, 1], atol = Seperation))
        dist = np.sqrt((np.array(Eventmask)[i, 0]-np.array(Eventmask)[i:, 0])**2 + (
            np.array(Eventmask)[i, 1]-np.array(Eventmask)[i:, 1])**2)
        dist = dist < 5

        coincident = coincident * dist
        if sum(coincident*1) > 1:
            newmask = Eventmask[i].copy()

            for j in (np.where(coincident)[0][1:] + i):
                newmask[0] = np.append(newmask[0], Eventmask[j][0])
                newmask[1] = np.append(newmask[1], Eventmask[j][1])
            eventmask2.append(newmask)
            Events = np.delete(Events, np.where(coincident)[0][1:]+i)
            Eventtime = np.delete(Eventtime, np.where(
                coincident)[0][1:]+i, axis=(0))
            killer = sorted(
                (np.where(coincident)[0][1:]+i), key=int, reverse=True)
            for kill in killer:
                del Eventmask[kill]
        else:
            eventmask2.append(Eventmask[i])
        i += 1
    return Events, Eventtime, eventmask2

def _Touch_masks(events,eventtime,eventmask):
    """
    Checks if any part of masks border each other, combines events if so.
    """
    
    i = 0
    skip = []
    copy = np.copy(eventmask)
    while len(eventmask)-1 > i:
    
        if i not in skip:
            
            event = copy[i]
            
            # -- Get x,y values of event -- #
            x1 = np.array([event[0]])
            y1 = np.array([event[1]])
            
            if len(x1.shape) > 1:
                x1 = event[0]
                y1 = event[1]
                        
            # -- Compare with other events -- #
            for j in range(i+1,len(copy)):
                compare_event = copy[j]
                
                x2 = np.array([compare_event[0]])
                y2 = np.array([compare_event[1]])
                
                if len(x2.shape) > 1:
                    x2 = compare_event[0]
                    y2 = compare_event[1]

                # -- Calculate distances between each pixel in two events -- #
                dist_array = []
                for ii in range(x1.shape[0]):
                    for jj in range(x2.shape[0]):
                        
                        dist = np.sqrt((x1[ii]-x2[jj])**2+(y1[ii]-y2[jj])**2)
                        dist_array.append(dist)

                dist_array = np.array(dist_array)

                # -- If pixels are equal/touch, combine events -- #
                if (1.0 in dist_array) | (0.0 in dist_array):
  
                    newX = np.append(copy[i][0],copy[j][0])
                    newY = np.append(copy[i][1],copy[j][1])
                    copy[i][0] = newX
                    copy[i][1] = newY
                    skip.append(j)
        i += 1
            
    # -- Delete obsolete events -- #
    newmask = np.delete(copy,skip,axis=0)
    newevents = np.delete(events,skip,axis=0)
    neweventtime = np.delete(eventtime,skip,axis=0)

    return newevents,neweventtime,newmask

def _Min_Px(min_px,events,eventtime,eventmask):
    """
    Removes events with pixels fewer than min.
    """

    remove_ind = []
    array = np.copy(eventmask)
    for ind,event in enumerate(array):
        x = event[0]
        if len(x.shape) == 0: 
            x=[x]
        length = len(x)
        if length < min_px:
            remove_ind.append(ind)
    events = np.delete(events,remove_ind,axis=0)
    eventtime = np.delete(eventtime,remove_ind,axis=0)
    eventmask = np.delete(eventmask,remove_ind,axis=0)
    
    remove_ind = []
    for ind,event in enumerate(eventmask):
        if (event[0][0] == event[0][1]) & (event[1][0] == event[1][1]):
            remove_ind.append(ind)
    events = np.delete(events,remove_ind,axis=0)
    eventtime = np.delete(eventtime,remove_ind,axis=0)
    eventmask = np.delete(eventmask,remove_ind,axis=0)

    return events, eventtime, eventmask

def _Asteroid_Filter(input_flux,events,eventtime,eventmask):
    """
    Filters out asteroids by requiring coincident peak times between pixels. Relatively dodgy.
    """

    array = np.copy(eventmask)
    remove_ind = []
    for p,mask in enumerate(array):
        max_inds = []
        for i in range(len(mask[0])):
            x = mask[0][i]
            y = mask[1][i]
            flux = input_flux[eventtime[p][0]-10:eventtime[p][-1]+20,x,y]
            max_ind = np.where(flux == max(flux))[0][0]
            max_inds.append(max_ind)
        if len(np.unique(max_inds)) > 1:
            remove_ind.append(p)
    
    events = np.delete(events,remove_ind,axis=0)
    eventtime = np.delete(eventtime,remove_ind,axis=0)
    eventmask = np.delete(eventmask,remove_ind,axis=0)

    return events, eventtime, eventmask

def _Rank_brightness(input_flux,events,eventtime,eventmask):
    """
    Ranks events by peak brightness.
    """        

    # -- Find maximums -- #
    maximums = []
    for i in range(len(events)):
        flux = input_flux[events[i]:eventtime[i,1],eventmask[i,0],eventmask[i,1]]
        maximum = np.max(flux),i
        maximums.append(maximum)

    # -- Get event index order based on maximums -- #
    maximums.sort(reverse=True)
    maximums = np.array(maximums)
    order = maximums[:,1]
    order = order.astype(int)
    return order

def _Lightcurve(Data, Mask, Normalise = False):
    """
    Old Code. Takes a whole data cube, and a binary mask of pixels to include
    in lightcurve.
    """    
    
    if type(Mask) == list:
        mask = np.zeros((Data.shape[1],Data.shape[2]))
        mask[Mask[0],Mask[1]] = 1
        Mask = mask*1.0
    Mask[Mask == 0.0] = np.nan
    LC = np.nansum(Data*Mask, axis = (1,2))
    LC[LC == 0] = np.nan
    for k in range(len(LC)):
        if np.isnan(Data[k]*Mask).all(): # np.isnan(np.sum(Data[k]*Mask)) & (np.nansum(Data[k]*Mask) == 0):
            LC[k] = np.nan
    if Normalise:
        LC = LC / np.nanmedian(LC)
    return LC

def _Save_space(Save,delete=False):
    """
    Creates a path if it doesn't already exist.
    """
    try:
        os.makedirs(Save)
    except FileExistsError:
        if delete:
            os.system(f'rm -r {Save}/')
            os.makedirs(Save)
        else:
            pass

class TessDetector():
    """
    Class for detecting events in TESS data.
    """

    def __init__(self,CutsObject,eventname,eventtime,sector,cam,chip,cutNum):

        self.eventname = eventname 
        self.cam = cam
        self.chip = chip
        self.cutNum = cutNum
        self.eventtime = eventtime
        self.sector = sector

        self.ra = CutsObject.ra
        self.dec = CutsObject.dec    
        self.path = f'{CutsObject.path}/..'
        self.error = CutsObject.error

        reducedName = f'{eventname}_cam{cam}_chip{chip}_error{self.error}_cut{cutNum}_Reduced.fits'
        self.reduced_file = f'{self.path}/Cam{cam}Chip{chip}/{reducedName}'

        self.cut = _Extract_fits(self.reduced_file)
        self.wcs = wcs.WCS(self.cut[2].header)
        self.cutSize = CutsObject.cutSizes[cutNum-1]
        self.cutCorner = CutsObject.cutCorners[cutNum-1]

        self.times = None
        self.flux = None 

    def _tpf_info(self):
        """
        Loads in flux, time values from reduced file.
        """

        if not os.path.exists(self.reduced_file):
            print('No reduced file.')
            
        else:
            tpf = lk.TessTargetPixelFile(self.reduced_file)
            print('Getting Flux')
            self.flux = _Strip_units(tpf.flux)
            print('Getting Time')
            self.times = tpf.time.value

    def _Get_err_px(self):
        """
        Get the px coords of error ellipse.
        """
        
        # -- Creates a 'circle' in realspace for RA,Dec -- #
        raEll = []
        decEll = []
        for ii in np.linspace(0,2*np.pi,10000):
            raEll.append(self.ra + self.error*np.cos(ii))
            decEll.append(self.dec + self.error*np.sin(ii))
        
        # -- Converts circle to pixel space -- #
        
        errpixels = self.wcs.all_world2pix(raEll,decEll,0)    
        errpixels = np.array(errpixels)

        where = np.where((errpixels[0] >= 0) & (errpixels[0] <= self.cutSize[0]) & (errpixels[1] >= 0) & (errpixels[1] <= self.cutSize[1]))
        ellipse_reduced = errpixels[:,where]
        ellipse_reduced = ellipse_reduced[:,0,:]
    
        return errpixels

    def detect(self,significance,min_length,min_px,max_events,plot,asteroids):
        """
        Detection function. 

        ------
        Inputs
        ------
        significance : int,float
            threshold for detection is median + {significance} * standard deviation
        min_length : int
            minimum number of consecutive frames a detected signal must remain above threshold
        min_px : int
            minimum number of pixels required to trigger for an event to be included
        max_events : int
            if more events than this, the printout is prevented
        
        -------
        Options
        -------
        plot : bool
            if True, plot the detection information
        asteroids : bool
            if False, remove 'asteroid' events (those which peak at different times, thus object transiting accross frame)
        
        -------
        Creates
        -------
        self.events : array

        self.eventtimes : array

        self.eventmask : array


        """
        
        # -- Load in data -- #
        if self.times is None: 
            print('--COLLECTING DATA--')
            self._tpf_info()
            print('--DONE--')
            print('\n')

        # -- Check if event occured just before time window -- #
        if self.eventtime - self.times[0] < 0:
            print('GRB Time just outside captured window.')
            print('\n')
            return
        
        print('Fetching Events')

        # -- Check if event occured during TESS break (dodgy code, doesn't always work) -- #
        #splitStart = np.where(np.diff(self.times) == max(np.diff(self.times)))[0][0]
        #if (self.eventtime >= self.times[splitStart]) & (self.eventtime < self.times[splitStart+1]):
        #    print('GRB occured during TESS split')

        event_index = np.argmin(abs(self.times-self.eventtime)) #closest index to eventtime
        
        # -- Create time/flux arrays in focused region around event -- #
        focus_times = self.times[event_index-10:event_index+20]
        focus_fluxes = self.flux[event_index-10:event_index+20,:,:]
        if self.times[event_index] > self.eventtime:
            detect = 11
        else:
            detect = 10

        # -- Calculate median/std of local region (30 indices before event) -- #
        med = np.nanmedian(self.flux[event_index-30:event_index], axis = 0)
        std = np.nanstd(self.flux[event_index-30:event_index],axis=0)

        # -- Find pixels that meet conditions -- #
        binary = (focus_fluxes >= (med+significance*std))
        summed_binary = np.nansum(binary[detect:detect+5],axis=0)
        X = np.where(summed_binary >= min_length)[0] # note that X = Y
        Y = np.where(summed_binary >= min_length)[1]
       
        # -- Old Code, essentially creates events,eventtimes,eventmask based on above conditions -- # 
        tarr = self.flux > 10000000
        tarr[event_index-10:event_index+20,:,:] = binary
        events = []
        eventmask = []
        eventtimes = []

        for i in range(len(X)):
            
            temp = np.insert(tarr[:,X[i],Y[i]],0,False) # add a false value to the start of the array
            testf = np.diff(np.where(~temp)[0])
            indf = np.where(~temp)[0]
            testf[testf == 1] = 0
            testf = np.append(testf,0)
        
            if len(indf[testf>min_length]) > 0:
                for j in range(len(indf[testf>min_length])):
                    start = indf[testf>min_length][j]
                    end = (indf[testf>min_length][j] + testf[testf>min_length][j]-1)
                    #if np.nansum(Eventmask[start:end,X[i],Y[i]]) / abs(end-start) > 0.5:
                    events.append(start)
                    eventtimes.append([start, end])
                    masky = [np.array(X[i]), np.array(Y[i])]
                    eventmask.append(masky)    
                   
        events = np.array(events) # initial indices in timelist of events
        eventtimes = np.array(eventtimes) # start and end indices of events

        events, eventtimes, eventmask = _Match_events(events, eventtimes, eventmask)  # collates coincident events into single events

        eventmask = np.asarray(eventmask, dtype="object")

        # -- Cuts events to only those which begin an hour or less after eventtime -- #  
        timeInterval = min(np.diff(self.times))
        allowedStart = np.floor(1/ (24 * timeInterval))
        event_checks = (events - event_index) < allowedStart
        events = events[np.where(event_checks)[0]]
        eventtimes = eventtimes[np.where(event_checks)[0]]
        eventmask = eventmask[np.where(event_checks)[0]]

        # -- Checks if any events touch / are too large for match_events -- #
        events, eventtimes, eventmask = _Touch_masks(events, eventtimes,eventmask)

        if min_px > 1:
            events,eventtimes,eventmask = _Min_Px(min_px,events,eventtimes,eventmask)   # restrict events to those with at least min_px
        
        if max_events is not None:
            if len(eventmask) > max_events:
                print('Too many events')
                return 
            
        if not asteroids:
            events,eventtimes,eventmask = _Asteroid_Filter(self.flux,events,eventtimes,eventmask)   # filter out asteroids

        if len(events) == 0:
            print('No events!')
            return
    
        # -- rank brightness so that brighter events are returned first -- #
        order = _Rank_brightness(self.flux,events,eventtimes,eventmask)     
        self.events = events[order]
        self.eventtimes = eventtimes[order]
        self.eventmask = eventmask[order]

        # -- plot events! -- #
        if plot:
            rat = self.cutSize[1]/self.cutSize[0]
            fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=False, figsize=(6,6*rat))
            ax.set_xlim(0,self.cutSize[0])
            ax.set_ylim(0,self.cutSize[1])
            
            ra_dec_px = self.wcs.all_world2pix(self.ra,self.dec,0)
            ellipse = self._Get_err_px()
            ax.plot(ellipse[0],ellipse[1])
            ax.scatter(ra_dec_px[0],ra_dec_px[1])
            
            ax.grid()
            
            counter = 1

            for event in eventmask:
                ax.scatter(event[1],event[0],marker='s',color='r',s=2,edgecolor='black')
                point = (np.median(event[1]),np.median(event[0])+5)
                ax.annotate('Event ' + str(counter),point,fontsize=10)
                
                fig, ax2 = plt.subplot_mosaic([[0,0,1]], constrained_layout=False, figsize=(9,4))
                ax2[0].set_title(f'Event {counter} (Time {self.eventtime:.3f})')
                    
                ax2[0].plot(focus_times,focus_fluxes[:,event[0],event[1]])
                ax2[0].axvline(self.eventtime,linestyle='--',alpha=1,color='black')
                ax2[0].ticklabel_format(useOffset=False)
                ax2[0].set_xlim(focus_times[0],focus_times[-1])
                ax2[0].set_xlabel('Time (MJD)')
                ax2[0].set_ylabel('TESS Counts (e$^-$/s)')
                ax2[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))

                ax2[1].set_aspect('equal')
                
                ax2[1].set_title('Event ' + str(counter) + ' Pixels')
                ax2[1].set_xlim(np.min(event[1])-15,np.max(event[1]+15))
                ax2[1].set_ylim(np.min(event[0])-10,np.max(event[0]+10))
                ax2[1].scatter(event[1]+0.5,event[0]+0.5,marker='s',color='r',s=50,edgecolor='black')
                ax2[1].grid()
                
                counter += 1

    def make_video(self,eventnumber):
        """
        Creates an mp4 of the event.

        ------
        Inputs
        ------
        eventnumber : int
            as presented by the detect function.
        
        -------
        Creates
        -------
        File in Videos folder within path.

        """

        # -- retrieve event -- #
        eventtime = self.eventtimes[eventnumber-1]
        eventmask = self.eventmask[eventnumber-1]
        
        # -- Creates a mask of pixels in the event -- #
        mask = np.zeros((self.flux.shape[1],self.flux.shape[2]))
        mask[eventmask[0],eventmask[1]] = 1
        
        # -- Finds brightest pixel -- #
        position = np.where(mask)
        Mid = ([position[0][0]],[position[1][0]])
        maxcolor = 0 # Set a bad value for error identification
        for j in range(len(position[0])):
            lcpos = np.copy(self.flux[eventtime[0]:eventtime[1],position[0][j],position[1][j]])
            nonanind = np.isfinite(lcpos)
            temp = sorted(lcpos[nonanind].flatten())
            temp = np.array(temp)
            if len(temp) > 0:
                temp  = temp[-1] # get brightest point
                if temp > maxcolor:
                    maxcolor = temp
                    Mid = ([position[0][j]],[position[1][j]])
                    
        Mid = (np.round(np.mean(position[0])),np.round(np.mean(position[1])))
                    
        # -- get lightcurve of flagged pixels -- #
        LC = _Lightcurve(self.flux, mask)
        lclim = np.copy(LC[eventtime[0]:eventtime[1]])
        
        # -- find limits of LC -- #
        temp = sorted(lclim[np.isfinite(lclim)].flatten())
        temp = np.array(temp)
        maxy  = temp[-1] # get 8th brightest point
        
        temp = sorted(LC[np.isfinite(LC)].flatten())
        temp = np.array(temp)
        miny  = temp[10] # get 10th faintest point
        
        ymin = miny - 0.1*miny
        ymax = maxy + 0.1*maxy
        
        ind = np.argmin(abs(self.times-self.eventtime))

        if ind % 2 == 1:
            if eventtime[1] % 2 == 1:
                endspan = eventtime[1]
            else:
                endspan = eventtime[1] - 1
        else:
            if eventtime[1] % 2 == 0:
                endspan = eventtime[1]
            else:
                endspan = eventtime[1] - 1
        
        Section = np.arange(ind-8,endspan+8,2)
        
        # -- Create an ImageNormalize object using a SqrtStretch object -- #
        norm = ImageNormalize(vmin=ymin/len(position[0]), vmax=maxcolor, stretch=SqrtStretch())
        
        height = 1100/2
        width = 2200/2
        my_dpi = 100
        
        # -- Create temporary location for video frames -- #
        FrameSave = f'{self.path}/VideoFrames'
        _Save_space(FrameSave,delete=True)

        # -- Make frames -- #
        print('Making Frames')
        for j in range(len(Section)):
            
            filename = FrameSave + '/Frame_' + str(int(j)).zfill(4)+".png"

            fig = plt.figure(figsize=(width/my_dpi,height/my_dpi),dpi=my_dpi)
            plt.subplot(1, 2, 1)
            plt.title('Event light curve')
            plt.axvspan(self.times[ind]-self.times[0],self.times[endspan]-self.times[0],color='orange',alpha = 0.5)
            plt.plot(self.times - self.times[0], LC,'k.')
        
            plt.ylim(ymin,ymax)
            plt.xlim(self.times[ind-8]-self.times[0],self.times[endspan+8]-self.times[0])
        
            plt.ylabel('Counts')
            plt.xlabel('Time (days)')
            plt.axvline(self.times[Section[j]]-self.times[0],color='red',lw=2)
        
            plt.subplot(1,2,2)
            plt.title('TESS image')
            self.flux[np.isnan(self.flux)] = 0
            plt.imshow(self.flux[Section[j]],origin='lower',cmap='gray',norm=norm)
            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='black')
        
            xlims = (Mid[1]-6.5,Mid[1]+6.5)
            ylims = (Mid[0]-6.5,Mid[0]+6.5)
            
            plt.xlim(xlims[0],xlims[1])
            plt.ylim(ylims[0],ylims[1])
            plt.ylabel('Row')
            plt.xlabel('Column')
            fig.tight_layout()
        
            ax = fig.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
            plt.savefig(filename,dpi=100)
            plt.close()
        
        # -- Make video -- #
        VidSave = f'{self.path}/Videos'
        _Save_space(VidSave)

        framerate = 2
        ffmpegcall = f'ffmpeg -y -nostats -loglevel 8 -f image2 -framerate {framerate} -i {FrameSave}/Frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {VidSave}/{self.eventname}-Event{eventnumber}.mp4'
        os.system(ffmpegcall)
        print('Video Made')

        # -- Delete frames -- #
        os.system(f'rm -r {FrameSave}')

    def _find_coords(self,eventnumber):
        """
        Finds the coords of the event, both with respect to the cut WCS and the OG fits file WCS.

        ------
        Inputs
        ------
        eventnumber : int
            as presented by the detect function.
        
        -------
        Returns
        -------
        cutWCS_coords, baseWCS_coords

        """

        eventmask = self.eventmask[eventnumber-1]   # initialise event

        event_xpix = np.median(eventmask[1])  # median x pixel
        event_ypix = np.median(eventmask[0])  # median y pixel

        cutWCS_coords = self.wcs.all_pix2world(event_xpix,event_ypix,0)

        filepath = glob(f'{self.path}/Cam{self.cam}Chip{self.chip}/*ffic.fits')[0]
        file = _Extract_fits(filepath)
        base_wcs = wcs.WCS(file[1].header)
        file.close()

        base_px = (np.floor(self.cutCorner[0])+event_xpix,np.floor(self.cutCorner[1])+event_ypix)

        baseWCS_coords = base_wcs.all_pix2world(base_px[0],base_px[1],0)

        return cutWCS_coords,baseWCS_coords

    def event_coords(self,eventnumber):
        """
        Print out coords for event.
        """

        cutWCS_coords, baseWCS_coords = self._find_coords(eventnumber)
        print(f'Cut WCS Coords: RA = {cutWCS_coords[0]}, DEC = {cutWCS_coords[1]}')
        print(f'Base WCS Coords: RA = {baseWCS_coords[0]}, DEC = {baseWCS_coords[1]}')

    def tessreduce_event(self,eventnumber,coords='cut'):
        """
        Generates tessreduce object for event.

        ------
        Inputs
        ------
        eventnumber : int
            as presented by the detect function.
        coords : str
            if 'cut' use cut WCS, if 'base' use OG FITS file WCS
        
        -------
        Returns
        -------
        tessreduce object

        """

        cutWCS_coords, baseWCS_coords = self._find_coords(eventnumber)
        if coords == 'cut':
            ra = cutWCS_coords[0]
            dec = cutWCS_coords[1]
        elif coords == 'base':
            ra = baseWCS_coords[0]
            dec = baseWCS_coords[1]
        
        trObj = tr.tessreduce(ra,dec,sector=self.sector,plot=False,reduce=True)
        return trObj












