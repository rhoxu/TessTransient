import os
from glob import glob

import tessreduce as tr

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import scipy

from time import time as t

from astrocut import CubeFactory
from astrocut import CutoutFactory
from astropy.io import fits
from astropy.time import Time
from astropy.io.fits import getdata
from astropy.table import Table

from .find_cut import FindCut
from .downloader import *
from .detector import TessDetector

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

def _Remove_emptys(files):
    """
    Deletes corrupt fits files before creating cube
    """

    deleted = 0
    for file in files:
        size = os.stat(file)[6]
        if size < 35500000:
            os.system('rm ' + file)
            deleted += 1
    return deleted

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
    
def _Print_buff(length,string):

    strLength = len(string)
    buff = '-' * int((length-strLength)/2)
    print(f"{buff}{string}{buff}")

class TessTransient():
    
    def __init__(self,ra,dec,eventtime,error=0,verbose=1,path=None,delete=True,eventname=None):
        """
        Transient Localisation for TESS!

        ------
        Inputs
        ------
        ra : float
            degree right ascension of event estimation
        dec : float
            degree declination of event estimation
        eventtime : float 
            MJD reference time for event
        error : float
            degree 1 dimensional error for event

        -------
        Options:
        -------
        verbose : int  
            0 = no printout , 1 (default) = partial printout, 2 = full printout
        path : str
            location of folder, creates temporary folder if None 
        eventname : str
            name of event to be encoded into downloads
        delete : bool
            deletes previous temporary folder if True
        
        """

        # Given
        self.ra = ra
        self.dec = dec
        self.error = error
        self.eventtime = eventtime
        self.verbose = verbose
        self.eventname = eventname
        self.path = path

        self._location_observed()
        
        if self.obs:
            self._make_path(delete)

        self.Cuts = None
        self.neighbours = None
        self.Detector = None

    def _make_path(self,delete):
        """
        Creates a folder for the path. 
        """

        if self.path is None:
            _Save_space('temporary',delete=delete)
            self.path = './temporary'
        else:
            if self.path[-1] == '/':
                self.path = self.path[:-1]
            if self.eventname is None:
                _Save_space(f'{self.path}/temporary',delete=delete)
                self.path = f'{self.path}/temporary'
            else:
                _Save_space(f'{self.path}/{self.eventname}',delete=delete)
                self.path = f'{self.path}/{self.eventname}'
        
    def _location_observed(self):
        """
        Check whether any part of error region is on a TESS CCD. 
        """

        obj = tr.spacetime_lookup(self.ra,self.dec,self.eventtime,print_table=self.verbose>1)   # look up location using TESSreduce - requires custom 

        obs = False
        for sector in obj:
            if sector[-1] == True:
                if self.verbose>0:
                    print('Event occured within TESS FoV!')
                self.sector = sector[2]
                self.cam = sector[3]    # only returned in custom TR package
                self.chip = sector[4]   # only returned in custom TR package
                obs=True
       
        if not obs:
            obs = self._check_surroundings()    # check if part of error region observed but not estimate itself

        self.obs = obs
       
        if not obs:
            if self.verbose > 0:
                print('Event did not occur within TESS FoV :(')
            self.sector = 'Event did not occur within TESS FoV :('
            self.cam = 'Event did not occur within TESS FoV :('
            self.chip = 'Event did not occur within TESS FoV :('
            return
            
    def _check_surroundings(self):
        """
        Check if error region overlaps with TESS CCD.
        """

        # -- Generate a bunch of coords in all directions around RA,DEC -- #
        distances = np.arange(2,self.error,2)
        distances = np.append(distances,self.error)
        raEll = []
        decEll = []
        for jj in np.linspace(0,2*np.pi-np.pi/4,8):
            for d in distances:
                raEll.append(self.ra + d*np.cos(jj))
                decEll.append(self.dec + d*np.sin(jj))
        coords = np.array(tuple(zip(raEll,decEll)))

        # -- Check if any of the coords overlap with TESS -- #
        goodcoords = []
        for coord in coords:
            try: 
                obj = tr.spacetime_lookup(coord[0],coord[1],self.eventtime,print_table=False)
                for sector in obj:
                    if sector[-1] == True:
                        goodcoords.append(coord)
            except:
                pass
        goodcoords = np.array(goodcoords)

        # -- If any do, find whether the RA,DEC is outside border of CCDs, or in a gap between CCDs -- #
        obs = False
        if len(goodcoords) != 0:
            goodestcoords = np.array(goodcoords)

            grb_point = np.array((165,26))
            grb_point = np.reshape(grb_point,(1,2))
            dist = scipy.spatial.distance.cdist(goodestcoords,grb_point)
            min_dist = min(dist)[0]
            
            meanRA = np.mean(goodestcoords[:,0])
            meanDEC = np.mean(goodestcoords[:,1])
            
            mean_point = np.array((meanRA,meanDEC))
            mean_point = np.reshape(mean_point,(1,2))
            
            meandist = math.dist(grb_point[0],mean_point[0])

            between = False
            if meandist < min_dist:    # if the mean distance is less than the minimum, event is between chips 
                between = True

                obj = tr.spacetime_lookup(goodcoords[int(len(goodcoords)/2)][0],goodcoords[int(len(goodcoords)/2)][1],self.eventtime,print_table=False)
                for sector in obj:
                    if sector[-1] == True:
                        self.sector = sector[2]
                        self.cam = sector[3]
                        self.chip = sector[4]
                        obs=True

            else:       # get an estimate on how much of error region is on CCDs

                percentage = ((min_dist)/(2*3.2))*100
                                # if min_dist == 2:
                                #     min_dist = '<2'

                                #     print('Nearest Point = <2 degrees ({:.1f}% of {}σ error)'.format(percentage,2))
                                # else:
                                #     print('Nearest Point = {:.1f} degrees ({:.1f}% of {}σ error)'.format(min_dist,percentage,2))

                obj = tr.spacetime_lookup(meanRA,meanDEC,self.eventtime,print_table=False)
                for sector in obj:
                    if sector[-1] == True:
                        self.sector = sector[2]
                        self.cam = sector[3]
                        self.chip = sector[4]
                        obs=True       

            if self.verbose > 0:
                if between:
                    print('Event located between chips')
                else:
                    print('Part of error region observed!') 

        return obs

    def download(self,number=None,cam=None,chip=None):
        """
        Function for downloading FFIs from MAST archive.

        ------
        Inputs
        ------
        cam : int
            desired camera, default home cam
        chip : int
            desired chip, default home chip

        -------
        Options:
        -------
        number : int
            if not None, downloads this many
        
        """

        if cam is None:
            cam = self.cam
        if chip is None:
            chip = self.chip

        if number is None:
            big = True
        elif number > 5:
            big = True
        else:
            big = False

        # -- Downloads to path, requires proper path if more than 5 are to be downloaded -- #
        if big:
            if self.eventname is None:
                print('Downloading to temporary folder is a bad idea!')
                return
            Download_cam_chip_FFIs(self.path,self.sector,
                                   cam,chip,
                                   self.eventtime,lower=2,upper=1,  # lower,upper = number of days either side of eventtime to download
                                   number=number)          

        else:
            Download_cam_chip_FFIs(self.path,self.sector,
                                   cam,chip,
                                   self.eventtime,lower=2,upper=1,  
                                   number=number)      

    def find_cuts(self,cam=None,chip=None,plot=True,proj=True):
        """
        Find cuts encompassing error region.
        
        ------
        Inputs
        ------
        cam : int
            desired camera (default = home cam)
        chip : int
            desired chip (default = home chip)

        -------
        Options
        -------
        plot : bool
            if True, display mpl plot of region and cuts
        proj : bool
            if True, display plot will have RA / Dec coords
        
        -------
        Creates
        -------
        self.Cuts : FindCut Objects
            contains information of the calculated cuts including
            - cutCorners : (x,y) pixel of the bottom left corner of each cut
            - cutCentrePx : (x,y) pixel of the centre of each cut
            - cutSizes : (x,y) diameter of each cut
            - cutCentreCoords : (ra,dec) coords of the centre of each

        """

        if cam is None:
            cam = self.cam
        if chip is None:
            chip = self.chip

        if self.eventname is None:
            path = self.path
        else:
            path = f'{self.path}/Cam{cam}Chip{chip}'

        self.Cuts = FindCut(self.ra,self.dec,self.error,path)
        if self.Cuts.wcs is None:
            return
        
        self.Cuts._find_cuts()
        
        if plot:
            # -- Plots data -- #
            fig = plt.figure(constrained_layout=False, figsize=(6,6))
            
            if proj:
                ax = plt.subplot(projection=self.Cuts.wcs)
                ax.set_xlabel(' ')
                ax.set_ylabel(' ')
            else:
                ax = plt.subplot()
            
            # -- Real rectangle edge -- #
            rectangleTotal = patches.Rectangle((44,0), 2048, 2048,edgecolor='black',facecolor='none',alpha=0.5)
            
            # -- Sets title -- #
            ax.set_title('Camera {} Chip {}'.format(cam,chip))

            if (cam == self.cam) & (chip == self.chip):
                ax.scatter(self.Cuts.grb_px[0],self.Cuts.grb_px[1],color='g',label='Field Centre',marker='x',s=100)

            ax.set_xlim(0,2136)
            ax.set_ylim(0,2078)

            ax.grid()
            ax.add_patch(rectangleTotal)
            
            ax.plot(self.Cuts.ellipse[0],self.Cuts.ellipse[1],color='g',marker='.')#,label='2$\sigma$ Position Error Radius')
            
            # -- Adds cuts -- #
            colors = ['red','blue','purple','orangered']

            for i, corner in enumerate(self.Cuts.cutCorners):
                x = self.Cuts.cutSizes[i,0]
                y = self.Cuts.cutSizes[i,1]
                    
                rectangle = patches.Rectangle(corner,x,y,edgecolor=colors[i],
                                              facecolor='none',alpha=1)
                ax.add_patch(rectangle)
                
            # fig.show()
            # plt.close(fig)

    def make_cube(self,cam=None,chip=None,delete_files=False,ask=True,cubing_message='Cubing'):
        """
        Make cube for this cam,chip.
        
        ------
        Inputs
        ------
        cam : int
            desired camera (default = home cam)
        chip : int
            desired chip (default = home chip)

        -------
        Options
        -------
        delete_files : bool  
            deletes all FITS files once cube is made
        ask : bool
            require manual input before creating cube
        cubing_message : str
            custom printout message for self.verbose > 0

        -------
        Creates
        -------
        Data cube fits file in path.

        """

        # -- Requires dedicated path -- #
        if self.eventname is None:
            print('Downloading to temporary folder is a bad idea!')
            return
        
        if cam is None:
            cam = self.cam
        if chip is None:
            chip = self.chip

        # -- Generate Cube Path -- #
        file_path = f'{self.path}/Cam{cam}Chip{chip}'
        cube_name = f'{self.eventname}_cam{cam}_chip{chip}_cube.fits'
        cube_path = f'{file_path}/{cube_name}'

        if os.path.exists(cube_path):
            print(f'Cam {cam} Chip {chip} cube already exists!')
            return

        input_files = glob(f'{file_path}/*ffic.fits')  # list of fits files in path
        if len(input_files) < 1:
            print('No files to cube!')
            return  
        
        deleted = _Remove_emptys(input_files)  # remove empty downloaded fits files
        if self.verbose > 1:
            print(f'Deleted {deleted} corrupted file/s.')
                    
        input_files = glob(f'{file_path}/*ffic.fits')  # regather list of good fits files
        if len(input_files) < 1:
            print('No files to cube!')
            return    

        if self.verbose > 1:
            print(f'Number of files to cube = {len(input_files)}')
            size = len(input_files) * 0.0355
            print(f'Estimated cube size = {size:.2f} GB')

        # -- Require input before creating the cube -- #
        if ask: 
            done = False
            while not done:
                go = input('Proceed? [y/n]')
                if go == 'y':
                    done = True
                elif go == 'n':
                    done = True
                    print('Aborted')
                    return
                else:
                    print('Answer format invalid! (y/n)')    

        # -- Allows for a custom cubing message (kinda dumb) -- #
        if self.verbose > 0:
            print(cubing_message)
        
        # -- Make Cube -- #
        cube_maker = CubeFactory()
        cube_file = cube_maker.make_cube(input_files,cube_file=cube_path,verbose=self.verbose>1,max_memory=200)

        # -- If true, delete files after cube is made -- #
        if delete_files:
            homedir = os.getcwd()
            os.chdir(file_path)
            os.system('rm *ffic.fits')
            os.chdir(homedir)

    def make_cuts(self,cam=None,chip=None):
        """
        Make cut(s) for this chip.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        chip : int
            desired chip

        ------
        Creates
        ------
        Save files for cut(s) in path.

        """

        # -- Requires dedicated event path -- #
        if self.eventname is None:
            print('Downloading to temporary folder is a bad idea!')
            return
        
        if cam is None:
            cam = self.cam
        if chip is None:
            chip = self.chip
        
        # -- Calling find_cuts() -- #
        if self.Cuts is None:
            try:
                self.find_cuts(cam=cam,chip=chip,plot=False)
            except:
                print('Something wrong with finding cut!')
                return
        
        file_path = f'{self.path}/Cam{cam}Chip{chip}'
        if not os.path.exists(file_path):
            print('No data to cut!')
            return
        
        num_cuts = len(self.Cuts.cutCentreCoords)  # number of cuts to make
        
        cube_name = f'{self.eventname}_cam{cam}_chip{chip}_cube.fits'
        cube_file = f'{file_path}/{cube_name}' # cube path

        # -- Iterate through cuts, if not already made, make cut -- #
        for i, coords in enumerate(self.Cuts.cutCentreCoords):
            name = f'{self.eventname}_cam{cam}_chip{chip}_error{self.error}_cut{i+1}.fits'
            if os.path.exists(f'{file_path}/{name}'):
                print(f'Cam {cam} Chip {chip} cut {i+1} already made!')
            else:
                if self.verbose > 0:
                    print(f'Cutting Cam {cam} Chip {chip} cut #{i+1} (of {num_cuts})')
                
                my_cutter = CutoutFactory() # astrocut class
                coords = self.Cuts.cutCentreCoords[i]
                cutsize = self.Cuts.cutSizes[i]
                                
                # -- Cut -- #
                self.cut_file = my_cutter.cube_cut(cube_file, 
                                                    f"{coords[0]} {coords[1]}", 
                                                    (cutsize[0],cutsize[1]), 
                                                    output_path = file_path,
                                                    target_pixel_file = name,
                                                    verbose=(self.verbose>1)) 

                if self.verbose > 0:
                    print('Cam {} Chip {} cut {} complete.'.format(cam,chip,i+1))
                    print('\n')


    def reduce(self,cam=None,chip=None):
        """
        Reduces all cuts on a chip using TESSreduce. bkg correlation 
        correction and final calibration are disabled due to time constraints.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        chip : int
            desired chip

        -------
        Creates
        -------
        Fits file in path with reduced data.

        """

        if cam is None:
            cam = self.cam
        if chip is None:
            chip = self.chip

        # -- Calling find_cuts() -- #
        if self.Cuts is None:
            try:
                self.find_cuts(cam=cam,chip=chip,plot=False)
            except:
                print('Something wrong with finding cut!')
                return
        
        filepath = f'{self.path}/Cam{cam}Chip{chip}'

        numCuts = len(self.Cuts.cutCentreCoords)   # number of cuts to reduce

        # -- Iterate over cuts to generate new FITS files with reduced photometric data -- #
        for i,_ in enumerate(self.Cuts.cutCentreCoords):
            cutName = f'{self.eventname}_cam{cam}_chip{chip}_error{self.error}_cut{i+1}.fits'
            cutPath = f'{filepath}/{cutName}'
            reducedName = f'{self.eventname}_cam{cam}_chip{chip}_error{self.error}_cut{i+1}_Reduced.fits'
            reducedPath = f'{filepath}/{reducedName}'
            
            if os.path.exists(reducedPath):
                if self.verbose > 0:
                    print(f'Cam {cam} Chip {chip} cut {i+1} already reduced!')
                return
            
            ts = t()
            
            if self.verbose > 0:
                print('\n')
                print(f'--Reduction Cam {cam} Chip {chip} Cut {i+1} (of {numCuts})--')
            
            try: 
                # -- Defining so can be deleted if failed -- #
                tessreduce = 0
                tCut = 0
                data = 0
                table = 0
                tableTime = 0
                timeMJD = 0
                timeBJD = 0
                hdulist = 0

                # -- reduce -- #
                tessreduce = tr.tessreduce(tpf=cutPath,reduce=True,
                                                        corr_correction=False,
                                                        calibrate = False)
                
                if self.verbose > 0:
                    print(f'--Reduction Complete (Time: {((t()-ts)/60):.2f} mins)--')
                    print('--Writing--')
                tw = t()   # write timeStart
                
                # -- Inputs information into fits HDU -- #

                cut = _Extract_fits(cutPath)  # open fits file
                                            
                hdulist = fits.HDUList(cut)
                hdulist[0].header['REDUCED'] = (True,'confirms if data stored in file is reduced by TESSreduce') # add header to confirm reduced 
                if self.verbose > 0:
                    print('getting data')
                data = getdata(cutPath, 1)  # open table of unreduced data
                table = Table(data)

                del(data)

                # -- Replace table data with reduced data -- #
                tableTime = table['TIME']
                timeMJD = Time(tessreduce.lc[0],format='mjd')
                timeBJD = timeMJD.btjd
                
                indices = []
                for time in timeBJD:
                    index = np.argmin(abs(np.array(tableTime) - time))
                    indices.append(index)

                if self.verbose > 0:
                    print('inputting data')
                tCut = table[indices]
                
                del(table)
                del(timeMJD)
                del(timeBJD)
                del(tableTime)
                
                tCut['TIME'] = tessreduce.lc[0]
                tCut['FLUX'] = tessreduce.flux
                tCut['FLUX_BKG'] = tessreduce.bkg
                
                # -- Deletes Memory -- #
                del(tessreduce)

                hdulist[1] = fits.BinTableHDU(data=tCut,header=hdulist[1].header)            
        
                del(tCut)
            
                # -- Writes data -- #
                if self.verbose > 0:
                    print('writing data')
                
                hdulist.writeto(reducedPath,overwrite=True) 
                
                hdulist.close()
                
                if self.verbose > 0:
                    print(f'--Writing Complete (Time: {((t()-ts)/60):.2f} mins)--')
                    print('\n')

            except:
                # -- Deletes Memory -- #
                del(tessreduce)
                del(tCut)
                del(data)
                del(table)
                del(tableTime)
                del(timeMJD)
                del(timeBJD)

                try:
                    del(hdulist)
                except:
                    pass

                if self.verbose > 0:
                    print(f'Reducing Cam {cam} Chip {chip} Cut {i+1} Failed :( Time Elapsed: {((t()-ts)/60):.2f} mins.')
                    print('\n')
                pass 
    

    def find_neighbour_chips(self,verbose=True):
        """
        Uses the camera/chip of the GRB and error ellipse pixels to 
        find the neighbouring camera/chip combinations that contain 
        some part of the ellipse. Pretty poorly coded, but I can't be 
        bothered cleaning it and debugging again haha.

        -------
        Options
        -------
        verbose : bool
            override printout

        -------
        Creates
        -------
        self.neighbours - List of tuples of cam,chip combinations required
        
        """
        
        # -- Create chip and inversion array that contain information 
        #    on the orientations of the TESS ccd as given by manual. 
        #    Note that the chip array is flipped horizontally from 
        #    the manual as our positive x-axis goes to the right -- #
        chipArray = np.array([[(4,3),(1,2)],[(4,3),(1,2)],[(2,1),(3,4)],[(2,1),(3,4)]])
        invertArray = np.array([[(True,True),(False,False)],
                                [(True,True),(False,False)],
                                [(True,True),(False,False)],
                                [(True,True),(False,False)]])
        
        # -- Check north/south pointing and initialise cam array accordingly -- #
        if self.dec > 0:
            north = True
        else:
            north = False
            
        if north:
            camArray = np.array([4,3,2,1])
        else:
            camArray = np.array([1,2,3,4])
            
        
        # -- Find the chipArray index based on self.cam/chip -- #
        
        for i,cam in enumerate(camArray):
            if self.cam == cam:
                camIndex = i
                
        for i in range(len(chipArray[camIndex])):
            for j in range(len(chipArray[camIndex][i])):
                if self.chip == chipArray[camIndex][i][j]:
                    chipIndex = (i,j) # row, column
            
        total_index = (camIndex,chipIndex) # [camIndex,(in-cam-row,in-cam-column)]
        
        # -- Create error ellipse and use max/min values to find if the ellipse
        #    intersects the up,down,left,right box edges -- #
        ellipse = self.Cuts.ellipse
        
        right = False
        left = False
        up = False
        down = False
        
        if max(ellipse[0]) > self.Cuts.xsize:
            right = True
        if min(ellipse[0]) < 0:
            left = True
        if max(ellipse[1]) > self.Cuts.ysize:
            up = True
        if min(ellipse[1]) < 0:
            down = True
            
        # -- Check if inversion is required and adjust accordingly-- #
        self.invert = invertArray[total_index[0]][total_index[1][0]][total_index[1][1]]
       
        if self.invert:
            right2 = left
            left = right
            right = right2
            up2 = down
            down = up
            up = up2

        # -- Check for diagonals -- #
        upright = False
        upleft = False
        downright = False
        downleft = False
    
        if up and right:
            upright = True
        if up and left:
            upleft = True
        if down and right:
            downright = True
        if down and left:
            downleft = True
            
        # -- Calculate the neighbouring chip information. If px area of 
        #    neighbour chip is <70,000, chip is disregarded as unworthy 
        #    of full download process. -- #
        neighbour_chips = []    

        if left:
            if chipIndex[1] == 1:
                leftchip = camArray[camIndex],chipArray[camIndex][chipIndex[0]][0]
                
                if self.invert:
                    size = self.Cuts._prelim_size_check('right')
                    if size > 70000:
                        leftchip = (leftchip[0],leftchip[1],'Right')
                        neighbour_chips.append(leftchip)
                    else:
                        neighbour_chips.append('Right chip too small')
                else:
                    size = self.Cuts._prelim_size_check('left')
                    if size > 70000:
                        leftchip = (leftchip[0],leftchip[1],'Left')
                        neighbour_chips.append(leftchip)
                    else:
                        neighbour_chips.append('Left chip too small')
                
        if right:
            if chipIndex[1] == 0:
                rightchip = camArray[camIndex],chipArray[camIndex][chipIndex[0]][1]
                
                if self.invert:
                    size = self.Cuts._prelim_size_check('left')
                    if size > 70000:
                        rightchip = (rightchip[0],rightchip[1],'Left')
                        neighbour_chips.append(rightchip)
                    else:
                        neighbour_chips.append('Left chip too small')
                else:
                    size = self.Cuts._prelim_size_check('right')
                    if size > 70000:
                        rightchip = (rightchip[0],rightchip[1],'Right')
                        neighbour_chips.append(rightchip)
                    else:
                        neighbour_chips.append('Right chip too small')
                
        if up:
            if not (total_index[0] == 0) & (total_index[1][0] == 0):
                if total_index[1][0] == 0:
                    upCam = camIndex - 1
                    upCcd = (1,total_index[1][1])
                    upchip = camArray[upCam],chipArray[upCam][1][total_index[1][1]]
                else:
                    upCam = camIndex
                    upCcd = (0,total_index[1][1])
                    upchip = camArray[upCam],chipArray[upCam][0][total_index[1][1]]
                
                if self.invert:
                    size = self.Cuts._prelim_size_check('down')
                    if size > 70000:
                        upchip = (upchip[0],upchip[1],'Down')
                        neighbour_chips.append(upchip)
                    else:
                        neighbour_chips.append('Down chip too small')
                else:
                    size = self.Cuts._prelim_size_check('up')
                    if size > 70000:
                        upchip = (upchip[0],upchip[1],'Up')
                        neighbour_chips.append(upchip)
                    else:
                        neighbour_chips.append('Up chip too small')
                            
        if down:
            if not (total_index[0] == 3) & (total_index[1][0] == 1):
                if total_index[1][0] == 1:
                    downCam = camIndex + 1
                    downCcd = (0,total_index[1][1])
                    downchip = camArray[downCam],chipArray[downCam][0][total_index[1][1]]
                else:
                    downCam = camIndex
                    downCcd = (1,total_index[1][1])
                    downchip = camArray[downCam],chipArray[downCam][1][total_index[1][1]]
                
                if self.invert:
                    size = self.Cuts._prelim_size_check('up')
                    if size > 70000:
                        downchip = (downchip[0],downchip[1],'Up')
                        neighbour_chips.append(downchip)
                    else:
                        neighbour_chips.append('Up chip too small')
                else:
                    size = self.Cuts._prelim_size_check('down')
                    if size > 70000:
                        downchip = (downchip[0],downchip[1],'Down')
                        neighbour_chips.append(downchip)
                    else:
                        neighbour_chips.append('Down chip too small')

        
        if upright:
            if not (total_index[0] == 0) & (total_index[1][0] == 0) | (chipIndex[1] == 1):
                if total_index[1][0] == 0:
                    urCam = camIndex - 1
                    urchip = camArray[urCam],chipArray[urCam][1][1]
                else:
                    urCam = camIndex
                    urchip = camArray[urCam],chipArray[urCam][0][1]
                
                if self.invert:
                    size = self.Cuts._prelim_size_check('downleft')
                    if size > 70000:
                        urchip = (urchip[0],urchip[1],'Downleft')
                        neighbour_chips.append(urchip)
                    else:
                        neighbour_chips.append('Downleft chip too small')
                else:
                    size = self.Cuts._prelim_size_check('upright')
                    if size > 70000:
                        urchip = (urchip[0],urchip[1],'Upright')
                        neighbour_chips.append(urchip)
                    else:
                        neighbour_chips.append('Upright chip too small')
                    
        
        if upleft:
            if not (total_index[0] == 0) & (total_index[1][0] == 0) | (chipIndex[1] == 0):
                if total_index[1][0] == 0:
                    ulCam = camIndex - 1
                    ulchip = camArray[ulCam],chipArray[ulCam][1][0]
                else:
                    ulCam = camIndex
                    ulchip = camArray[ulCam],chipArray[ulCam][0][0]
                
                if self.invert:
                    size = self.Cuts._prelim_size_check('downright')
                    if size > 70000:
                        ulchip = (ulchip[0],ulchip[1],'Downright')
                        neighbour_chips.append(ulchip)
                    else:
                        neighbour_chips.append('Downright chip too small')
                else:
                    size = self.Cuts._prelim_size_check('upleft')
                    if size > 70000:
                        ulchip = (ulchip[0],ulchip[1],'Upleft')
                        neighbour_chips.append(ulchip)
                    else:
                        neighbour_chips.append('Upleft chip too small')

        if downright:
            if not (total_index[0] == 3) & (total_index[1][0] == 1) | (chipIndex[1] == 1):
                if total_index[1][0] == 1:
                    drCam = camIndex + 1
                    drchip = camArray[drCam],chipArray[drCam][0][1]
                else:
                    drCam = camIndex
                    drchip = camArray[drCam],chipArray[drCam][1][1]
                
                if self.invert:
                    size = self.Cuts._prelim_size_check('upleft')
                    if size > 70000:
                        drchip = (drchip[0],drchip[1],'Upleft')
                        neighbour_chips.append(drchip)
                    else:
                        neighbour_chips.append('Upleft chip too small')
                else:
                    size = self.Cuts._prelim_size_check('downright')
                    if size > 70000:
                        drchip = (drchip[0],drchip[1],'Downright')
                        neighbour_chips.append(drchip)
                    else:
                        neighbour_chips.append('Downright chip too small')
            
        if downleft:
            if not (total_index[0] == 3) & (total_index[1][0] == 1) | (chipIndex[1] == 0):
                if total_index[1][0] == 1:
                    dlCam = camIndex + 1
                    dlchip = camArray[dlCam],chipArray[dlCam][0][0]
                else:
                    dlCam = camIndex
                    dlchip = camArray[dlCam],chipArray[dlCam][1][0]
                
                if self.invert:
                    size = self.Cuts._prelim_size_check('upright')
                    if size > 70000:
                        dlchip = (dlchip[0],dlchip[1],'Upright')
                        neighbour_chips.append(dlchip)
                    else:
                        neighbour_chips.append('Upright chip too small')
                else:
                    size = self.Cuts._prelim_size_check('downleft')
                    if size > 70000:
                        dlchip = (dlchip[0],dlchip[1],'Downleft')
                        neighbour_chips.append(dlchip)
                    else:
                        neighbour_chips.append('Downleft chip too small')

        # -- prints information -- #
        if verbose & (self.verbose > 0):

            if north:
                print('Pointing: North')
            else:
                print('Pointing: South')
            
            print(f'This chip: Camera {self.cam}, Chip {self.chip}.')
            print('------------------------------')
            print('Neighbouring Chips Required:')
            if neighbour_chips != []:
                for item in neighbour_chips:
                    if type(item) == str:
                        print(item)
                    else:
                        print(f'Camera {item[0]}, Chip {item[1]} ({item[2]}).')
                        
            else:
                print('No neighbour chips available/required.')
            print('\n')
                        
        # -- Removes disregarded chip info to create self.neighbours -- #
        if neighbour_chips != []:
            self.neighbours = []
            for item in neighbour_chips:
                if type(item) == tuple:
                    self.neighbours.append(item[:-1])

    def get_neighbour_chips(self,number=None):
        """
        Download neighbouring chip data.
        
        ------
        Options
        ------
        number : int
            number of files to download per chip

        """

        if self.neighbours is None:
            self.find_neighbour_chips()
        
        if len(self.neighbours) < 1:
            print('No neighbour chips!')
            return
        
        numNeighbours = len(self.neighbours)
        
        # -- Iterate through neighbour chips and call self.download() -- #
        for i,neighbour in enumerate(self.neighbours):
            cam = neighbour[0]
            chip = neighbour[1]

            if os.path.exists(f'{self.path}/Cam{cam}Chip{chip}'):
                print(f'Cam {cam} Chip {chip} data already downloaded.')
                print('\n')
            else:
                print(f'Cam {cam} Chip {chip} downloading ({i+1} of {numNeighbours}).')
                print('\n')
                self.download(cam=cam,chip=chip,number=number)
        
    def get_neighbour_cubes(self):
        """
        Make data cubes of neighbour chips.
        """
        
        if self.neighbours is None:
            self.find_neighbour_chips()
        
        if len(self.neighbours) < 1:
            print('No neighbour chips!')
            return
        
        numNeighbours = len(self.neighbours)
        
        # -- Iterate over neighbour chips and call self.make_cube() -- #
        for i,neighbour in enumerate(self.neighbours):
            cam = neighbour[0]
            chip = neighbour[1]
                        
            if self.verbose > 0:
                print(f'Cubing Cam {cam} Chip {chip} ({i+1} of {numNeighbours})')
            try:
                self.make_cube(cam, chip, ask=False)
                print('\n')
            except:
                if self.verbose > 0:
                    print(f'Cubing Cam {cam} Chip {chip} Failed! :( ')
                    print('\n')
                pass                    

    def get_neighbour_cuts(self):
        """
        Make appropriate cuts for neighbour chips.
        """
        
        if self.neighbours is None:
            self.find_neighbour_chips()
        
        if len(self.neighbours) < 1:
            print('No neighbour chips!')
            return
        
        numNeighbours = len(self.neighbours)
        
        # -- Iterate over neighbour chips and call self.make_cuts() -- #
        for i,neighbour in enumerate(self.neighbours):
            cam = neighbour[0]
            chip = neighbour[1]

            self.find_cuts(cam,chip,plot=False)
            if self.verbose > 1:
                print(f'Cutting Cam {cam} Chip {chip} ({i+1} of {numNeighbours})')
            
            try:
                self.make_cuts(cam,chip)
                print('\n')

            except:
                if self.verbose > 0:
                    print(f'Cutting Cam {cam} Chip {chip} failed!')
                    print('\n')
                pass

    def get_neighbour_reductions(self):
        """
        Make appropriate reductions for neighbour chips.
        """
        
        if self.neighbours is None:
            self.find_neighbour_chips()
        
        if len(self.neighbours) < 1:
            print('No neighbour chips!')
            return      
          
        numNeighbours = len(self.neighbours)
        
        # -- Iterate over neighbour chips and call self.reduce() -- #
        for i,neighbour in enumerate(self.neighbours):
            cam = neighbour[0]
            chip = neighbour[1]

            ts = t()  # timeStart
            
            if self.verbose > 0:
                print(f'Reducing Cam {cam} Chip {chip} ({i+1} of {numNeighbours})')
            try:
                self.reduce(cam,chip)
                print('\n')
            except:
                if self.verbose > 0:
                    print('Reducing Cam {} Chip {} Failed :( Time Elapsed: {:.2f} mins.'.format(cam,chip,(t()-ts)/60))
                    print('\n')

    def entire_chip(self,cam=None,chip=None):
        """
        Downloads, cubes, cuts, reduces. All in one.        
        """

        if cam is None:
            cam = self.cam
        if chip is None:
            chip = self.chip

        # -- Requires dedicated event path -- #
        if self.eventname is None:
            print('Downloading to temporary folder is a bad idea!')
            return
        
        ts = t() # timeStart

        if self.verbose > 0:
            _Print_buff(50,f'Downloading {self.eventname} Cam {cam} Chip {chip}')
        self.download(cam=cam,chip=chip)

        if self.verbose > 0:
            _Print_buff(50,f'Cubing {self.eventname} Cam {cam} Chip {chip}')
        self.make_cube(cam=cam,chip=chip,ask=False)

        self.find_cuts(cam=cam,chip=chip,plot=False)

        if self.verbose > 0:
            _Print_buff(50,f'Cutting {self.eventname} Cam {cam} Chip {chip}')
        self.make_cuts(cam=cam,chip=chip)

        if self.verbose > 0:
            _Print_buff(50,f'Reducing {self.eventname} Cam {cam} Chip {chip}')
        self.reduce(cam=cam,chip=chip)

        print(f'Chip Complete (Time Taken = {((t()-ts)/60):.2f} minutes)')

    def _old_entire(self):
        """
        Finds, downloads, cubes, cuts, reduces. All in one.        
        """
        
        ts = t() # timeStart
        
        self.find_cuts(plot=False) # 
        
        self.get_neighbour_chips()

        if self.verbose > 0:
            print('Cubing Main Chip')
        self.make_cube(ask=False)
        
        if self.neighbours is not None:
            if self.verbose > 0:
                print('\n')
                print(f'------------{self.eventname} -- {len(self.neighbours)} Chips to Compute----------')
                print('\n')
            
                print('---------Getting Chip Cubes---------')
                print('\n')
            self.get_neighbour_cubes()
            
        if self.verbose > 0:
            print('---------Getting Chip Cuts---------')
            print('\n')
            print('Cutting Main Chip')

        self.make_cuts()
        print('\n')
        
        if self.neighbours is not None:
            self.get_neighbour_cuts()
                
        if self.verbose > 0:
            print('------Getting Chip Reductions------')
            print('\n')
            print('Reducing Main Chip')        
        
        self.reduce()
        print('\n')

        if self.neighbours is not None:
            self.get_neighbour_reductions()
            
        if self.verbose > 0:
            print(f'-----------{self.eventname} Complete (Total Time: {((t()-ts)/3600):.2f} hrs)---------')

    
    def full(self):
        """
        Downloads, cubes, cuts, reduces. For all chips!
        """

        ts = t() 

        _Print_buff(70,self.eventname)
        print('\n')
        self.entire_chip()
        self.find_neighbour_chips(verbose=False)
        
        if self.neighbours is not None:
            print('\n')
            _Print_buff(70,f'{len(self.neighbours)} Neighbour Chips')
            for neighbour in self.neighbours:
                cam = neighbour[0]
                chip = neighbour[1]
                self.entire_chip(cam,chip)
            
        print('\n')
        _Print_buff(70,f'{self.eventname} Complete (Time Taken {((t()-ts)/3600):.2f} Hours)')


    def detect_events(self,cam=None,chip=None,cut=None,significane=4,min_length=2,min_px=1,max_events=None,plot=True,asteroids=True):
        """
        Reduces all cuts on a chip using TESSreduce. bkg correlation 
        correction and final calibration are disabled due to time constraints.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        chip : int
            desired chip
        cut : int
            cut number (those of self.Cuts)
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
        self.Detector : TessDetector Class
            has useful functions such as
            - make_video : generate a video of a chosen event
            - event_coords : printout the real world coords of the event
            - tessreduce_event : returns a tessreduce object for full reduction of the event_coords

        """

        if cam is None:
            cam = self.cam
        if chip is None:
            chip = self.chip
        if cut is None:
            cut = 1

        # -- If previous attempt made on same cut, no need to reinitialise -- #
        if self.Detector is None:
            initialise = True
        elif (self.Detector.cam != cam) or (self.Detector.chip != chip) or (self.Detector.cut != cut):
            initialise = True
        else:
            initialise = False
            self.Detector.eventtime = self.eventtime    # if eventtime has changed, adjust accordingly

        if initialise:
            self.find_cuts(cam,chip,plot=False)
            self.Detector = TessDetector(self.Cuts,self.eventname,self.eventtime,self.sector,cam,chip,cut)

        # -- Detect! -- #
        self.Detector.detect(significance=significane,
                             min_length=min_length,
                             min_px=min_px,
                             max_events=max_events,
                             plot=plot,asteroids=asteroids)
        
        

        
