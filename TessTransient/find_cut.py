from glob import glob
import numpy as np
import math

from astropy.io import fits
from astropy import wcs

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

class FindCut():
    """
    Class for finding appropriate cuts for TessTransient.
    """

    def __init__(self,ra,dec,error,path):

        self.ra = ra
        self.dec = dec
        self.error = error
        self.path = path

        self.xsize = 2078  # x size of downloaded fits image
        self.ysize = 2136  # y size of downloaded fits image

        self.wcs = None

        self._get_wcs() 

    def _get_wcs(self):
        """
        Get WCS data from a file in the path
        """

        if glob(f'{self.path}/*ffic.fits'):
            filepath = glob(f'{self.path}/*ffic.fits')[0]
        else:
            print('No Data!')
            return
        file = _Extract_fits(filepath)
        wcsItem = wcs.WCS(file[1].header)
        self.wcs = wcsItem
        file.close()

        self.grb_px = self.wcs.all_world2pix(self.ra,self.dec,0)

    def _get_err_px(self):
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
        
        return errpixels
    
    def _cutoff_err_ellipse(self,ellipse):
        """
        Cutoff error ellipse at chip boundary.
        """
        
        # -- Finds where ellipse is inside ffi, new ellipse = og ellipsed indexed at the places -- #
        where = np.where((ellipse[0] >= 0) & (ellipse[0] <= self.xsize) & (ellipse[1] >= 0) & (ellipse[1] <= self.ysize))
        ellipse_reduced = ellipse[:,where]
        ellipse_reduced = ellipse_reduced[:,0,:]
        
        return ellipse_reduced
    
    def _find_full_bounding_box(self):
        """
        Finds the full bounding box for the error ellipse.
        """

        xEdges = (44,2092) 
        yEdges = (0,2048)

        ellipse = self.ellipse

        if len(ellipse[0]) == 0:
            cornerLB = (0,0)
            cornerRB = (self.xsize,0)
            cornerLU = (0,self.ysize)
            cornerRU = (self.xsize,self.ysize)

        else:
            # -- Finds box min/max with 50 pixel buffer, ensures they are inside ffi -- #
            boxXmin = math.floor(min(ellipse[0])) - 50 
            boxYmin = math.floor(min(ellipse[1])) - 50 
            boxXmax = math.ceil(max(ellipse[0])) + 50
            boxYmax = math.ceil(max(ellipse[1])) + 50
        
            if boxXmin < xEdges[0]:
                boxXmin = xEdges[0]
            if boxYmin < yEdges[0]:
                boxYmin = yEdges[0] 
            if boxXmax > xEdges[1]:
                boxXmax = xEdges[1]
            if boxYmax > yEdges[1]:
                boxYmax = yEdges[1]
            
            upLength = math.ceil(abs(self.grb_px[1]-boxYmax))
            downLength = math.floor(abs(self.grb_px[1]-boxYmin))
            leftLength = math.floor(abs(self.grb_px[0]-boxXmin))
            rightLength = math.ceil(abs(self.grb_px[0]-boxXmax))
    
            # -- Finds corners of box based on above -- #
            cornerLB = (self.grb_px[0]-leftLength,self.grb_px[1]-downLength)
            cornerRB = (self.grb_px[0]+rightLength,self.grb_px[1]-downLength)
            cornerLU = (self.grb_px[0]-leftLength,self.grb_px[1]+upLength)
            cornerRU = (self.grb_px[0]+rightLength,self.grb_px[1]+upLength)

        # -- Ensures corners are inside ffi -- #
        cornerLB = (max([xEdges[0],cornerLB[0]]),max(yEdges[0],cornerLB[1]))
        cornerRB = (min([xEdges[1],cornerRB[0]]),max(yEdges[0],cornerRB[1]))
        cornerLU = (max([xEdges[0],cornerLU[0]]),min(yEdges[1],cornerLU[1]))
        cornerRU = (min([xEdges[1],cornerRU[0]]),min(yEdges[1],cornerRU[1]))
    
        # -- Calculates the x,y radii of the box -- #
        xRad = (cornerRB[0]-cornerLB[0])/2
        yRad = (cornerRU[1]-cornerRB[1])/2
                
        # -- Finds patch centre in px space -- #
        #self.cutCentrePx = (cornerLB[0]+xRad,cornerLB[1]+yRad)
    
        return cornerLB,xRad,yRad
    
    def _find_split(self,xRad,yRad):
        """
        Checks if proposed cut would be larger 2/5 or 4/5 total area.
        If size > 4/5, cut is quartered. Else if size > 2/5, cut is halved. 
        """

        # -- Finds area of proposed cut -- #
        cut_area = 4*xRad*yRad
        
        # -- If small area, no splitting -- #
        if cut_area < (2048*2048/(5/2)):
            self.split = None

        # -- If large area, quarter -- #
        elif cut_area > 4/5 * 2048*2048:
            self.split = 'quarter'            
        
        # -- If mid area, halve -- #
        else:
            if xRad >= yRad:
                self.split = 'vert'
            else:
                self.split = 'hor'
            
    def _split_cuts(self,cornerLB,xRad,yRad):
        """
        Splits the cut in 2 or 4. Calculates sizes and corners.
        """

        cutCorners = []

        if self.split is None:
            cutCorners.append(cornerLB)
            cut_sizes = (2*xRad,2*yRad)

        if self.split == 'vert':
            corner1 = cornerLB
            corner2 = (cornerLB[0]+xRad,cornerLB[1])
            cutCorners.append(corner1)
            cutCorners.append(corner2)
            cut_sizes = (xRad,2*yRad)

        elif self.split == 'hor':
            corner1 = cornerLB
            corner2 = (cornerLB[0],cornerLB[1]+yRad)
            cutCorners.append(corner1)
            cutCorners.append(corner2)
            cut_sizes = (2*xRad,yRad)

        elif self.split == 'quarter':
            corner1 = cornerLB
            corner2 = (cornerLB[0]+xRad,cornerLB[1])
            corner3 = (cornerLB[0],cornerLB[1]+yRad)
            corner4 = (cornerLB[0]+xRad,cornerLB[1]+yRad)
            cutCorners.append(corner1)
            cutCorners.append(corner2)
            cutCorners.append(corner3)
            cutCorners.append(corner4)
            cut_sizes = (xRad,yRad)

        return cut_sizes,cutCorners

    def _narrow_cut(self,LB,cutSizes):
        """
        Narrows a cut to better bound error region.
        """

        ellipse = self.ellipse

        # -- Finds corners and thus cut size -- #
        RB = (LB[0]+cutSizes[0],LB[1])
        LU = (LB[0],LB[1]+cutSizes[1])
    
        xsize = (LB[0],RB[0])
        ysize = (LB[1],LU[1])
        
        # -- Finds where ellipse is inside cut, new ellipse = og ellipsed indexed at the places -- #
        where = np.where((ellipse[0] >= xsize[0]) & (ellipse[0] <= xsize[1]) & (ellipse[1] >= ysize[0]) & (ellipse[1] <= ysize[1]))
        ellipse_reduced = ellipse[:,where]
        ellipse_reduced = ellipse_reduced[:,0,:]
    
        # -- If no ellipse inside cut, implied no narrowing required -- #
        if ellipse_reduced.shape[1] == 0:
            xRad = (xsize[1]-xsize[0])/2
            yRad = (ysize[1]-ysize[0])/2
            return LB,xRad,yRad
    
        # -- Finds box min/max with 50 pixel buffer -- #
        boxXmin = math.floor(min(ellipse_reduced[0])) - 50 
        boxYmin = math.floor(min(ellipse_reduced[1])) - 50 
        boxXmax = math.ceil(max(ellipse_reduced[0])) + 50
        boxYmax = math.ceil(max(ellipse_reduced[1])) + 50
        
        # -- Finds where ellipse is inside cut X range -- #
        where = np.where((ellipse[0] >= xsize[0]) & (ellipse[0] <= xsize[1])) 
        ellipseX = ellipse[:,where]
        ellipseX = ellipseX[:,0,:]
        
        # -- Finds where ellipse is inside cut Y range -- #
        where = np.where((ellipse[1] >= ysize[0]) & (ellipse[1] <= ysize[1]))
        ellipseY = ellipse[:,where]
        ellipseY = ellipseY[:,0,:]    
    
        # -- Sets boxX/Y min/max -- #
        if (boxXmin < xsize[0]) | (min(ellipseY[0]) < xsize[0]):
            boxXmin = xsize[0]
        if (boxYmin < ysize[0]) | (min(ellipseX[1]) < ysize[0]):
            boxYmin = ysize[0] 
        if (boxXmax > xsize[1]) | (max(ellipseY[0]) > xsize[1]):
            boxXmax = xsize[1]
        if (boxYmax > ysize[1]) | (max(ellipseX[1]) > ysize[1]):
            boxYmax = ysize[1]

        LB = (boxXmin,boxYmin)
    
        xRad = (boxXmax - boxXmin)/2
        yRad = (boxYmax - boxYmin)/2

        return LB, xRad, yRad

    def _find_cuts(self):
        """
        Find the cut!
        """

        self.ellipse = self._get_err_px() # gets ellipse
        cornerLB,xRad,yRad = self._find_full_bounding_box() # finds bounding box
        self._find_split(xRad,yRad) # finds split based on size

        cutsizes, corners = self._split_cuts(cornerLB,xRad,yRad) # splits bounding box into cuts

        # -- Gathers information for each cut -- #
        cutCorners = []
        cutCentrePx = []
        cutSizes = []
        for corner in corners:
            cornerLB,xRad,yRad = self._narrow_cut(corner,cutsizes)
            cutCorners.append(cornerLB)
            cutCentrePx.append((cornerLB[0]+xRad,cornerLB[1]+yRad))
            cutSizes.append((2*xRad,2*yRad))

        self.cutCorners = np.array(cutCorners)
        self.cutCentrePx = np.array(cutCentrePx)
        self.cutSizes = np.array(cutSizes)
        self.cutCentreCoords = np.array(self.wcs.all_pix2world(self.cutCentrePx[:,0],self.cutCentrePx[:,1],0)).transpose()

    def _prelim_size_check(self,border):
        """
        Finds, based on initial chip cube, whether proposed neighbour cut is 
        large enough for the whole download process to be worth it.
        
        ------
        Inputs
        ------
        border : str
            which border to look at, defines conditions. 
        """
        
        ellipse = self._get_err_px()

        # -- Generates ellipse cutoff conditions based on border direction -- #
        if border == 'left':
            condition = (ellipse[0] <= 0) & (ellipse[1] >= 0) & (ellipse[1] <= self.ysize)
        elif border == 'right':
            condition = (ellipse[0] >= self.xsize) & (ellipse[1] >= 0) & (ellipse[1] <= self.ysize)
        elif border == 'up':
            condition = (ellipse[0] >= 0) & (ellipse[0] <= self.xsize) & (ellipse[1] >= self.ysize)
        elif border == 'down':
            condition = (ellipse[0] >= 0) & (ellipse[0] <= self.xsize) & (ellipse[1] <= 0)
        elif border == 'upleft':
            condition = (ellipse[0] <= 0) & (ellipse[1] >= self.ysize) 
        elif border == 'upright':
            condition = (ellipse[0] >= self.xsize) & (ellipse[1] >= self.ysize) 
        elif border == 'downleft':
            condition = (ellipse[0] <= 0) & (ellipse[1] <= 0) 
        elif border == 'downright':
            condition = (ellipse[0] >= self.xsize) & (ellipse[1] <= 0) 

        # -- Cuts ellipse -- #
        where = np.where(condition)
        ellipse = ellipse[:,where]
        ellipse = ellipse[:,0,:]
        
        # -- Calculate size of cut required to encompass ellipse region -- #
        if len(ellipse[0]) > 0:
            x1 = max(ellipse[0])
            x2 = min(ellipse[0])
            x = abs(x1 - x2)
            
            y1 = max(ellipse[1])
            y2 = min(ellipse[1])
            y = abs(y1 - y2)
            
            size = x*y
        
        else:
            size = 0
            
        return size
