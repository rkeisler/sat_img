import numpy as np
from urllib import urlretrieve
import pdb
http_base='http://api.tiles.mapbox.com/v2/rkeisler.gh8kebdo/'
savepath='/farmshare/user_data/rkeisler/img/'

def latlong_to_xyz(lat_deg, lon_deg, zoom):
    lat_rad = lat_deg*np.pi/180.
    lon_rad = lon_deg*np.pi/180.
    n = 2. ** zoom
    xtile = n * ((lon_deg + 180) / 360)
    ytile = n * (1 - (np.log(np.tan(lat_rad) + 1./np.cos(lat_rad)) / np.pi)) / 2.
    return int(xtile), int(ytile), zoom

def xyz_to_ZXY_string(x,y,z):
    return '%i/%i/%i'%(z,x,y)

def latlong_to_ZXY_string(lat_deg, lon_deg, zoom):
    x,y,z = latlong_to_xyz(lat_deg, lon_deg, zoom)
    return xyz_to_ZXY_string(x,y,z)

def latlong_rectange_to_xyz(lat_min, lat_max, lon_min, lon_max, zoom):
    x_min, y_max, zoom = latlong_to_xyz(lat_min, lon_min, zoom)
    x_max, y_min, zoom = latlong_to_xyz(lat_max, lon_max, zoom)
    return x_min, x_max, y_min, y_max

def xyz_to_savename(x,y,z,prefix='tmp'):
    return prefix+'_%i_%i_%i'%(x,y,z)+'.png'

def download_one(x,y,zoom,prefix='tmp'):
    url=http_base+xyz_to_ZXY_string(x,y,zoom)+'.png'
    savename=savepath+xyz_to_savename(x,y,zoom,prefix=prefix)
    urlretrieve(url, savename)
    
def download_rectangle(lat_min, lat_max, lon_min, lon_max, 
                       zoom, prefix='tmp', download=True):
    x_min, x_max, y_min, y_max = latlong_rectange_to_xyz(lat_min, lat_max, lon_min, lon_max, zoom)
    n_x = np.abs(x_max-x_min+1)
    n_y = np.abs(y_max-y_min+1)
    n_tiles = n_x*n_y
    x_count=0
    print 'Downloading X=(%i,%i), Y=(%i,%i)'%(x_min,x_max,y_min,y_max)
    print 'That is %i tiles.'%n_tiles
    for x_tmp in range(x_min, x_max):
        x_count+=1
        print '%i/%i'%(x_count,n_x)
        for y_tmp in range(y_min, y_max):
            if download: download_one(x_tmp,y_tmp,zoom,prefix=prefix)
    return

def define_chunks():
    sf1=dict(prefix='sf1',
             lat_min=30.0, lax_max=30.1, 
             lon_min=-50.0, lon_max=-49.9)

    chi=dict(prefix='chi', 
             lat_min=hms_to_deg(41,46,24.77),
             lat_max=hms_to_deg(41,59,51.98),
             long_min=hms_to_deg(-87.728894), 
             long_max=hms_to_deg(-87.564863))

    atx=dict(prefix='atx')
    
    chunks=dict(sf1=sf1)
    return chunks

def hms_to_deg(hour, min, sec):
    return np.sign(hour)*(np.abs(hour)+min/60.+sec/3600.)

    

    
    
