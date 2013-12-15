import numpy as np
from urllib import urlretrieve
import pdb
import matplotlib.pylab as pl

http_base='http://api.tiles.mapbox.com/v2/rkeisler.gh8kebdo/'
#savepath='/farmshare/user_data/rkeisler/img/'
basepath = '/Users/rkeisler/Desktop/satellite/'
imgpath = basepath+'img/'
labelpath = basepath+'label/'


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

def latlong_rectange_to_xyz(lat1, lat2, lon1, lon2, zoom):
    lat_min=np.min([lat1,lat2])
    lat_max=np.max([lat1,lat2])    
    lon_min=np.min([lon1,lon2])
    lon_max=np.max([lon1,lon2])        
    x_min, y_max, zoom = latlong_to_xyz(lat_min, lon_min, zoom)
    x_max, y_min, zoom = latlong_to_xyz(lat_max, lon_max, zoom)
    return x_min, x_max, y_min, y_max

def xyz_to_savename(x,y,z,prefix='tmp'):
    return prefix+'_x%i_y%i_z%i'%(x,y,z)+'.png'

def download_one(x,y,zoom,prefix='tmp'):
    url=http_base+xyz_to_ZXY_string(x,y,zoom)+'.png'
    savename=imgpath+xyz_to_savename(x,y,zoom,prefix=prefix)
    urlretrieve(url, savename)

def download_chunk(name, zoom, download=True):
    d=define_chunk(name)
    download_rectangle(d['lat1'],d['lat2'],
                       d['lon1'],d['lon2'],
                       zoom, prefix=d['prefix'], 
                       download=download)
    
def download_rectangle(lat1, lat2, lon1, lon2, 
                       zoom, prefix='tmp', download=True):
    x_min, x_max, y_min, y_max = latlong_rectange_to_xyz(lat1, lat2, lon1, lon2, zoom)
    n_x = x_max-x_min
    n_y = y_max-y_min
    n_tiles = n_x*n_y
    x_count=0
    print 'Downloading X=(%i,%i), Y=(%i,%i)'%(x_min,x_max,y_min,y_max)
    print 'n_x: %i'%n_x
    print 'n_y: %i'%n_y
    print 'That is %i tiles.'%n_tiles
    if not(download): return
    for x_tmp in range(x_min, x_max):
        x_count+=1
        print '%i/%i'%(x_count,n_x)
        for y_tmp in range(y_min, y_max):
            download_one(x_tmp,y_tmp,zoom,prefix=prefix)
    return

def define_chunk(name):
    sf1=dict(prefix='sf1',
             lat1=30.0, lax_max=30.1, 
             lon1=-50.0, lon2=-49.9)

    chi=dict(prefix='chi', 
             lat1=hms_to_deg(41,46,24.77),
             lat2=hms_to_deg(41,59,51.98),
             lon1=hms_to_deg(-87,43,44.02),
             lon2=hms_to_deg(-87,33,53.51))

    atx=dict(prefix='atx',
             lat1=hms_to_deg(30,20,21.95),
             lat2=hms_to_deg(30,12,32.97),
             lon1=hms_to_deg(-97,50,33.81),
             lon2=hms_to_deg(-97,38,12.31))
    chunks=dict(sf1=sf1, chi=chi, atx=atx)
    return chunks[name]

def hms_to_deg(hour, min, sec):
    return np.sign(hour)*(np.abs(hour)+min/60.+sec/3600.)

    
def label_data(prefix, size=100, savename=None):
    from glob import glob
    from os.path import basename
    from PIL import Image
    from os.path import isfile
    
    if savename==None: savename=labelpath+'label_'+prefix+'.txt'


    # We want to avoid labeling an image twice, so keep track
    # of what we've labeled in previous labeling sessions.
    if isfile(savename):
        fileout = open(savename,'r')
        already_seen = [line.split(',')[0] for line in fileout]
        fileout.close()
    else: already_seen = []

    # Now reopen the file for appending.
    fileout = open(savename,'a')
    pl.ion()
    pl.figure(1,figsize=(9,9))
    files = glob(imgpath+prefix+'*.png')
    for file in np.random.choice(files, size=size, replace=False):
        if basename(file) in already_seen: continue
        pl.clf()
        pl.subplot(1,1,1)
        pl.imshow(np.array(Image.open(file)))
        pl.title(file)
        pl.axis('off')
        pl.draw()
        label = get_one_char()
        if label=='q': break
        fileout.write(basename(file)+','+label+'\n')
        print file,label
    fileout.close()
    return

def get_one_char():
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
    finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_label(prefix):
    savename = labelpath+'label_'+prefix+'.txt'
    file=open(savename,'r')
    n={}
    for line in file:
        tmp=line.split(',')
        if is_number(tmp[1]): n[tmp[0]]=int(tmp[1])
        else: n[tmp[0]]=0
    file.close()
    return n



def try_train(prefix='atx', nside=32):
    from PIL import Image
    tmp=read_label(prefix)
    X=[]
    y=[]
    for name,label in tmp.iteritems():
        img_name = imgpath+name
        img = Image.open(img_name)
        if nside!=256: img=img.resize((nside,nside),Image.ANTIALIAS)
        img = np.array(img)
        if False:
            print img.shape
            pl.imshow(img)
            pdb.set_trace()
        X.append(img.ravel())
        y.append(label>0)
    X = np.vstack(np.array(X))
    y = np.array(y).astype(int)

    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    from sklearn.dummy import DummyClassifier
    #rf = DummyClassifier(strategy='stratified')
    rf = ExtraTreesClassifier(n_estimators=200, n_jobs=6, max_depth=None, max_features=0.01)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
    print '...fitting...'
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba[:,1])
    auc = metrics.auc(fpr, tpr)
    pl.clf(); pl.plot(fpr, tpr)
    pl.plot(fpr, fpr/0.065, 'r--'); pl.ylim(0,1); pl.xlim(0,1)
    pl.title('AUC: %0.3f'%auc)
    pdb.set_trace()
        
def images_to_batches(prefix='atx', nside=32):
    import pickle
    from glob import glob
    from PIL import Image
    from os import system
    tmp=read_label(prefix)
    X=[]
    y=[]
    for name,label in tmp.iteritems():
        img_name = imgpath+name
        img = Image.open(img_name)
        if nside!=256: img=img.resize((nside,nside),Image.ANTIALIAS)
        img = np.array(img)
        img = np.rollaxis(img, 2)  # only for cuda convnet
        img = img.reshape(-1)
        X.append(img)
        y.append(label>0)
    X = np.vstack(np.array(X))
    y = np.array(y).astype(int)

    nbatches = 2
    nimg_total, nfeatures = X.shape
    nimg_per_batch = np.floor(1.*nimg_total/nbatches)
    system('rm batches/data_batch*')

    global_counter = -1
    sum_data = np.zeros(nfeatures)
    for ibatch in np.arange(nbatches):
        data = np.array(X[ibatch*nimg_per_batch:(ibatch+1)*nimg_per_batch], dtype=np.float).T
        labels = y[ibatch*nimg_per_batch:(ibatch+1)*nimg_per_batch]
        sum_data += np.sum(data,axis=1)
        print '...dumping...'
        output = {'data':np.array(data,dtype=np.uint8),
                  'labels':list(np.array(labels, dtype=np.int))}
        pickle.dump(output, open('batches/data_batch_%i'%ibatch, 'w'))
    mean_data = 1.*sum_data/nbatches/nimg_per_batch
    meta = {'num_cases_per_batch':nimg_per_batch, 
            'num_vis':nfeatures, 
            'data_mean':mean_data[:,np.newaxis],
            'label_names':['notpool','pool']}
    pickle.dump(meta, open('batches/batches.meta', 'w'))    


