import numpy as np
from urllib import urlretrieve
import pdb
import ipdb
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


def load_labeled(prefix='atx', nside=32, quick=True):
    import cPickle as pickle
    savename='tmp_train_'+prefix+'_%i'%nside+'.pkl'
    if quick:
        X, y = pickle.load(open(savename,'r'))
        return X, y
    from PIL import Image
    tmp=read_label(prefix)
    X=[]; y=[]
    for name,label in tmp.iteritems():
        img_name = imgpath+name
        img = Image.open(img_name)
        if nside!=256: img=img.resize((nside,nside),Image.ANTIALIAS)
        img = np.array(img)
        if False:
            print img.shape
            pl.imshow(img)
            pdb.set_trace()
        X.append(img)
        y.append(label>0)
    X = np.array(X)
    y = np.array(y).astype(int)    
    pickle.dump((X,y), open(savename, 'w'))
    return X,y


def get_features(X_img, ncolors=2):
    nside=X_img.shape[1]
    sigmas=np.array([4,8,16])*nside/32.
    colors=[]; 
    for i in range(ncolors):
        colors.append(np.array([154, 211,  205]) + np.random.randint(-10,high=10,size=3))
    x,y=np.mgrid[0:nside,0:nside]-np.mean(np.arange(nside))
    r=np.sqrt(x**2. + y**2.)
    fx = np.fft.fft2(X_img, axes=(1,2))
    features = []
    for sigma in sigmas:
        gauss=np.exp(-0.5*(-r/sigma)**2.)
        gauss/=np.sum(gauss)
        kern=np.dstack((gauss,gauss,gauss))
        fkern = np.fft.fft2(kern,axes=(0,1))
        smx=np.real(np.fft.ifft2((fx*fkern),axes=(1,2)))
        for color in colors:
            tmp = np.sum(smx*color,axis=-1)
            max_tmp = np.max(np.max(tmp,axis=-1),axis=-1)
            med_tmp = np.median(np.median(tmp,axis=-1),axis=-1)
            features.append(max_tmp)
            features.append(med_tmp)
    features = np.vstack(features).T
    return features


def get_features2(X_img, ncolors=40, thresh=20):
    nside=X_img.shape[1]
    sigmas=np.array([1,3,5])*nside/32.

    # get smoothing kernel
    x,y=np.mgrid[0:nside,0:nside]-np.mean(np.arange(nside))
    r=np.sqrt(x**2. + y**2.)

    # get colors
    colors=[]; 
    for i in range(ncolors):
        colors.append(np.array([154, 211,  205]) + np.random.randint(-50,high=30,size=3))
        colors.append(np.random.randint(0,high=255,size=3))
    features = []

    for color in colors:
        dist_color = np.sqrt(np.sum((X_img - np.array(color))**2.,axis=-1))
        ok_color = np.array(dist_color<thresh, dtype=np.float)
        fok_color = np.fft.fft2(ok_color)
        for sigma in sigmas:
            gauss=np.exp(-0.5*(-r/sigma)**2.)
            gauss/=np.sum(gauss)    
            fkern = np.fft.fft2(gauss).conjugate()
            sm_ok_color = np.real(np.fft.ifft2(fok_color*fkern))
            max_sm = np.max(np.max(sm_ok_color,axis=-1),axis=-1)
            sum_sm = np.sum(np.sum(sm_ok_color,axis=-1),axis=-1)
            features.append(max_sm)
            features.append(sum_sm)

    features = np.vstack(features).T
    return features    


    fx = np.fft.fft2(X_img, axes=(1,2))
    features = []
    for sigma in sigmas:
        gauss=np.exp(-0.5*(-r/sigma)**2.)
        gauss/=np.sum(gauss)
        kern=np.dstack((gauss,gauss,gauss))
        fkern = np.fft.fft2(kern,axes=(0,1))
        smx=np.real(np.fft.ifft2((fx*fkern),axes=(1,2)))
        for color in colors:
            tmp = np.sum(smx*color,axis=-1)
            max_tmp = np.max(np.max(tmp,axis=-1),axis=-1)
            med_tmp = np.median(np.median(tmp,axis=-1),axis=-1)
            features.append(max_tmp)
            features.append(med_tmp)
    features = np.vstack(features).T


   

def try_train(prefix='atx', nside=32):
    X_img,y=load_labeled(prefix=prefix,nside=nside)
    X = get_features2(X_img)

    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics

    rf = ExtraTreesClassifier(n_estimators=200, n_jobs=6, max_depth=None, max_features=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.9)
    print '...fitting...'
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba[:,1])
    auc = metrics.auc(fpr, tpr)
    pl.clf(); pl.plot(fpr, tpr, 'b-o')
    pl.plot(fpr, fpr/0.065, 'r--'); pl.ylim(0,1); pl.xlim(0,1)
    pl.title('AUC: %0.3f'%auc)

    ipdb.set_trace()

        
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


def blue_gaussian():
    blue = np.array([154, 211,  205]) + np.random.randint(-150,high=30,size=3)
    blue = np.random.randint(0,high=255,size=3)
    nside=32
    x,y=np.mgrid[0:nside,0:nside]-np.mean(np.arange(nside))
    r=np.sqrt(x**2. + y**2.)
    sigma=3.
    gauss=np.exp(-0.5*(-r/sigma)**2.)
    kern=np.dstack((gauss,gauss,gauss))*blue
    pl.clf(); pl.imshow(np.array(kern,dtype=np.uint8))

#def viewimage(img):
               
               


