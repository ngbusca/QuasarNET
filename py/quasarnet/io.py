from __future__ import print_function

from os.path import dirname

import numpy as np
from numpy import random
import fitsio
from random import randint
import glob
from os.path import dirname


def read_sdrq(sdrq):
    '''
    Constructs two dictionaries thing_id: class, z_conf, z, 
            and (plate, mjd, fiberid): thing_id
    input: (str) full path to Superset_DRQ.fits
    output: (list of dictionaries)
    '''

    sdrq = fitsio.FITS(sdrq)
    thid = sdrq[1]["THING_ID"][:]
    class_person = sdrq[1]["CLASS_PERSON"][:]
    z_conf = sdrq[1]["Z_CONF_PERSON"][:]
    z_vi = sdrq[1]["Z_VI"][:]

    plate = sdrq[1]['PLATE'][:]
    mjd = sdrq[1]['MJD'][:]
    fiberid = sdrq[1]['FIBERID'][:]

    sdrq = {t:(c,zc,z) for t,c,zc,z in zip(thid, class_person, z_conf, z_vi)}
    pmf2tid = {(p,m,f):t for p,m,f,t in zip(plate, mjd, fiberid, thid)}

    return sdrq, pmf2tid

def read_data(fi, sdrq, z_lim=2.1):
    h=fitsio.FITS(fi)
    tids = h[1]["TARGETID"][:].astype(int)

    ## remove thing_id == -1 or not in sdrq
    w = (tids != -1) & (np.in1d(tids, list(sdrq.keys())))
    tids = tids[w]
    Xtmp = h[0].read()
    Xtmp = Xtmp[w]
    
    ## remove zconf == 0 (not inspected)
    observed = [sdrq[t][:2]!=(0,0) for t in tids]
    observed = np.array(observed, dtype=bool)
    tids = tids[observed]
    Xtmp = Xtmp[observed]

    assert Xtmp.shape[0]==tids.shape[0]
    print("INFO: found {} spectra".format(len(tids)))

    mdata = np.average(Xtmp[:,:443], weights = Xtmp[:,443:], axis=1)
    sdata = np.average((Xtmp[:,:443]-mdata[:,None])**2, weights = Xtmp[:,443:], axis=1)
    sdata=np.sqrt(sdata)
    Xtmp=Xtmp[:,:443]-mdata[:,None]
    Xtmp/=sdata[:,None]
    X = np.zeros(Xtmp.shape)
    X[:] = Xtmp
    del Xtmp


    ## fill redshifts
    z = np.zeros(X.shape[0])
    z[:] = [sdrq[t][2] for t in tids]

    ## fill bal 
    bal = np.zeros(X.shape[0])
    bal[:] = [sdrq[t][:2]==(30, 3) for t in tids]
    
    ## fill classes
    ## classes: 0 = STAR, 1=GALAXY, 2=QSO_LZ, 3=QSO_HZ, 4=BAD (zconf != 3)
    nclasses = 5
    sdrq_class = np.array([sdrq[t][0] for t in tids])
    z_conf = np.array([sdrq[t][1] for t in tids])

    Y = np.zeros((X.shape[0],nclasses))
    ## STAR
    w = (sdrq_class==1) & (z_conf==3)
    Y[w,0] = 1

    ## GALAXY
    w = (sdrq_class==4) & (z_conf==3)
    Y[w,1] = 1

    ## QSO_LZ
    w = ((sdrq_class==3) | (sdrq_class==30)) & (z<z_lim) & (z_conf==3)
    Y[w,2] = 1

    ## QSO_HZ
    w = ((sdrq_class==3) | (sdrq_class==30)) & (z>=z_lim) & (z_conf==3)
    Y[w,3] = 1

    ## BAD
    w = z_conf != 3
    Y[w,4] = 1

    ## check that all spectra have exactly one classification
    assert (Y.sum(axis=1).min()==1) and (Y.sum(axis=1).max()==1)

    return tids,X,Y,z,bal

def read_desi_truth(fin):
    h=fitsio.FITS(fin)
    truth = {}
    for t,c,z in zip(h[1]["TARGETID"][:], h[1]["TRUESPECTYPE"][:], h[1]["TRUEZ"][:]):
        c = c.strip()
        if c==b"QSO":
            c=3
        elif c==b"GALAXY":
            c=4
        elif c==b"STAR":
            c=1
        assert isinstance(c,int)
        truth[t] = (c,3,z)
    return truth


llmin = np.log10(3600)
llmax = np.log10(10000)
dll = 1e-3

nbins = int((llmax-llmin)/dll)
nmasked_max = nbins
wave = 10**(llmin + np.arange(nbins)*dll)

def read_spcframe(b_spcframe,r_spcframe):
    data = []
    fids = []

    hb = fitsio.FITS(b_spcframe)
    hr = fitsio.FITS(r_spcframe)
    target_bits = hb[5]["BOSS_TARGET1"][:]
    wqso = np.zeros(len(target_bits),dtype=bool)
    mask = [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44]
    for i in mask:
        wqso = wqso | (target_bits & 2**i)
    ## SEQUELS
    try:
        mask = [10, 11 ,12 ,13, 14, 15, 16, 17, 18]
        target_bits = h[5]["EBOSS_TARGET0"][:]
        for i in mask:
            wqso = wqso | (target_bits & 2**i)
    except:
        pass

    ## EBOSS
    try:
        mask = [10, 11 ,12 ,13, 14, 15, 16, 17, 18]
        target_bits = h[5]["EBOSS_TARGET1"][:]
        for i in mask:
            wqso = wqso | (target_bits & 2**i)
    except:
        pass
    wqso = wqso>0
    print("INFO: found {} quasars in file {}".format(wqso.sum(),b_spcframe))

    plate = hb[0].read_header()["PLATEID"]
    fid = hb[5]["FIBERID"][:]
    fl = np.hstack((hb[0].read(),hr[0].read()))
    iv = np.hstack((hb[1].read()*(hb[2].read()==0),hr[1].read()*(hr[2].read()==0)))
    ll = np.hstack((hb[3].read(),hr[3].read()))

    fid = fid[wqso]
    fl = fl[wqso,:]
    iv = iv[wqso,:]
    ll = ll[wqso,:]

    for i in range(fl.shape[0]):
        fl_aux = np.zeros(nbins)
        iv_aux = np.zeros(nbins)
        bins = ((ll[i]-llmin)/dll).astype(int)
        wbin = (bins>=0) & (bins<nbins) & (iv[i]>0)
        bins=bins[wbin]
        c = np.bincount(bins,weights=fl[i,wbin]*iv[i,wbin])
        fl_aux[:len(c)]=+c
        c = np.bincount(bins,weights=iv[i,wbin])
        iv_aux[:len(c)]=+c
        nmasked = (iv_aux==0).sum()
        if nmasked >= nmasked_max :
            print("INFO: skipping specrum {} with too many masked pixels {}".format(fid[i],nmasked))
            continue
        data.append(np.hstack((fl_aux,iv_aux)))
        fids.append(fid[i])

        assert ~np.isnan(fl_aux,iv_aux).any()

    if len(data)==0:
        return

    data = np.vstack(data)
    assert ~np.isnan(data).any()
    ## now normalize coadded fluxes
    norm = data[:,nbins:]*1.
    w = norm==0
    norm[w] = 1.
    data[:,:nbins]/=norm

    assert ~np.isnan(data).any()

    return fids, data

def read_spall(spall):
    spall = fitsio.FITS(spall)
    plate=spall[1]["PLATE"][:]
    mjd = spall[1]["MJD"][:]
    fid = spall[1]["FIBERID"][:]
    tid = spall[1]["THING_ID"][:].astype(int)
    specprim=spall[1]["SPECPRIMARY"][:]

    pmf2tid = {(p,m,f):t for p,m,f,t,s in zip(plate,mjd,fid,tid,specprim)}
    spall.close()
    return pmf2tid

def read_exposures(plates,pmf2tid,nplates=None, random_exp=False):

    '''
    Given a list of plates, returns the thing_id list and the 
        rebinned fluxes for all the exposures in the plates

    input:
        -- plates: list of str. List of paths to the spPlate files
        -- pmf2tid: dictionary containing (plate, mjd, fiber): thing_id
        -- nplates: use only the first nplates in the list
        -- random_exp: read only one random exposure from all the available
            exposures
    output: thid, data
        -- thid: list of thing ids of length equal the the number of exposures
        -- data: numpy array of float of shape (nexps, nbins)
    '''

    data = []
    read_plates = 0
    tids = []

    plate_mjd_in_pmf2tid = np.empty(len(pmf2tid), dtype=object)
    print('calculating plates-mjd combos')
    plate_mjd_in_pmf2tid[:] =[(k[0], k[1]) for k in pmf2tid.keys()]
    print('uniq-ing')
    plate_mjd_in_pmf2tid = list(np.unique(plate_mjd_in_pmf2tid))
    print('done')

    if nplates is not None:
        plates = plates[:nplates]
    for p in plates:
        h=fitsio.FITS(p)
        head = h[0].read_header()
        plateid = head['PLATEID']
        m = head['MJD']
        if (plateid,m) not in plate_mjd_in_pmf2tid:
            print('{} {} not in list'.format(plateid,m))
            continue


        exps = []
        ## read b,r exposures
        try:
            nexp_b = head["NEXP_B1"]+head["NEXP_B2"]
        except:
            continue
        if nexp_b>99:
            nexp_b=99
        for exp in range(nexp_b):
            str_exp = str(exp+1)
            if exp<9:
                str_exp = '0'+str_exp
            exp_b = head["EXPID{}".format(str_exp)][:11]
            exp_r = exp_b.replace("b", "r")
            exps.append((exp_b, exp_r))

        exps_spectro_1 = [e for e in exps if 'b1' in e[0]]
        exps_spectro_2 = [e for e in exps if 'b2' in e[0]]
        if random_exp:
            irand1 = randint(0,len(exps_spectro_1)-1)
            irand2 = randint(0,len(exps_spectro_2)-1)
            exps = [exps_spectro_1[irand1], exps_spectro_2[irand2]]
         
        for exp_b, exp_r in exps:
            spcframe_b = dirname(p)+"/spCFrame-{}.fits".format(exp_b)
            spcframe_r = dirname(p)+"/spCFrame-{}.fits".format(exp_r)
            res = read_spcframe(spcframe_b, spcframe_r)
            if res is not None:
                plate_fid, plate_data = res
                data.append(plate_data)
                tids = tids + [pmf2tid[(plateid,m,f)] for f in plate_fid]

        if nplates is not None:
            if len(data)//2==nplates:
                break

    data = np.vstack(data)

    return tids, data

def export(fout,tids,data):
    h = fitsio.FITS(fout,"rw",clobber=True)
    h.write(data,extname="DATA")
    tids = np.array(tids)
    h.write([tids],names=["TARGETID"],extname="METADATA")
    h.close()


def read_desi_spectra(fin):
    from desitarget import desi_mask
    h=fitsio.FITS(fin)
    nbins = int((llmax-llmin)/dll)
    wqso = h[1]['DESI_TARGET'][:] & desi_mask.mask('QSO')
    wqso = wqso>0
    tids = h[1]["TARGETID"][:][wqso]
    utids = np.unique(tids)

    nspec = len(utids)
    fl = np.zeros((nspec, nbins))
    iv = np.zeros((nspec, nbins))
    if nspec == 0: return None
    for band in ["B", "R", "Z"]:
        wave = h["{}_WAVELENGTH".format(band)].read()
        w = (np.log10(wave)>llmin) & (np.log10(wave)<llmax)
        wave = wave[w]
        bins = np.floor((np.log10(wave)-llmin)/dll).astype(int)
        fl_aux = h["{}_FLUX".format(band)].read()[:,w]
        iv_aux = h["{}_IVAR".format(band)].read()[:,w]
        fl_aux = fl_aux[wqso]
        iv_aux = iv_aux[wqso]
        ivfl_aux = fl_aux*iv_aux
        for i,t in enumerate(tids):
            j = np.argwhere(utids==t)[0]
            c = np.bincount(bins, weights=ivfl_aux[i])
            fl[j,:len(c)] += c
            c = np.bincount(bins, weights = iv_aux[i])
            iv[j,:len(c)]+=c

    w = iv>0
    fl[w]/=iv[w]
    fl = np.hstack((fl,iv))

    print("INFO: founds {} qso targets".format(wqso.sum()))
    return utids, fl


def read_spplate(fin, fibers):

    '''
    reads data from spplates
    '''

    h=fitsio.FITS(fin)
    head = h[0].read_header()
    c0 = head["COEFF0"]
    c1 = head["COEFF1"]
    p = head["PLATEID"]
    m = head["MJD"]
    
    fids = h[5]["FIBERID"][:]
    wqso = np.in1d(fids, fibers)
    fids=fids[wqso]

    nspec = len(fibers)
    nbins = int((llmax-llmin)/dll)
    fl = np.zeros((nspec, nbins)) 
    iv = np.zeros((nspec, nbins))
    nbins = fl.shape[1]

    fl_aux = h[0].read()[wqso,:]
    iv_aux = h[1].read()[wqso,:]*(h[2].read()[wqso]==0)
    wave = 10**(c0 + c1*np.arange(fl_aux.shape[1]))
    bins = np.floor((np.log10(wave)-llmin)/dll).astype(int)
    w = (bins>=0) & (bins<nbins)
    bins = bins[w]

    fl_aux=fl_aux[:,w]
    iv_aux=iv_aux[:,w]
    for i in range(nspec):
        c = np.bincount(bins, weights=fl_aux[i]*iv_aux[i])
        fl[i,:len(c)] += c
        c = np.bincount(bins, weights = iv_aux[i])
        iv[i,:len(c)]+=c

    w = iv>0
    fl[w]/=iv[w]
    fl = np.hstack((fl,iv))
    print(fl.shape)
    wbad = iv==0
    w=wbad.sum(axis=1)>10
    print('INFO: rejecting {} spectra with too many bad pixels'.format(w.sum()))
    if (~w).sum()==0:
        return None
    fl=fl[~w,:]
    return fids[~w],fl

from .utils import absorber_IGM
from scipy.interpolate import interp1d
def box_offset(z, line='LYA', nboxes = 13):
    wave_to_i = interp1d(wave, np.arange(len(wave)), bounds_error=False, 
            fill_value=-1)
    wave_line = (1+z)*absorber_IGM[line]
    pos = wave_to_i(wave_line)/len(wave)*nboxes
    ipos = np.floor(pos).astype(int)

    box = np.zeros((len(z), nboxes))
    offset = np.zeros((len(z), nboxes))

    w = ipos>=0
    box[w, ipos[w]] = 1
    offset[w, ipos[w]] = (pos-ipos)[w]
    weights = np.ones(len(z))
    weights[~w]=0

    return box, offset, weights
