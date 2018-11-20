import glob
from keras.models import load_model
from quasarnet.models import custom_loss
from quasarnet.io import read_sdrq, read_data, read_desi_truth
from scipy.interpolate import interp1d
from scipy import zeros, arange
from quasarnet.io import wave
from quasarnet.utils import absorber_IGM
import fitsio
import argparse
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--model',type=str,required=True)
parser.add_argument('--data',type=str,required=True)
parser.add_argument('--data-training',type=str,required=True)
parser.add_argument('--spall',type=str,required=False)
parser.add_argument('--out',type=str,required=True)

args=parser.parse_args()

h=fitsio.FITS(args.data)
tids=h[1]['TARGETID'][:]
X = h[0].read()
m = np.average(X[:,:443],axis=1,weights=X[:,443:])
s = np.average((X[:,:443]-m[:,None])**2,axis=1,weights=X[:,443:])
s = np.sqrt(s)
X = (X[:,:443]-m[:,None])/s[:,None]

m=args.model
print('loading model {}'.format(m))
model=load_model(m, custom_objects={'custom_loss':custom_loss})

print('predicting...')
aux=model.predict(X[:,:,None])
print('prediction done')
nboxes = aux[0].shape[1]//2
lines=['LYA', 'CIV(1548)', 'CIII(1909)', 'MgII(2796)' ,'Hbeta', 'Halpha']
nlines = len(lines)
lines_bal=['CIV(1548)']
nlines_bal = len(lines_bal)

nspec=X.shape[0]
z_line=zeros((nlines, nspec))
p_line=zeros((nlines, nspec))
i_to_wave = interp1d(arange(len(wave)), wave, bounds_error=False, fill_value='extrapolate')

for il in range(len(lines)):
    l=absorber_IGM[lines[il]]
    j = aux[il][:,:13].argmax(axis=1)
    offset  = aux[il][arange(nspec, dtype=int), nboxes+j]
    z_line[il]=i_to_wave((j+offset)*X.shape[1]*1./nboxes)/l-1
    p_line[il]=aux[il][:,:13].max(axis=1)

z_line_bal=zeros((nlines_bal, nspec))
p_line_bal=zeros((nlines_bal, nspec))
for il in range(len(lines_bal)):
    l=absorber_IGM[lines_bal[il]]
    j = aux[nlines+il][:,:13].argmax(axis=1)
    offset  = aux[il+nlines][arange(nspec, dtype=int), nboxes+j]
    z_line_bal[il]=i_to_wave((j+offset)*X.shape[1]*1./nboxes)/l-1
    p_line_bal[il]=aux[il+nlines][:,:13].max(axis=1)

zbest=z_line[p_line.argmax(axis=0),arange(nspec)]
zbest=np.array(zbest)

wqso = (p_line>0.4).sum(axis=0)>0

h_train=fitsio.FITS(args.data_training)

in_train = np.in1d(tids,h_train[1]['TARGETID'][:])
if args.spall is not None:
    plate=h[1]['PLATE'][:]
    mjd = h[1]['MJD'][:]
    fid = h[1]['FIBERID'][:]

    spall = fitsio.FITS(args.spall)
    spall_dict = {t:(p,m,f) for t,p,m,f in zip(spall[1]['THING_ID'][:],
        spall[1]['PLATE'][:],spall[1]['MJD'][:],spall[1]['FIBERID'][:])}

    spall.close()
    pmf_train=[spall_dict[t] for t in h_train[1]['TARGETID'][:]]
    pmf_train = ['{}-{}-{}'.format(p,m,f) for p,m,f in pmf_train]
    pmf = ['{}-{}-{}'.format(p,m,f) for p,m,f in zip(plate, mjd, fid)]

    in_train = np.in1d(pmf,pmf_train)

hout=fitsio.FITS(args.out,'rw',clobber=True)

cols = [tids]
names = ['THING_ID']
if args.spall is not None:
    cols += [plate,mjd,fid]
    names += ['PLATE','MJD','FIBERID']
    
cols += [zbest, wqso.astype(int), in_train.astype(int)]
names += ['ZBEST','IS_QSO','IN_TRAIN']
cols.append(p_line.T)
names.append('C_LINES')
cols.append(z_line.T)
names.append('Z_LINES')
header=[{'name':'LINE_{}'.format(il),'value':absorber_IGM[l],'comment':l} for il,l in enumerate(lines)]

cols.append(p_line_bal.T)
names.append('C_LINES_BAL')
cols.append(z_line_bal.T)
names.append('Z_LINES_BAL')
hout.write(cols,names=names,header=header)

hout.close()
