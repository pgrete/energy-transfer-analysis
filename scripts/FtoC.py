import numpy as np
import time
import h5py as h5
import sys
import os
import gc


Type = sys.argv[1]
Id  = sys.argv[2]
RES = int(sys.argv[3])
Fluid = sys.argv[4]
Forced = sys.argv[5]
#yt.enable_parallelism()
start=time.time()

if Type[:6] == "Athena":
    DirId = Id
    Fields = ["density","velocity_x","velocity_y","velocity_z",
    ]
    if Forced == 'forced':
        Fields += ["acceleration_x","acceleration_y","acceleration_z"]
    if Fluid == 'mhd':
        Fields += ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]

Sizes = {
    128  : [-1,-1],
    512  : [536873056,67111008],
    1024 : [4294969440,536873056],
    2048 : [34359740512, 4294969440],
}


for field in Fields:

    if os.path.isfile("%s/%s-%i-C.hdf5"%(DirId,field,RES)):
        if int(os.path.getsize("%s/%s-%i-C.hdf5"%(DirId,field,RES))) == Sizes[RES][0]:
            print("%s is done" % field)
            continue

    hdf5FileIn = h5.File("%s/%s-%i.hdf5"%(DirId,field,RES), 'r')

    h5Data = hdf5FileIn[field]

    origField = h5Data[0,:,:,:].T.reshape((1,RES,RES,RES))

    np.save('%s/%s-xy.npy' % (DirId,field),origField[0,:,:,49 % RES])
    np.save('%s/%s-yz.npy' % (DirId,field),origField[0,835 % RES,:,:])
    np.save('%s/%s-xz.npy' % (DirId,field),origField[0,:,1532 % RES,:])
    
    hdf5File = h5.File("%s/%s-%i-C.hdf5"%(DirId,field,RES), 'w')
    dataSet = hdf5File.create_dataset(field, data=origField)
    
    dataSet.attrs["Component_Rank"] = np.array((1), dtype=np.int64)
    dataSet.attrs["Component_Size"] = np.array(((RES)**3), dtype=np.int64)
    dataSet.attrs["Dimensions"] = np.array((RES, RES, RES), dtype=np.int64)
    dataSet.attrs["Rank"] = np.array((3), dtype=np.int64)
    
    hdf5FileIn.close()
    hdf5File.close()
    
    print("%s done after (%.2f sec)" % (field, time.time() - start))

#for field in Fields:

    if int(os.path.getsize("%s/%s-%i-C.hdf5"%(DirId,field,RES))) == Sizes[RES][0]:
        print("Filesize of %s is good" % field)
        if os.path.isfile("%s/%s-%i.hdf5"%(DirId,field,RES)):
            print("Removing %s F order" % field)
            os.remove("%s/%s-%i.hdf5"%(DirId,field,RES))
    else:
        print("Filesize of %s is wrong!!!" % field)
