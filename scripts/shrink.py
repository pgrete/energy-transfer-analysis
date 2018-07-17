import yt
import numpy as np
import time
import h5py as h5
import sys
import os
import gc
import glob


Type = sys.argv[1]
Id  = sys.argv[2]
RES = int(sys.argv[3])
Fluid = sys.argv[4]
Forced = sys.argv[5]
Order = sys.argv[6]
#yt.enable_parallelism()
start=time.time()

if Type[:6] == "Athena":
    ds = yt.load("id0/Turb." + Id + ".vtk")
    DirId = Id
    ytfield = 'athena'
    Fields = ["density","velocity_x","velocity_y","velocity_z",
    ]
    if Forced == 'forced':
        Fields += ["acceleration_x","acceleration_y","acceleration_z"]
    if 'mhd' in Fluid:
        Fields += ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    if 'adiabatic' in Fluid:
        Fields += ['pressure']

elif Type[:4] == "Enzo":
    ds = yt.load("DD" + Id + "/data" + Id)
    DirId = "DD" + Id
    ytfield = 'enzo'
    Fields = ["Density","x-velocity","y-velocity","z-velocity","Bx","By","Bz"]
    if Forced == 'forced':
        Fields += ["x-acceleration","y-acceleration","z-acceleration"]

elif Type[:7] == "AccEnzo":
    ds = yt.load("DD" + Id + "/data" + Id)
    DirId = "DD" + Id
    Fields = ['x-acceleration','y-acceleration','z-acceleration']

elif Type[:2] == "DV":
    ds = yt.load(Id + "/" + Type + ".vtk")
    DirId = Id
    Fields = [Type]

    
else:
    print("Unknown sim Type in param 1")
    sys.exit(1)

Sizes = {
    128  : [-1,-1],
    512  : [536873056,67111008],
    1024 : [4294969440,536873056],
    2048 : [34359740512, 4294969440],
}

#all_data = ds.covering_grid(level=0, left_edge=[0,0.0,0.0],
#    dims=ds.domain_dimensions)

#if yt.is_root():
#for field in yt.parallel_objects(Fields, -1):

if Order == 'F':
    Suffix = ''
elif Order == 'C':
    Suffix = '-C'
else:
    print('Unknown order: %s' % Order)
    sys.exit(1)

for field in Fields:

    skip = True
    if not (os.path.exists("%s/%s-%i%s.hdf5"%(DirId,field,RES,Suffix))):
        skip = False
        print("%s/%s-%i.hdf5 misses "%(DirId,field,RES))
    elif not (int(os.path.getsize("%s/%s-%i%s.hdf5"%(DirId,field,RES,Suffix))) == Sizes[RES][0]):
        skip = False
        print("%s/%s-%i%s.hdf5 wrong size act %d versus %d "%(DirId,field,RES,Suffix,int(os.path.getsize("%s/%s-%i%s.hdf5"%(DirId,field,RES,Suffix))),Sizes[RES][0]))
    
    if not(os.path.exists("%s/%s-%i%s.hdf5"%(DirId,field,int(RES/2),Suffix))):
        skip = False
        print("%s/%s-%i%s.hdf5 misses "%(DirId,field,int(RES/2),Suffix))
    elif not(int(os.path.getsize("%s/%s-%i%s.hdf5"%(DirId,field,int(RES/2),Suffix))) == Sizes[RES][1]):
        skip = False
        print("%s/%s-%i%s.hdf5 wrong size act %d versus %d "%(DirId,field,int(RES/2),Suffix,os.path.getsize("%s/%s-%i%s.hdf5"%(DirId,field,int(RES/2),Suffix)),Sizes[RES][1]))

    sys.stdout.flush()
    
    if skip:
        print("Done with %s. Continue..." % field)
        continue

    all_data = ds.covering_grid(level=0, left_edge=[0,0.0,0.0],
        dims=ds.domain_dimensions, fields = [(ytfield,field)])
    
    origField = all_data[(ytfield,field)].astype("float32")
    del all_data

    # call gargabe collector explicitly as need all available mem immediately
    gc.collect()

    # transpose here, so that data get stored column major later on (as read by Enzo)
    if Order == 'F':
        origField = origField.T


    hdf5File = h5.File("%s/%s-%i%s.hdf5"%(DirId,field,RES,Suffix), 'w')

    dataSet = hdf5File.create_dataset(field, data=origField.reshape((1,RES,RES,RES)))
    
    dataSet.attrs["Component_Rank"] = np.array((1), dtype=np.int64)
    dataSet.attrs["Component_Size"] = np.array(((RES)**3), dtype=np.int64)
    dataSet.attrs["Dimensions"] = np.array((RES, RES, RES), dtype=np.int64)
    dataSet.attrs["Rank"] = np.array((3), dtype=np.int64)
    
    hdf5File.close()
    
    print(origField.dtype)
    destField = np.zeros((int(RES/2),int(RES/2),int(RES/2)),dtype=origField.dtype)

    destField = (origField[::2,::2,::2] + origField[::2,1::2,::2] + 
        origField[::2,::2,1::2] + origField[::2,1::2,1::2] +
        origField[1::2,::2,::2] + origField[1::2,1::2,::2] + 
        origField[1::2,::2,1::2] + origField[1::2,1::2,1::2])/8.
    
    hdf5File = h5.File("%s/%s-%i%s.hdf5"%(DirId,field,int(RES/2),Suffix), 'w')

    dataSet = hdf5File.create_dataset(field, data=destField.reshape((1,int(RES/2),int(RES/2),int(RES/2))))
    
    dataSet.attrs["Component_Rank"] = np.array((1), dtype=np.int64)
    dataSet.attrs["Component_Size"] = np.array(int((RES/2)**3), dtype=np.int64)
    dataSet.attrs["Dimensions"] = np.array((int(RES/2), int(RES/2), int(RES/2)), dtype=np.int64)
    dataSet.attrs["Rank"] = np.array((3), dtype=np.int64)
    
    hdf5File.close()

    del origField, dataSet, destField, hdf5File
    print("%s done after (%.2f sec)" % (field, time.time() - start))


skip = True
for field in Fields:

    if not (os.path.exists("%s/%s-%i%s.hdf5"%(DirId,field,RES,Suffix))):
        skip = False
        print("%s/%s-%i.hdf5 misses "%(DirId,field,RES))
    elif not (int(os.path.getsize("%s/%s-%i%s.hdf5"%(DirId,field,RES,Suffix))) == Sizes[RES][0]):
        skip = False
        print("%s/%s-%i%s.hdf5 wrong size act %d versus %d "%(DirId,field,RES,Suffix,int(os.path.getsize("%s/%s-%i%s.hdf5"%(DirId,field,RES,Suffix))),Sizes[RES][0]))
    
    if not(os.path.exists("%s/%s-%i%s.hdf5"%(DirId,field,int(RES/2),Suffix))):
        skip = False
        print("%s/%s-%i%s.hdf5 misses "%(DirId,field,int(RES/2),Suffix))
    elif not(int(os.path.getsize("%s/%s-%i%s.hdf5"%(DirId,field,int(RES/2),Suffix))) == Sizes[RES][1]):
        skip = False
        print("%s/%s-%i%s.hdf5 wrong size act %d versus %d "%(DirId,field,int(RES/2),Suffix,os.path.getsize("%s/%s-%i%s.hdf5"%(DirId,field,int(RES/2),Suffix)),Sizes[RES][1]))

    sys.stdout.flush()
    
if skip: 
    for File in glob.glob("id*/Turb*%s.vtk" % Id):
        os.remove(File)
    print("Done with %s. deleting vtk ..." % DirId)

print("total execution time (%.2f sec)" % (time.time() - start))
