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


#yt.enable_parallelism()
start=time.time()

if Type[:6] == "Athena":
    DirId = Id
    Fields = ["density","velocity_x","velocity_y","velocity_z",
    "cell_centered_B_x","cell_centered_B_y","cell_centered_B_z",
    "acceleration_x","acceleration_y","acceleration_z",
    ]

elif Type[:4] == "Enzo":
    ds = yt.load("DD" + Id + "/data" + Id)
    DirId = "DD" + Id
    Fields = ["Density","x-velocity","y-velocity","z-velocity","Bx","By","Bz"]

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

#all_data = ds.covering_grid(level=0, left_edge=[0,0.0,0.0],
#    dims=ds.domain_dimensions)

#if yt.is_root():
#for field in yt.parallel_objects(Fields, -1):

Sizes = {
    512  : [536873056,67111008],
    1024 : [4294969440,536873056],
    2048 : [34359740512, 4294969440],
}

skip = True

for field in Fields:

    if not (os.path.exists("%s/%s-%i.hdf5"%(DirId,field,RES))):
        skip = False
        print("%s/%s-%i.hdf5 misses "%(DirId,field,RES))
    elif not (int(os.path.getsize("%s/%s-%i.hdf5"%(DirId,field,RES))) == Sizes[RES][0]):
        skip = False
        print("%s/%s-%i.hdf5 wrong size act %d versus %d "%(DirId,field,RES,int(os.path.getsize("%s/%s-%i.hdf5"%(DirId,field,RES))),Sizes[RES][0]))
    
    if not(os.path.exists("%s/%s-%i.hdf5"%(DirId,field,int(RES/2)))):
        skip = False
        print("%s/%s-%i.hdf5 misses "%(DirId,field,int(RES/2)))
    elif not(int(os.path.getsize("%s/%s-%i.hdf5"%(DirId,field,int(RES/2)))) == Sizes[RES][1]):
        skip = False
        print("%s/%s-%i.hdf5 wrong size act %d versus %d "%(DirId,field,int(RES/2),os.path.getsize("%s/%s-%i.hdf5"%(DirId,field,int(RES/2))),Sizes[RES][1]))

    sys.stdout.flush()
    
if skip: 
    for File in glob.glob("id*/Turb*%s.vtk" % Id):
        os.remove(File)
    print("Done with %s. deleting vtk ..." % DirId)


print("total execution time (%.2f sec)" % (time.time() - start))
