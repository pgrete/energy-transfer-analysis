import yt
import numpy as np
import time
import h5py as h5
import sys



Type = sys.argv[1]
Id  = sys.argv[2]
RES = int(sys.argv[3])


yt.enable_parallelism()
start=time.time()

if Type[:6] == "Athena":
    ds = yt.load("id0/Turb." + Id + ".vtk")
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

all_data = ds.covering_grid(level=0, left_edge=[0,0.0,0.0],
    dims=ds.domain_dimensions)

#if yt.is_root():
for field in yt.parallel_objects(Fields, -1):

    # transpose here, so that data get stored column major later on (as read by Enzo)
    origField = np.float32(all_data[field].T)


    hdf5File = h5.File("%s/%s-%i.hdf5"%(DirId,field,RES), 'w')

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
    
    hdf5File = h5.File("%s/%s-%i.hdf5"%(DirId,field,int(RES/2)), 'w')

    dataSet = hdf5File.create_dataset(field, data=destField.reshape((1,int(RES/2),int(RES/2),int(RES/2))))
    
    dataSet.attrs["Component_Rank"] = np.array((1), dtype=np.int64)
    dataSet.attrs["Component_Size"] = np.array(int((RES/2)**3), dtype=np.int64)
    dataSet.attrs["Dimensions"] = np.array((int(RES/2), int(RES/2), int(RES/2)), dtype=np.int64)
    dataSet.attrs["Rank"] = np.array((3), dtype=np.int64)
    
    hdf5File.close()


print("total execution time (%.2f sec)" % (time.time() - start))

