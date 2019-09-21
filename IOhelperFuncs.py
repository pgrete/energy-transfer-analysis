import yt
import numpy as np
from mpi4py import MPI
from mpi4py_fft import newDistArray
import FFTHelperFuncs
import sys
import h5py

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_fields(args):
    """
    Read all fields of a simulation snapshot

    args : forwarded command line arguments from the main script
    """
    # data dictionary
    fields = {
        'B' : None,
        'Acc' : None,
        'P' : None,
    }
    pressField = None
    magFields = None
    accFields = None

    if args['data_type'] == 'Enzo':
        rhoField = "Density"
        velFields = ["x-velocity","y-velocity","z-velocity"]
        if args['b']:
            magFields = ["Bx","By","Bz"]
        if args['forced']:
            accFields = ['x-acceleration','y-acceleration','z-acceleration']

        fields  = readAllFieldsWithYT(args['data_path'], args['res'],
                                      rhoField, velFields, magFields,
                                      accFields, pressField)

    elif args['data_type'][:8] == 'AthenaPP':
        rhoField = ('athena_pp', 'rho')
        velFields = [('athena_pp', 'vel1'), ('athena_pp', 'vel2'), ('athena_pp', 'vel3')]
        if args['b']:
            magFields = [('athena_pp', 'Bcc1'), ('athena_pp', 'Bcc2'), ('athena_pp', 'Bcc3')]
        if args['forced']:
            accFields = [('athena_pp', 'acceleration_x'),
                         ('athena_pp', 'acceleration_y'),
                         ('athena_pp', 'acceleration_z')]

        if args['eos'] == 'adiabatic':
            pressField = ('athena_pp', 'press')

        if 'HDF' in args['data_type']:
            readAllFieldsWithHDF(fields,'./Turb.prim.' + args['data_path'], args['res'],
                                rhoField, velFields, magFields,
                                None, pressField,'F',use_athena_hdf=True)
            readAllFieldsWithHDF(fields,'./Turb.acc.' + args['data_path'], args['res'],
                                None, None, None,
                                accFields, None,'F',use_athena_hdf=True)
        else:
            readAllFieldsWithYT(fields,'./Turb.prim.' + args['data_path'], args['res'],
                                rhoField, velFields, magFields,
                                None, pressField)
            readAllFieldsWithYT(fields,'./Turb.acc.' + args['data_path'], args['res'],
                                None, None, None,
                                accFields, None)



    elif args['data_type'] == 'AthenaHDFC':
        rhoField = 'density'
        velFields = ['velocity_x', 'velocity_y', 'velocity_z']
        if args['b']:
            magFields = ['cell_centered_B_x', 'cell_centered_B_y', 'cell_centered_B_z']
        if args['forced']:
            accFields = ['acceleration_x', 'acceleration_y', 'acceleration_z']

        if args['eos'] == 'adiabatic':
            pressField = 'pressure'

        order = 'C'

        fields  = readAllFieldsWithHDF(args['data_path'], args['res'],
                                       rhoField, velFields, magFields,
                                       accFields, pressField,order)

    elif args['data_type'] == 'Athena':
        rhoField = 'density'
        velFields = ['velocity_x', 'velocity_y', 'velocity_z']
        if args['b']:
            magFields = ['cell_centered_B_x', 'cell_centered_B_y', 'cell_centered_B_z']
        if args['forced']:
            accFields = ['acceleration_x', 'acceleration_y', 'acceleration_z']

        if args['eos'] == 'adiabatic':
            pressField = 'pressure'

        order = 'C'

        fields  = readAllFieldsWithYT(args['data_path'], args['res'],
                                      rhoField, velFields, magFields,
                                      accFields, pressField)

    else:
        raise SystemExit('Unknown data type: ', data_type)

    return fields

def readAllFieldsWithYT(fields,loadPath,Res,
    rhoField,velFields,magFields,accFields,pressField=None):
    """
    Reads all fields using the yt frontend. Data is read in parallel.

    """
    raise SystemExit(
        'Reading data with yt not adapted for pencil demp yet!')
	
    if Res % size != 0:
        raise SystemExit(
            'Data cannot be split evenly among processes. ' +
            'Abort (for now) - fix me!')

    FinalShape = (Res//size,Res,Res)   

    ds = yt.load(loadPath)
    dims = ds.domain_dimensions
    left_edge = ds.domain_left_edge
    right_edge = ds.domain_right_edge

    dims[0] /= size

    start_pos = left_edge
    start_pos[0] += rank * (right_edge[0] - left_edge[0])/np.float(size)
    if rank == 0:
        print("Loading "+ loadPath)
        print("Chunk dimensions = ", FinalShape)
        print("WARNING: remember assuming domain of L = 1")


    ad = ds.h.covering_grid(level=0, left_edge=start_pos,dims=dims)

    if rhoField is not None:
        fields['rho'] = ad[rhoField].d

    if pressField is not None:
        fields['P'] = ad[pressField].d
#TODO FIXME for new layout
#    else:
#        if rank == 0:
#            print("WARNING: assuming isothermal EOS with c_s = 1, i.e. P = rho")
#        P = rho
    
    if velFields is not None:
        U = np.zeros((3,) + FinalShape,dtype=np.float64)
        U[0] = ad[velFields[0]].d
        U[1] = ad[velFields[1]].d
        U[2] = ad[velFields[2]].d
        fields['U'] = U

    if magFields is not None:
        B = np.zeros((3,) + FinalShape,dtype=np.float64)  
        B[0] = ad[magFields[0]].d
        B[1] = ad[magFields[1]].d
        B[2] = ad[magFields[2]].d
        fields['B'] = B

    if accFields is not None:
        Acc = np.zeros((3,) + FinalShape,dtype=np.float64)  
        Acc[0] = ad[accFields[0]].d
        Acc[1] = ad[accFields[1]].d
        Acc[2] = ad[accFields[2]].d
        fields['Acc'] = Acc


def readOneFieldWithHDF(loadPath,FieldName,Res,order):

    if order == 'F':
        Filename = loadPath + '/' + FieldName + '-' + str(Res) + '.hdf5'

        if rank == 0:

            h5Data = h5py.File(Filename, 'r')[FieldName]
        
            tmp = np.float64(h5Data[0,:,:,:]).T.reshape((size,int(Res/size),Res,Res))

            data = comm.scatter(tmp)

        else:
            data = comm.scatter(None)

    elif order == 'C':
        Filename = loadPath + '/' + FieldName + '-' + str(Res) + '-C.hdf5'
        
        chunkSize = Res/size
        startIdx = int(rank * chunkSize)
        endIdx = int((rank + 1) * chunkSize)
        if endIdx == Res:                
            endIdx = None
        
        h5Data = h5py.File(Filename, 'r')[FieldName]
        data = np.float64(h5Data[0,startIdx:endIdx,:,:])

    if rank == 0:
        print("[%03d] done reading %s" % (rank,FieldName))

    return np.ascontiguousarray(data)

def readOneFieldWithAthenaPPHDF(loadPath,FieldName,Res,order):
    """
    reading (K-)Athena++ HDF data dumps
    """

    # stripping the yt field type
    FieldName = FieldName[1]

    tmp = np.zeros(FFTHelperFuncs.local_shape, dtype=np.float64)
    loc_slc = tmp.shape

    n_proc = np.array(FFTHelperFuncs.FFT.global_shape(), dtype=int) // loc_slc

    if h5py.h5.get_config().mpi:
        h5py_kwargs = {
            'driver' : 'mpio',
            'comm' : comm,
            }
    else:
        h5py_kwargs = {}

    with h5py.File(loadPath,'r', **h5py_kwargs) as f:
        mb_size = f.attrs['MeshBlockSize']
        rg_size = f.attrs['RootGridSize']

        field_idx = None
        for i in range(f.attrs['NumVariables'][0]):
            if f.attrs['VariableNames'][i] == np.array(FieldName,dtype=bytes):
                field_idx = i
                break
        if field_idx is None:
            raise SystemExit(
                'Error: Cannot find field "%s" in dataset' % FieldName
                )

        if (loc_slc % mb_size != 0).any():
            raise SystemExit(
                'Error: local data size  ', loc_slc,
                'cannot be matched to meshblock size of ', mb_size)

        n_mb = rg_size // mb_size # number of meshblocks in each dimension
        gid_x_s = rank // n_proc[1] * tmp.shape[0] # global x start index
        gid_x_e = rank // n_proc[1] * tmp.shape[0] + tmp.shape[0] # global x end index
        gid_y_s = rank % n_proc[1] * tmp.shape[1] # global y start index
        gid_y_e = rank % n_proc[1] * tmp.shape[1] + tmp.shape[1] # global y end index

        log_loc_all = np.copy(f['LogicalLocations']) # all logical meshblock locations

        for i, loc in enumerate(log_loc_all):
            gid_mb = loc * mb_size # index of meshblock
            # make sure meshblock belong to this MPI proc
            if (gid_mb[0] < gid_x_s or
                gid_mb[0] >= gid_x_e or
                gid_mb[1] < gid_y_s or
                gid_mb[1] >= gid_y_e):
                continue

            try:
                data = f['prim'][field_idx,i,:,:,:] # actual meshblock data
            except KeyError:
                try:
                    data = f['hydro'][field_idx,i,:,:,:] # actual meshblock data
                except KeyError:
                    raise SystemExit(
                        'Can neither find prim nor hydro field in dataset'
                        )

            # currently these are global loc, need local loc
            tmp[loc[0]*mb_size[0] - gid_x_s : (loc[0]+1)*mb_size[0] - gid_x_s,
                loc[1]*mb_size[1] - gid_y_s : (loc[1]+1)*mb_size[1] - gid_y_s,
                loc[2]*mb_size[2] : (loc[2]+1)*mb_size[2]] = data.T

    return np.ascontiguousarray(np.float64(tmp))

def readAllFieldsWithHDF(fields,loadPath,Res,
    rhoField,velFields,magFields,accFields,pField,order,use_athena_hdf=False):
    """
    Reads all fields using the HDF5. Data is read in parallel.

    """
	
    if Res % size != 0:
        print("Data cannot be split evenly among processes. Abort (for now) - fix me!")
        sys.exit(1)

    if order is not "C" and order is not "F":
        print("For safety reasons you have to specify the order (row or column major) for your data.")
        sys.exit(1)

    FinalShape = FFTHelperFuncs.local_shape

    if use_athena_hdf:
        readOneFieldWithX = readOneFieldWithAthenaPPHDF
    else:
        readOneFieldWithX = readOneFieldWithHDF

    if rhoField is not None:
        fields['rho'] = readOneFieldWithX(loadPath,rhoField,Res,order)

    if velFields is not None:
        U = np.zeros((3,) + FinalShape,dtype=np.float64)
        U[0] = readOneFieldWithX(loadPath,velFields[0],Res,order)
        U[1] = readOneFieldWithX(loadPath,velFields[1],Res,order)
        U[2] = readOneFieldWithX(loadPath,velFields[2],Res,order)
        fields['U'] = U

    if magFields is not None:
        B = np.zeros((3,) + FinalShape,dtype=np.float64)  
        B[0] = readOneFieldWithX(loadPath,magFields[0],Res,order)
        B[1] = readOneFieldWithX(loadPath,magFields[1],Res,order)
        B[2] = readOneFieldWithX(loadPath,magFields[2],Res,order)
        fields['B'] = B

    if accFields is not None:
        Acc = np.zeros((3,) + FinalShape,dtype=np.float64)  
        Acc[0] = readOneFieldWithX(loadPath,accFields[0],Res,order)
        Acc[1] = readOneFieldWithX(loadPath,accFields[1],Res,order)
        Acc[2] = readOneFieldWithX(loadPath,accFields[2],Res,order)
        fields['Acc'] = Acc

    if pField is not None:
        fields['P'] = readOneFieldWithX(loadPath,pField,Res,order)
#TODO FIXME for new layout
#    else:
#        # CAREFUL assuming isothermal EOS here with c_s = 1 -> P = rho in code units
#        if rank == 0:
#            print("WARNING: remember assuming isothermal EOS with c_s = 1, i.e. P = rho")
#        P = rho
