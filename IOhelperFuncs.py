#import yt
import numpy as np
from mpi4py import MPI
from mpi4py_fft import newDistArray
import FFTHelperFuncs
import sys
import h5py

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
eos = 'unset'

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
    rhoField = None
    pressField = None
    magFields = None
    accFields = None

    global eos
    eos = args['eos']

    time_start = MPI.Wtime()

    if args['data_type'] == 'Enzo':
        rhoField = "Density"
        velFields = ["x-velocity","y-velocity","z-velocity"]
        if args['b']:
            magFields = ["Bx","By","Bz"]
        if args['forced']:
            accFields = ['x-acceleration','y-acceleration','z-acceleration']
        if args['eos'] == 'adiabatic':
            pressField = 'pressure'

        readAllFieldsWithYT(fields, args['data_path'], args['res'],
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

        if 'HDF' == args['data_type'][-3:]:
            readAllFieldsWithHDF(fields,'./Turb.prim.' + args['data_path'], args['res'],
                                rhoField, velFields, magFields,
                                None, pressField,'F',use_athena_hdf=True)
            readAllFieldsWithHDF(fields,'./Turb.acc.' + args['data_path'], args['res'],
                                None, None, None,
                                accFields, None,'F',use_athena_hdf=True)
        elif 'HDFC' == args['data_type'][-4:]:
            readAllFieldsWithHDF(fields,args['data_path'], args['res'],
                                rhoField, velFields, magFields,
                                accFields, pressField,'C')
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

        readAllFieldsWithHDF(fields,args['data_path'], args['res'],
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

        readAllFieldsWithYT(fields, args['data_path'], args['res'],
                            rhoField, velFields, magFields,
                            accFields, pressField)

    elif args['data_type'] == 'JHTDB':
        velFields = [(None,'vel_0'), (None,'vel_1'), (None,'vel_2')]
        if args['b']:
            magFields = [(None,'B_0'), (None,'B_1'), (None,'B_2')]

        if args['eos'] == 'adiabatic':
            pressField = (None, 'pressure')

        order = 'C'

        readAllFieldsWithHDF(fields,args['data_path'], args['res'],
                             rhoField, velFields, magFields,
                             accFields, pressField,order)

        # allow analysis to run with incompressible data
        fields['rho'] = np.ones(fields['U'][0].shape,dtype=np.float64)


    else:
        raise SystemExit('Unknown data type: ', data_type)

    time_elapsed = MPI.Wtime() - time_start
    time_elapsed = comm.gather(time_elapsed)

    if comm.Get_rank() == 0:
        print("Reading data done in %.3g +/- %.3g" %
            (np.mean(time_elapsed), np.std(time_elapsed)))
        sys.stdout.flush()

    if args['eos'] == 'isothermal':
        fields['P'] = args['cs']**2. * fields['rho']

    return fields

def readAllFieldsWithYT(fields,loadPath,Res,
    rhoField,velFields,magFields,accFields,pressField=None):
    """
    Reads all fields using the yt frontend. Data is read in parallel.

    """
    pencil_shape = FFTHelperFuncs.local_shape
    if (np.array(FFTHelperFuncs.FFT.global_shape(), dtype=int) % pencil_shape != 0).any():
        raise SystemExit(
            'Data cannot be split evenly among processes. ' +
            'Abort (for now) - fix me!')

    ds = yt.load(loadPath)
    left_edge = ds.domain_left_edge
    right_edge = ds.domain_right_edge

    n_proc = np.array(FFTHelperFuncs.FFT.global_shape(), dtype=int) // pencil_shape
    gid_x_s = rank // n_proc[1] * pencil_shape[0] # global x start index
    gid_y_s = rank % n_proc[1] * pencil_shape[1] # global y start index

    start_pos = left_edge
    start_pos[0] += gid_x_s / Res * (right_edge[0] - left_edge[0])
    start_pos[1] += gid_y_s / Res * (right_edge[1] - left_edge[1])
    if rank == 0:
        print("Loading "+ loadPath)
        print("Chunk dimensions = ", pencil_shape)


    ad = ds.h.covering_grid(level=0, left_edge=start_pos,dims=FFTHelperFuncs.local_shape)

    if rhoField is not None:
        fields['rho'] = ad[rhoField].d

    if pressField is not None:
        fields['P'] = ad[pressField].d

    if velFields is not None:
        U = np.zeros((3,) + pencil_shape,dtype=np.float64)
        U[0] = ad[velFields[0]].d
        U[1] = ad[velFields[1]].d
        U[2] = ad[velFields[2]].d
        fields['U'] = U

    if magFields is not None:
        B = np.zeros((3,) + pencil_shape,dtype=np.float64)
        B[0] = ad[magFields[0]].d
        B[1] = ad[magFields[1]].d
        B[2] = ad[magFields[2]].d
        fields['B'] = B

    if accFields is not None:
        Acc = np.zeros((3,) + pencil_shape,dtype=np.float64)
        Acc[0] = ad[accFields[0]].d
        Acc[1] = ad[accFields[1]].d
        Acc[2] = ad[accFields[2]].d
        fields['Acc'] = Acc


def readOneFieldWithHDF(loadPath,FieldName,Res,order):
    pencil_shape = FFTHelperFuncs.local_shape
    n_proc = np.array(FFTHelperFuncs.FFT.global_shape(), dtype=int) // pencil_shape
    gid_x_s = rank // n_proc[1] * pencil_shape[0] # global x start index
    gid_y_s = rank % n_proc[1] * pencil_shape[1] # global y start index

    if order == 'F':
        Filename = loadPath + '/' + FieldName + '-' + str(Res) + '.hdf5'

        if rank == 0:

            h5Data = h5py.File(Filename, 'r')[FieldName]
        
            tmp = np.float64(h5Data[0,:,:,:]).T.reshape((size,int(Res/size),Res,Res))

            data = comm.scatter(tmp)

        else:
            data = comm.scatter(None)

    elif order == 'C':
        if h5py.h5.get_config().mpi:
            h5py_kwargs = {
                'driver' : 'mpio',
                'comm' : comm,
                }
            if rank == 0:
                print("Using HDF5 with MPIIO backend.")
        else:
            h5py_kwargs = {}
            if rank == 0:
                print("Using HDF5 with serial backend.")

# TODO(pgrete): fix this to work with AthenaC data again
        # stripping the yt field type
        FieldName = FieldName[1]
        with h5py.File(loadPath,'r', **h5py_kwargs) as f:
            h5Data = f[FieldName]
            data = np.float64(h5Data[gid_x_s:gid_x_s+pencil_shape[0],
                                     gid_y_s:gid_y_s+pencil_shape[1],
                                     :])

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
        if rank == 0:
            print("Using HDF5 with MPIIO backend.")
    else:
        h5py_kwargs = {}
        if rank == 0:
            print("Using HDF5 with serial backend.")

    with h5py.File(loadPath,'r', **h5py_kwargs) as f:
        mb_size = f.attrs['MeshBlockSize']
        rg_size = f.attrs['RootGridSize']

        if 'rho' == FieldName:
            field_idx = 0
            ds_name = 'prim'
        elif 'press' == FieldName:
            field_idx = 1
            ds_name = 'prim'
        elif 'vel' in FieldName:
            if eos == 'isothermal':
                offset = 0
            elif eos == 'adiabatic':
                offset = 1
            else:
                raise SystemExit('Unknown eos: ', eos)
            field_idx = offset + int(FieldName[-1])
            ds_name = 'prim'
        elif 'Bcc' in FieldName:
            field_idx = int(FieldName[-1]) - 1
            ds_name = 'B'
        elif 'acc' in FieldName:
            # translate from ..._x, _y, _z to index 0, 1, 2
            field_idx = ord(FieldName[-1]) - 120
            ds_name = 'hydro'
        else:
            raise SystemExit(
                'Unknown field: ', FieldName)


        if not ((loc_slc[0] % mb_size[0] == 0 or mb_size[0] % loc_slc[0] == 0) and
                (loc_slc[1] % mb_size[1] == 0 or mb_size[1] % loc_slc[1] == 0)):
            raise SystemExit(
                'Error: local data size  ', loc_slc,
                'cannot be matched to meshblock size of ', mb_size)

        gid_x_s = rank // n_proc[1] * tmp.shape[0] # global x start index
        gid_x_e = rank // n_proc[1] * tmp.shape[0] + tmp.shape[0] # global x end index
        gid_y_s = rank % n_proc[1] * tmp.shape[1] # global y start index
        gid_y_e = rank % n_proc[1] * tmp.shape[1] + tmp.shape[1] # global y end index

        log_loc_all = np.copy(f['LogicalLocations']) # all logical meshblock locations

        for i, loc in enumerate(log_loc_all):
            gid_mb = loc * mb_size # index of meshblock
            # make sure meshblock belong to this MPI proc
            if not ((gid_mb[0] <= gid_x_s < gid_mb[0] + mb_size[0] or
                     gid_x_s <= gid_mb[0] < gid_x_e) and
                    (gid_mb[1] <= gid_y_s < gid_mb[1] + mb_size[1] or
                     gid_y_s <= gid_mb[1] < gid_y_e)):
                continue

            try:
                data = f[ds_name][field_idx,i,:,:,:] # actual meshblock data
            except KeyError:
                raise SystemExit(
                    'Cannot find data in dataset: ', ds_name, field_idx
                    )

            # if local x pencil dim smaller than a meshblock use entire local pencil
            if mb_size[0] >= loc_slc[0]:
                loc_x_s = 0
                loc_x_e = tmp.shape[0]
            else:
                loc_x_s = loc[0]*mb_size[0] - gid_x_s
                loc_x_e = (loc[0]+1)*mb_size[0] - gid_x_s
            sl_x = slice(gid_x_s % mb_size[0],gid_x_s % mb_size[0] + tmp.shape[0])

            # if local y pencil dim smaller than a meshblock use entire local pencil
            if mb_size[1] >= loc_slc[1]:
                loc_y_s = 0
                loc_y_e = tmp.shape[1]
            else:
                loc_y_s = loc[1]*mb_size[1] - gid_y_s
                loc_y_e = (loc[1] + 1)*mb_size[1] - gid_y_s
            sl_y = slice(gid_y_s % mb_size[1],gid_y_s % mb_size[1] + tmp.shape[1])

            tmp[loc_x_s: loc_x_e,
                loc_y_s: loc_y_e,
                loc[2]*mb_size[2] : (loc[2] + 1)*mb_size[2]] = data.T[sl_x,sl_y,:]

    return np.ascontiguousarray(np.float64(tmp))

def readAllFieldsWithHDF(fields,loadPath,Res,
    rhoField,velFields,magFields,accFields,pField,order,use_athena_hdf=False):
    """
    Reads all fields using the HDF5. Data is read in parallel.

    """

    FinalShape = FFTHelperFuncs.local_shape

    if (FinalShape[0] * FFTHelperFuncs.FFT.subcomm[0].Get_size() != Res or
        FinalShape[1] * FFTHelperFuncs.FFT.subcomm[1].Get_size() != Res):
        print("Data cannot be split evenly among processes. Abort (for now) - fix me!")
        sys.exit(1)

    if order is not "C" and order is not "F":
        print("For safety reasons you have to specify the order (row or column major) for your data.")
        sys.exit(1)

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
