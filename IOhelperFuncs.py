import yt
import numpy as np
from mpi4py import MPI
import sys
import h5py
try:
    import athena_read
    missing_athena_read = False
except ImportError:
    missing_athena_read = True


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
            if missing_athena_read:
                raise SystemExit(
                    'Asking to read using athena_read (through AthenaPPHDF), but '
                    'athena_read import was unsuccessful.\n' +
                    'Make sure the /path/to/(k)athena/vis/python is in your PYTHONPATH'
                    )
            readAllFieldsWithHDF(fields,'./Turb.prim.' + args['data_path'], args['res'],
                                rhoField, velFields, magFields,
                                None, pressField,'F',use_athena_read=True)
            readAllFieldsWithHDF(fields,'./Turb.acc.' + args['data_path'], args['res'],
                                None, None, None,
                                accFields, None,'F',use_athena_read=True)
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

def readOneFieldWithAthenaRead(loadPath,FieldName,Res,order):
    """
    reading (K-)Athena++ HDF data dumps (which are kji/Fortan/column major
    and thus require a transpose as all internal structures here are row major
    """

    # stripping the yt field type
    FieldName = FieldName[1]
    chunkSize = Res//size
    startIdx = int(rank * chunkSize)
    endIdx = int((rank + 1) * chunkSize)
    if endIdx == Res:
        endIdx = None

    h5Data = athena_read.athdf(loadPath, dtype=np.float32)[FieldName]
    # the copy is important so that the data is contiguous for the MPI comm
    local_slice_in = np.copy(h5Data[startIdx:endIdx,:,:].T,order='C')
    local_slice_out = np.zeros((chunkSize,Res,Res))

    recv = np.zeros_like(local_slice_in)
    comm.Alltoall(local_slice_in,recv)

    # reordering data
    for i in range(chunkSize):
        for j in range(size):
            end_idx = (j+1)*chunkSize
            if end_idx == Res:
                end_idx = None
            local_slice_out[i,:,j*chunkSize:end_idx] = recv[i+j*chunkSize,:,:]

    return np.ascontiguousarray(np.float64(local_slice_out))

def readAllFieldsWithHDF(fields,loadPath,Res,
    rhoField,velFields,magFields,accFields,pField,order,use_athena_read=False):
    """
    Reads all fields using the HDF5. Data is read in parallel.

    """
	
    if Res % size != 0:
        print("Data cannot be split evenly among processes. Abort (for now) - fix me!")
        sys.exit(1)

    if order is not "C" and order is not "F":
        print("For safety reasons you have to specify the order (row or column major) for your data.")
        sys.exit(1)

    FinalShape = (Res//size,Res,Res)  

    if use_athena_read:
        readOneFieldWithX = readOneFieldWithAthenaRead
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
