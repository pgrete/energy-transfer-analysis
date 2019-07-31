import argparse
from mpi4py import MPI
from IOhelperFuncs import read_fields 
from EnergyTransfer import EnergyTransfer
from FlowAnalysis import FlowAnalysis
import os
import sys
import pickle

analysis_description = (
    "MPI parallel turbulence simulation analysis"
)
analysis_epilog = (
    "Full documentation of options available at "
    "https://github.com/pgrete/energy-transfer-analysis"
)


parser = argparse.ArgumentParser(description=analysis_description, epilog=analysis_epilog)

parser.add_argument('--res',
                    required=True,
                    type=int,
                    help='set linear resolution of cubic box')

parser.add_argument('--type',
                    required=True,
                    type=str,
                    choices=['transfer','flow','unit-test'],
                    help='set analysis type')

parser.add_argument('--data_type',
                    required=True,
                    type=str,
                    choices=['Enzo', 'AthenaPP', 'AthenaHDFC', 'Athena'],
                    help='set data cube type')

parser.add_argument('--data_path',
                    required=True,
                    type=str,
                    help='set data location')

parser.add_argument('--outfile',
                    type=str,
                    default=None,
                    help='set file to store results')

parser.add_argument('--extrema_file',
                    type=str,
                    help='Pickled Python dict containing extrema for flow analysis')

parser.add_argument('-b',
                    action='store_true',
                    default=False,
                    help='enable magnetic fields')

parser.add_argument('-forced',
                    action='store_true',
                    default=False,
                    help='output is actively forced')

parser.add_argument('--eos',
                    required=True,
                    type=str,
                    choices=['isothermal','adiabatic'],
                    help='set equation of state')

parser.add_argument('--gamma',
                    type=float,
                    default=None,
                    help='set adiabatic gamma index')


parser.add_argument('-approx-isothermal',
                    action='store_true',
                    default=False,
                    help='assume c_s^2 / gamma = p/rho = 1')
                       

parser.add_argument('--terms',
                    type=str,
                    nargs='+',
                    default=None,
                    choices = ['All', 'Int', 'UU', 'BUT', 'BUP', 'UBT', 'UBPb',
                               'BB', 'BUPbb', 'UBPbb', 'SS', 'SU', 'US', 'PU', 'FU'],
                    help='set energy transfer terms to analyze')

parser.add_argument('--binning',
                    default=None,
                    type=str,
                    choices=['log', 'lin', 'test'],
                    help='set binning used in energy transfer analysis')

parser.add_argument('--kernels',
                    default=None,
                    type=str,
                    nargs='+',
                    choices=['Box', 'Sharp', 'Gauss'],
                    help='choose convolution kernel type(s): Box, Sharp, or Gauss')

args = vars(parser.parse_args())

# Check equation of state parameters
if args['eos'] == 'adiabatic' and args['gamma'] == None:
    raise SystemExit('Please set gamma for when using adiabatic EOS')

# Check energy transfer arguments
if args['type'] != 'transfer' and args['terms'] != None:
    raise SystemExit('--terms to analyze set but --type is not transfer')

if args['type'] == 'transfer' and args['terms'] == None:
    raise SystemExit('Asking for energy transfer analysis but no terms chosen')

if args['type'] == 'transfer' and args['binning'] == None:
    raise SystemExit('Asking for energy transfer analysis but no binnig chosen')

# Set mpi vars
comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parse energy transfer arguments
if args['type'] == 'transfer':
    magnetic_terms = ['BB', 'BUT', 'BUP', 'UBT', 'UBPb']
    terms_to_analyze = args['terms']

    if 'All' in terms_to_analyze:
        terms_to_analyze += ['UU']
        if args['b']:
            terms_to_analyze += magnetic_terms
        
        while 'All' in terms_to_analyze:
            terms_to_analyze.remove('All')
    
    if 'Int' in terms_to_analyze:
        terms_to_analyze += ['SS', 'US', 'SU']
        while 'Int' in terms_to_analyze:
            terms_to_analyze.remove('Int')

    terms_to_analyze = list(set(terms_to_analyze))

    for this_term in magnetic_terms:
        if this_term in terms_to_analyze and not args['b']:
            raise SystemExit(
                'Asking for magnetic energy transfer analysis but ' +
                'data is identified as not containing magnetic fields.\n' +
                'Try adding the -b if the simulation contains magnetic fields.'
                )
    if 'FU' in terms_to_analyze and not args['forced']:
        raise SystemExit(
            'Asking for acceleration field energy transfer analysis but ' +
            'data is identified as not containing acceleration fields.\n' +
            'Try adding the -forced if the simulation contains acceleration fields.'
            )


    if args['binning'] == 'lin':
	    bins = np.concatenate((np.linspace(0.5,resolution/2-0.5,resolution/2,
                                           endpoint=True),
                               [float(resolution)/2.*np.sqrt(3)]))
    
    elif args['binning'] == "log":
        resolution_exp = np.log(resolution/8)/np.log(2) * 4 + 1
        bins = np.concatenate(
            (np.array([0.]), 4.* 2** ((np.arange(0,resolution_exp + 1) - 1.) /4.)))
    
    elif args['binning'] == "test":
        bins = [0.5,1.5,2.5,16.0,26.5,28.5,32.0]

    else:
        raise SystemExit('Binning undetermined')


if args['outfile'] is None and args['type'] != 'unit-test':
    raise SystemExit('Outfile required for analysis.')

outfile = args['outfile']
resolution = args['res']
if args['eos'] == 'adiabatic':
    gamma = args['gamma']
else:
    gamma = None

# Load data

# data dictionary
fields = read_fields(args)

# Run energy transfer analysis
if args['type'] == 'transfer':
    
    ET = EnergyTransfer(MPI,resolution,fields,gamma)

    if rank == 0:
        if os.path.isfile(outfile):
            print("Reading previous transfer file")
            if sys.version_info[0] < 3:
                results = pickle.load(open(outfile,"rb"))
            else:
                results = pickle.load(open(outfile,"rb"),encoding='latin1')
        else:
            results = {}
    else:
        results = None
    
    # threoretically k and q binnig could be independent
    k_bins = bins
    q_bins = bins
    
    ET.getTransferWWAnyToAny(results, k_bins, q_bins, terms_to_analyze)
    
    if rank == 0:
        pickle.dump(results,open(outfile,"wb"))    

elif args['type'] == 'flow':
    
    FA = FlowAnalysis(MPI,args,fields)
    FA.run_analysis()

elif args['type'] == 'unit-test':
    
    FA = FlowAnalysis(MPI,args,fields)
    FA.run_test()

else:
    raise SystemExit('Unknown transfer type: ', args['type'])
