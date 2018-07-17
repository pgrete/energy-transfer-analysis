#!/bin/bash


BINTYPE="Log"
#FLUID='hydro'
FLUID='adiabatic-mhd'
#FLUID='mhd'

#RUNTYPE="JustShrink"
#RUNTYPE="FlowAndEnTrans"
RUNTYPE="Stats"


RUNS=(
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.0001--1.0-0.250--1.000" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.0001--1.0-1.000--1.000" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.0001--1.0-4.000--1.000" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.4000-0.5-0.250-0.010" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.4000-0.5-1.000-0.165" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.4000-1.0-0.250-0.010" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.4000-1.0-1.000-0.125" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.6667-0.5-0.250-0.020" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.6667-0.5-1.000-0.250" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.6667-1.0-0.250-0.020" 512
#"Athena" "/nobackup/pgrete/run-stripe1/512-1.6667-1.0-1.000-0.300" 512
"Athena" "/nobackup/pgrete/run-stripe1/1024-1.4000-1.0-1.000-0.125" 1024
"Athena" "/nobackup/pgrete/run-stripe1/1024-1.4000-0.5-1.000-0.165" 1024
"Athena" "/nobackup/pgrete/run-stripe1/1024-1.6667-0.5-1.000-0.250" 1024
)


for I in `seq 1 3 ${#RUNS[@]}`
do
    TYPE=${RUNS[$((I-1))]}
    ROOTDATADIR=${RUNS[$I]}
    RES=${RUNS[$((I+1))]}
    GAMMA=`echo $ROOTDATADIR | awk -F '-' '{print $3}'`

    if [[ $RUNTYPE == "JustShrink" ]]; then

      NUMNODES=1
      
      if [[ $RES == 512 ]]; then
        NODETYPE="has"
        NMPI=24
        WALLTIME="1:0:0"
        FSIZE=513
      elif [[ $RES == 1024 ]]; then
        NODETYPE="bro"
        NMPI=28
        WALLTIME="2:0:0"
        FSIZE=4097
      elif [[ $RES == 2048 ]]; then
        NODETYPE="has"
        NMPI=24
        WALLTIME="8:0:0"
        FSIZE=32769
      else
        echo "unknown RES: $RES"
        exit 1
      fi

    elif [[ $RUNTYPE == "FlowAndEnTrans" ]]; then
      if [[ $RES == 512 ]]; then
        NODETYPE="bro"
        NMPI=24
        WALLTIME="3:00:0"
#        WALLTIME="2:30:0"
        NUMNODES=6
        NPROC=128
      elif [[ $RES == 1024 ]]; then
        NODETYPE="has"
        NMPI=24
        WALLTIME="6:30:0"
        NUMNODES=11
        NPROC=256 
      else
        echo "unknown RES: $RES"
        exit 1
      fi
    elif [[ $RUNTYPE == "Stats" ]]; then
      if [[ $RES == 512 ]]; then
        NODETYPE="bro"
        NMPI=24
        WALLTIME="0:20:0"
#        WALLTIME="2:30:0"
        NUMNODES=1
        NPROC=16
      elif [[ $RES == 1024 ]]; then
        NODETYPE="has"
        NMPI=24
        WALLTIME="2:00:0"
#        WALLTIME="2:30:0"
        NUMNODES=6
        NPROC=128
      else
        echo "unknown RES: $RES"
        exit 1
      fi

    fi

		BNAME=`basename $ROOTDATADIR`

    cd $ROOTDATADIR

#		for DUMP in `seq -w 0 1 0100`; do
		for DUMP in `seq -w 0 5 0100`; do
#		for DUMP in `seq -w 0 3 0500`; do
#		for DUMP in `seq -w 00 1 0050`; do
#		for DUMP in 0050; do
    
		if [[ $TYPE == Athena* ]]; then
    PREFIX=""
		DUMPFILE="id0/Turb.${DUMP}.vtk"
		#DUMPFILE="${DUMP}-stats-${RES}.hdf5"
    elif [[ $TYPE == Enzo* ]]; then
    PREFIX="DD"
		#DUMPFILE=${PREFIX}${DUMP}/data${DUMP}
		DUMPFILE=${PREFIX}${DUMP}/Bz-${RES}.hdf5
    else
    echo "Whoopsie wrong datatype"
    continue
    fi

if [ ! -f $DUMPFILE ]; then
echo $BNAME $DUMP missing dump
continue
fi

#############################################################################
############### Check if justShrink is done
#############################################################################
if [[ $RUNTYPE == "JustShrink" ]]; then
SKIP=1
for fieldName in density velocity_x velocity_y velocity_z cell_centered_B_x cell_centered_B_y cell_centered_B_z acceleration_x acceleration_y acceleration_z; do
  if [[ ! -e ${PREFIX}${DUMP}/${fieldName}-${RES}.hdf5 ]]; then
    SKIP=0
    echo "Doing $BNAME $DUMP for missing ${PREFIX}${DUMP}/${fieldName}-${RES}.hdf5"
    break
  fi
  THISSIZE=$( du -m "${PREFIX}${DUMP}/${fieldName}-${RES}.hdf5" | cut -f 1)
  if [[  ! $THISSIZE -eq $FSIZE ]]; then
    SKIP=0
    echo "Doing $BNAME $DUMP for too small ${PREFIX}${DUMP}/${fieldName}-${RES}.hdf5"
    break
  fi
done
fi

if [[ $SKIP -eq 1 ]]; then
  echo done $BNAME $DUMP 
  continue
fi
#continue

#############################################################################
############### Check if FlowAndEnTrans is done
#############################################################################
if [[ $RUNTYPE == "FlowAndEnTrans" ]]; then
#if [ -f ${DUMP}-All-${BINTYPE}-${RES}.pkl ]; then
if [ -f ${DUMP}-All-Forc-Pres-${BINTYPE}-${RES}.pkl ]; then
#if [ -f ${DUMP}-PSMagEn-${RES}.npy ]; then
#if [ -f ${DUMP}-Forc-Pres-${BINTYPE}-${RES}.pkl ]; then
#echo $BNAME $DUMP done 
continue
fi
fi
#if [ ! -f ${PREFIX}${DUMP}/acceleration_z-${RES}.hdf5 ]; then
#  echo missing $BNAME $DUMP 
#  continue
#fi


echo $BNAME $DUMP doing now
#continue

echo "
#!/bin/bash

#PBS -l select=${NUMNODES}:ncpus=${NMPI}:mpiprocs=${NMPI}:model=$NODETYPE
#PBS -l walltime=${WALLTIME}
#PBS -q normal
#PBS -N ${BNAME}-${DUMP}-${RUNTYPE}

#module load mpi-sgi/mpt
#export PATH=~/src/yt-conda/bin:\$PATH

module load mpi-sgi/mpt gcc/4.9.4 python3/3.5.2
export LD_LIBRARY_PATH=~/.local/lib:\$LD_LIBRARY_PATH
export PATH=~/.local/bin:\$PATH
export PYTHONPATH=/u/pgrete/src/energy-transfer-analysis:\$PYTHONPATH

ulimit -n 4096

cd $ROOTDATADIR

" > tmp.sh

#if [[ $RUNTYPE == "JustShrink" ]]; then
  echo "
mkdir ${DUMP}
lfs setstripe -c 4 ${DUMP}
date
python ~/src/energy-transfer-analysis/scripts/shrink.py Athena ${DUMP} ${RES} ${FLUID} forced C
#python ~/src/energy-transfer-analysis/scripts/delVtk.py Athena ${DUMP} ${RES} ${FLUID} forced C
date
" >> tmp.sh
#fi

if [[ $RUNTYPE == "FlowAndEnTrans" ]]; then
  echo "
date
#mpiexec -np $NPROC python ~/src/energy-transfer-analysis/runFlowQuant.py ${DUMP} ${RES} AthenaHDF ${FLUID} 

date
mpiexec -np $NPROC python ~/src/energy-transfer-analysis/runTransfer.py $DUMP All-Forc-Pres ${RES} AthenaHDF  ${FLUID} $BINTYPE 
date
" >> tmp.sh
fi

if [[ $RUNTYPE == "Stats" ]]; then
  echo "
date
#python ~/src/energy-transfer-analysis/scripts/FtoC.py Athena ${DUMP} ${RES} ${FLUID} forced
#mpiexec -np $NPROC python ~/src/energy-transfer-analysis/runHigherFlowQuant.py ${DUMP} ${RES} AthenaHDFC ${FLUID} forced /home1/pgrete/src/energy-transfer-shared/notebooks/globMinMax-M05.pkl ${GAMMA}
mpiexec -np $NPROC python ~/src/energy-transfer-analysis/runHigherFlowQuant.py ${DUMP} ${RES} AthenaHDFC ${FLUID} forced null ${GAMMA}


date
" >> tmp.sh
fi

qsub tmp.sh

done
done
