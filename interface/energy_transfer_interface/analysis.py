# python tools for analyzing Parthenon simulation data

# TODO: 
# - the internal time units should always be in initial Alfven times.

import yt
yt.funcs.mylog.setLevel("ERROR") # suppress yt warnings 
import glob
import os
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd
import matplotlib
import pickle
import energy_transfer_interface.io_utils as io_utils

## Functions for standalone analysis (not tied to a "simulation" object): 
def get_transfer_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    types = ['BB', 'UU', 'UBTb', "H"]
    transfer_data = {}

    for type_ in types:
        # Extract transfer array and bins:
        try:
            transfer_dict = data['WW'][type_]['AnyToAny']
        except KeyError:
            raise KeyError(f"Transfer type {type_} not found in transfer analysis data.")
        transfer_array, k_labels, _ = io_utils.dict_to_array(transfer_dict)
        if type_ == 'UBTb':
            type_ = 'UBT'  # rename for consistency
        transfer_data[type_] = transfer_array            
    k_edges = np.array([tuple(map(float, s.split('-'))) for s in k_labels])
    k_geom = np.sqrt(k_edges[:, 0] * k_edges[:, 1])

    time = None

    energy_transfer = EnergyTransfer(transfer_data, k_geom, k_edges, parent_simulation=None, time=time)
    return energy_transfer

# Map simple keys to parthenon keys
prim_key_dict = {
    'rho': 'prim_density', 
    'Bx': 'prim_magnetic_field_1',
    'By': 'prim_magnetic_field_2',
    'Bz': 'prim_magnetic_field_3',
    'p': 'prim_pressure',
    'vx': 'prim_velocity_1',
    'vy': 'prim_velocity_2',
    'vz': 'prim_velocity_3',
}

# Map simple keys to latex strings
key_latex_dict = {
    'rho': r'$\rho$',
    'Bx': r'$B_x$',
    'By': r'$B_y$',
    'Bz': r'$B_z$',
    'p': r'$p$',
    'vx': r'$v_x$',
    'vy': r'$v_y$',
    'vz': r'$v_z$',
}

class SimSnap: # wrapper for yt dataset with additional functionality
    
    def __init__(self, filename):
        self.filename = filename
        self.ds = yt.load(filename)
    
    def plot_slice(self, field, axis="z", center="l"):
        # Assuming uniform grid
        ds = self.ds
        slc = yt.SlicePlot(ds, axis, prim_key_dict[field], center='l') 
        slc.set_log(prim_key_dict[field], False)  # linear scale
        slc.show() 
    
# Simple container for spectrum data
class Spectrum:
    def __init__(self, type_, bins, values, parent, binscale='linear', time=None): # Warning: currently only linear bins supported
        self.bins = bins
        self.values = values
        self.type = type_  
        self.parent = parent # parent Simulation object
        if time is None:
            print("Warning: Spectrum created without time information; setting time=0.0")
            self.time = 0.0
        else:
            self.time = time
    
    def plot(self, label=None, color=None, linestyle='-', marker=None, type='values', have_label=True):
        if label is None and have_label:
            label = f't={self.time}'
        if type == 'values':
            plt.loglog(self.bins, self.values, label=label, color=color, linestyle=linestyle, marker=marker)
            plt.ylabel('Power')
        elif type == 'logslope':
            # compute local log-log slope of the spectrum
            slopes = np.zeros_like(self.values)
            for i in range(1, len(self.bins)-1):
                km1, kp1 = self.bins[i-1], self.bins[i+1]
                Pm1, Pp1 = self.values[i-1], self.values[i+1]
                slope = (np.log(Pp1) - np.log(Pm1)) / (np.log(kp1) - np.log(km1))
                slopes[i] = slope
            plt.semilogx(self.bins, slopes, label=label, color=color, linestyle=linestyle, marker=marker)
            plt.ylabel('d log(P) / d log(k)')
        plt.xlabel('k')
        plt.title(f'Power Spectrum: {self.type}')
        plt.legend()

    def get_val(self, k):
        # retrieve value at given k via interpolation
        if k < self.bins[0] or k > self.bins[-1]:
            raise ValueError(f"k={k} out of bounds ({self.bins[0]} to {self.bins[-1]})")
        val = np.interp(k, self.bins, self.values)
        return val
    
    def get_logslope(self, k=None):
        # compute local log-log slope at given k via second-order accurate centered difference
        if k is None:
            # Compute slope for all k 
            slopes = np.zeros_like(self.bins)
            for i in range(1, len(self.bins)-1):
                km1, kp1 = self.bins[i-1], self.bins[i+1]
                Pm1, Pp1 = self.values[i-1], self.values[i+1]
                slope = (np.log(Pp1) - np.log(Pm1)) / (np.log(kp1) - np.log(km1))
                slopes[i] = slope
            return self.bins, slopes

        if k <= self.bins[0] or k >= self.bins[-1]:
            raise ValueError(f"k={k} out of bounds ({self.bins[0]} to {self.bins[-1]})")

        idx = np.searchsorted(self.bins, k)

        # get i-1 and i+1 samples
        km1, kp1 = self.bins[idx-1], self.bins[idx+1]
        Pm1, Pp1 = self.values[idx-1], self.values[idx+1]

        # compute slope = d log(P) / d log(k)
        slope = (np.log(Pp1) - np.log(Pm1)) / (np.log(kp1) - np.log(km1))
        return slope

    @property
    def tot_val(self):
        # compute total integrated value of the spectrum
        total = np.trapezoid(self.values, self.bins)
        return total
    
    import numpy as np

    @property
    def int_scale(self):
        """
        Compute integral scale of the spectrum E(k), defined as:
        L_int = ( ∫ E(k)/k dk ) / ( ∫ E(k) dk )

        Safely handles zero bins by ignoring them.
        """
        bins = np.array(self.bins, dtype=float)
        values = np.array(self.values, dtype=float)

        # Avoid zero bins for numerator
        mask = bins != 0

        if not np.any(mask) or np.sum(values) == 0:
            return 0.0  # handle empty or zero-only spectrum

        # Numerator: integrate E(k)/k over non-zero bins
        numerator = np.trapezoid(values[mask] / bins[mask], bins[mask])

        # Denominator: integrate E(k) over all bins
        denominator = np.trapezoid(values, bins)

        if denominator == 0:
            return 0.0

        L_int = numerator / denominator
        return L_int
    
class EnergyTransfer:
    # Abstraction for energy transfer data. Needs to come with a spectrum object at the same time.
    def __init__(self, transfer_data, bins, bin_edges, parent_simulation, time=None):
        # type_: string indicating type of transfer. For the current incompressible formalism without external forcing,
        # can be one of: 'BB', 'UU', 'BUT', 'UBT'
        # transfer_array: 2D numpy array with shape (bins, bins), where element (i,j) indicates energy transfer from shell j to shell i
        # bins: 1D numpy array with geometric mean of bin edges
        # parent: Simulation object that this transfer data belongs to
        self.bins = bins
        self.parent_simulation = parent_simulation
        self.bin_edges = bin_edges
        # get transfer arrays: 

        # before defining, check dimensions:
        for key, array in transfer_data.items():
            if array.shape != (len(bins), len(bins)):
                raise ValueError(f"Transfer array for {key} has shape {array.shape}, expected {(len(bins), len(bins))}")
        self.transfer_data = transfer_data
        self.BB = transfer_data['BB']
        self.UU = transfer_data['UU']
        self.UBT = transfer_data['UBT']
        self.BUT = - self.UBT  # by conservation, transfer from U to B is negative of transfer from B to U

        if time is None:
            print("Warning: EnergyTransfer created without time information; setting time=0.0")
            self.time = 0.0
        else:
            self.time = time

        # From bin edges, compute bin volumes:
        self.bin_widths = self.bin_edges[:, 1] - self.bin_edges[:, 0]
        
        # Get corresponding spectrum object: 
        try: 
            self.Bspectrum = self.parent_simulation.get_spectrum(time, field='B')
        except Exception as e:
            print(f"Warning: Could not get B spectrum for EnergyTransfer: {e}")
            self.Bspectrum = None
        # get integral scale from spectrum
        if self.Bspectrum is not None:
            self.int_scale = self.Bspectrum.int_scale
        else:
            self.int_scale = None
        
    def get(self, K, Q, channel):
        # retrieve transfer value for given K and Q via interpolation
        if K < self.bins[0] or K > self.bins[-1]:
            raise ValueError(f"K={K} out of bounds ({self.bins[0]} to {self.bins[-1]})")
        if Q < self.bins[0] or Q > self.bins[-1]:
            raise ValueError(f"Q={Q} out of bounds ({self.bins[0]} to {self.bins[-1]})")
        K_idx = np.searchsorted(self.bins, K)
        Q_idx = np.searchsorted(self.bins, Q)
        val = self.transfer_data[channel][K_idx, Q_idx]
        return val
    
    def plot(self, channel, norm='linear', linthresh=0.01, linscale=1,
            show_integral_scale=True, ax=None, add_colorbar=True,
            show_title=True, show_xlabel=True, show_ylabel=True,
            show_xticks=True):

        arr = self.transfer_data[channel]
        arr = arr / np.max(np.abs(arr))

        # Create figure only if no axis is provided
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            created_fig = True
        else:
            fig = ax.figure

        # --- Normalization ---
        if norm == 'symlog':
            norm_obj = SymLogNorm(linthresh=linthresh, linscale=linscale,
                                vmin=-1, vmax=1)
        else:
            norm_obj = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        # --- Integral scale lines ---
        if show_integral_scale:
            int_k = 1 / self.int_scale
            ax.axvline(int_k, color='gray', linestyle='-',
                    label='Integral Scale', alpha=0.3, linewidth=4)
            ax.axhline(int_k, color='gray', linestyle='-',
                    alpha=0.3, linewidth=4)
            print(int_k)

        # --- Image ---
        X, Y = np.meshgrid(self.bins, self.bins)
        im = ax.pcolormesh(
            X, Y, arr.T,
            cmap='RdBu_r',
            norm=norm_obj,
            shading='auto'
        )

        if add_colorbar:
            fig.colorbar(im, ax=ax, label='Energy Transfer')

        # Diagonal line
        ax.plot(self.bins, self.bins,
                color='black', linestyle='-', linewidth=1)

        # Axis labels
        if show_ylabel:
            ax.set_ylabel('Q Shell (giving)')
        if show_xlabel:
            ax.set_xlabel('K Shell (receiving)')
        if show_title:
            ax.set_title(f'Shell-to-Shell Energy Transfer for {channel} at t = {self.time}')

        # Axis scales
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Hide x ticks if requested
        if not show_xticks:
            ax.set_xticks([])
            ax.set_xticklabels([])

        ax.legend(title=channel)
        fig.tight_layout()

        # Only show if we created the figure
        if created_fig:
            plt.show()

        return im


    
    def get_tot_to_B(self, channel, K=None):
        # define array: 
        if channel in self.transfer_data.keys():
            arr = self.transfer_data[channel]
        elif channel.lower() == 'all':
            arr = self.transfer_data['BB'] + self.transfer_data['UBT']
        else: 
            raise ValueError(f"Unknown channel {channel}")
        if K is None:
            # get energy transfer to all K shells
            totals = np.sum(arr, axis=1)
            return self.bins, totals
        # get total energy received by shell K from all Q shells
        if K < self.bins[0] or K > self.bins[-1]:
            raise ValueError(f"K={K} out of bounds ({self.bins[0]} to {self.bins[-1]})")
        K_idx = np.searchsorted(self.bins, K)
        total = np.sum(arr[K_idx, :])
        return total

    def B_timescale(self, normalize=False, channel='all'):
        # For each magnetic mode K, compute the timescale tau_K = E_K / (dE_K/dt), where dE_K/dt is the total energy received by shell K from all Q shells
        _, dEk_dt = self.get_tot_to_B(channel)
        timescales = np.zeros_like(self.bins)
        for i, k in enumerate(self.bins):
            if k == 0.0:
                timescales[i] = np.nan
            else:
                P_k = self.Bspectrum.get_val(k)
                E_k = P_k * self.bin_widths[i]
                timescales [i] = E_k / dEk_dt[i] if dEk_dt[i] != 0.0 else np.nan
        if normalize:
            timescales *= 1024**3 / (2 * np.pi)**2
        return self.bins, timescales
    
    def plot_tot(self, channel, show_integral_scale=True, scale='linear', type='values', relative=False, label=None, return_vals=False):
        _, totals = self.get_tot_to_B(channel)
        if relative:
            totals_normalized = []
            for i, k in enumerate(self.bins):
                if k == 0.0:
                    totals_normalized.append(0.0)
                else:
                    P_k = self.Bspectrum.get_val(k)
                    E_k = P_k * self.bin_widths[i] # 1D power spectrum corresponds to energy per unit k. Multiply by bin width to get energy in bin
                    totals_normalized.append(totals[i] / E_k)
            totals = np.array(totals_normalized)
        if show_integral_scale:
            int_k = 1 / self.int_scale
            plt.axvline(int_k, color='gray', linestyle='-', label='Integral Scale', alpha=0.3, linewidth=4)
        if type == 'values':
            plt.semilogx(self.bins, totals, label=label)
        elif type == 'logslope':
            # compute local log-log slope of totals
            slopes = np.zeros_like(totals)
            for i in range(1, len(self.bins)-1):
                km1, kp1 = self.bins[i-1], self.bins[i+1]
                Pm1, Pp1 = totals[i-1], totals[i+1]
                slope = (np.log(Pp1) - np.log(Pm1)) / (np.log(kp1) - np.log(km1))
                slopes[i] = slope
            plt.semilogx(self.bins, slopes, label=label)
        plt.yscale(scale)
        plt.xlabel('K Shell')
        plt.ylabel('Total Energy Received')
        plt.title(f'Total Energy Received by Each K Shell for {channel}')
        plt.legend()
        if return_vals:
            return self.bins, totals

    def plot_Q(self, channel, k, show_integral_scale=True, scale='linear', type='values', show_k=True, label=None, relative=False):
        if channel in self.transfer_data.keys():
            arr = self.transfer_data[channel]
        elif channel.lower() == 'all':
            arr = self.transfer_data['BB'] + self.transfer_data['UBT']
        else: 
            raise ValueError(f"Unknown channel {channel}")
        if label is None:
            label = f't={self.time}'
        # plot energy received by shell k from all Q shells
        if k < self.bins[0] or k > self.bins[-1]:
            raise ValueError(f"k={k} out of bounds ({self.bins[0]} to {self.bins[-1]})")
        k_idx = np.searchsorted(self.bins, k)
        Q_values = arr[k_idx, :]
        if relative:
            # get k-power spectrum value at k:
            P_k = self.Bspectrum.get_val(k)
            Q_values = Q_values / P_k
        if show_integral_scale:
            int_k = 1 / self.int_scale
            plt.axvline(int_k, color='gray', linestyle='-', label='Integral Scale', alpha=0.3, linewidth=4)
        # plot vline at k:
        if show_k:
            plt.axvline(k, color='red', linestyle='-', label='k', alpha=0.3, linewidth=4)
        if type == 'values':
            plt.semilogx(self.bins, Q_values, label=label)
        elif type == 'logslope':
            # compute local log-log slope of Q_values
            slopes = np.zeros_like(Q_values)
            for i in range(1, len(self.bins)-1):
                km1, kp1 = self.bins[i-1], self.bins[i+1]
                Pm1, Pp1 = Q_values[i-1], Q_values[i+1]
                slope = (np.log(Pp1) - np.log(Pm1)) / (np.log(kp1) - np.log(km1))
                slopes[i] = slope
            plt.semilogx(self.bins, slopes, label=label)
        plt.yscale(scale)
        plt.xlabel('Q Shell')
        plt.ylabel(f'Energy Received by K={k} from Q Shells')
        plt.legend()

# I.O for athena simulation data analysis
class Simulation:

    """
    Interface for analyzing Parthenon simulation data.
    Defined via the directory containing the simulation output as well as analysis files. 
    Currently assumes that all files are directly placed in "directory", no subdirectories.
    """

    def __init__(self, directory, pattern="parthenon.prim.*.phdf"):
        self.directory = directory

        # Find all matching phdf files, sorted by timestep number:
        self.files = sorted(glob.glob(os.path.join(directory, pattern)))
        # find input file:
        input_files = glob.glob(os.path.join(directory, "*.in"))
        if not input_files:
            raise FileNotFoundError(f"No input file (*.in) found in directory {directory}")
        if len(input_files) > 1:
            raise RuntimeError(f"Multiple input files found in directory {directory}: {input_files}")
        self.input_file = input_files[0]
        self.config = io_utils.parse_parthenon_input(self.input_file)
        self.dt = self.config['parthenon/output2']['dt'] # timespan between outputs => used to map time to snapshot index

        # Find all flow analysis .hdf5 files:
        pattern = "flow_analysis_BB_parthenon.prim.*.hdf5"
        self.flow_analysis_files = sorted(glob.glob(os.path.join(directory, pattern)))

        # Find all spectral .spc files: 
        pattern = "*.[0-9][0-9][0-9][0-9][0-9].spc"
        self.spectral_files = sorted(glob.glob(os.path.join(directory, pattern)))
        self.spectral_files_vel = [f for f in self.spectral_files if "spec_type_0" in f]
        self.spectral_files_mag = [f for f in self.spectral_files if "spec_type_2" in f]

        # Find all transfer analysis .pkl files:
        pattern = "transfer_analysis_parthenon.prim.*.pkl"
        self.transfer_analysis_files = sorted(glob.glob(os.path.join(directory, pattern)))

        # Calculate turnover time given initial conditions:
        config = self.config 
        B_rms = config['problem/stochastic_B_field']['B_rms']
        rho = config['problem/stochastic_B_field']['rho0']
        P = config['problem/stochastic_B_field']['p0']
        kI = config['problem/stochastic_B_field']['kI']
        self.box_size = config['parthenon/mesh']['x1max'] - config['parthenon/mesh']['x1min']
        v_alfven = B_rms / np.sqrt(rho)
        self.initial_alfven_time = (v_alfven * kI)**(-1)

        # get linear resolution
        self.N = config['parthenon/mesh']['nx1']
        # Nyquist wavenumber from resulution (assuming box size = 2pi)
        self.k_ny = self.N / 2

    def __repr__(self):
        s = "Simulation directory: "+self.directory+"\n"
        s += "Number of Snapshots: "+str(len(self.files))+"\n"
        s += "Covered time: "+str(len(self.files)*self.dt)
        return s
    
    @property
    def history(self):
        # read history file if it exists, return as pandas DataFrame
        hist_files = sorted(glob.glob(os.path.join(self.directory, "*.hst")))
        if len(hist_files) == 0:
            raise FileNotFoundError(f"No history file (*.hst) found in directory {self.directory}")
        if len(hist_files) > 1:
            print(f"Warning: multiple history files found in directory {self.directory}, using {hist_files[0]}")
        hist_file = hist_files[0]
        try:
            df = pd.read_csv(hist_file, delim_whitespace=True, comment='#')
        except:
            df = pd.read_csv(hist_file, sep=r"\s+", comment="#", engine="python")
        # Extract column names from the first commented line
        with open(hist_file) as f:
            for line in f:
                if line.startswith("# ["):
                    columns = [entry.split("=")[1].strip() for entry in line.split() if "=" in entry]
                    break
        df.columns = columns
        # Compute dlog(ME)/dlog(t)
        df['dlogME_dlogt'] = np.gradient(np.log(df['ME']), np.log(df['time']))
        df["dlogKE_dlogt"] = np.gradient(np.log(df['KE']), np.log(df['time']))
        return df
    
    def v_rms(self, time):
        # retrieve v_rms from history file at given time
        df = self.history
        if time < df['time'].min() or time > df['time'].max():
            raise ValueError(f"Time {time} out of range ({df['time'].min()} to {df['time'].max()})")
        v_rms = np.interp(time, df['time'], (2*df['KE']/self.box_size**3)**0.5) 
        return v_rms
    
    def B_rms(self, time):
        # retrieve B_rms from history file at given time
        df = self.history
        if time < df['time'].min() or time > df['time'].max():
            raise ValueError(f"Time {time} out of range ({df['time'].min()} to {df['time'].max()})")
        B_rms = np.interp(time, df['time'], (2*df['ME']/self.box_size**3)**0.5) 
        return B_rms
    
    def get_snapshot(self, time, timeunit='code'):
        # retrieve raw snapshot that is closest to "time":
        step = round(time/self.dt)
        if step < 0 or step >= len(self.files):
            raise ValueError(f"Time {time} out of range (0 to {self.dt * len(self.files)})")
        filename = self.files[step]
        return SimSnap(filename)
    
    def get_flow_analysis_file(self, time):
        # retrieve flow analysis file that is closest to "time":
        step = round(time/self.dt)
        if step < 0 or step >= len(self.files):
            raise ValueError(f"Time {time} out of range (0 to {self.dt * len(self.files)})")
        filename = min(self.flow_analysis_files, key=lambda f: abs(io_utils.extract_index(f) - step))
        data = io_utils.read_h5_to_dict(filename)
        return data

    def get_spectrum_(self, time, field="B"):
        # retrieve flow analysis file that is closest to "time":
        if field == "B":
            files = self.spectral_files_mag
        elif field == "v": 
            files = self.spectral_files_vel

        if len(self.spectral_files) == 0:
            raise RuntimeError("No spectral files found in simulation directory.")
        step = round(time/self.dt)
        if step < 0 or step >= len(files):
            raise ValueError(f"Time {time} out of range (0 to {self.dt * len(files)})")
        filename = min(files, key=lambda f: abs(io_utils.extract_index(f) - step))
        # get the exact time from the snapshot index:
        step_exact = io_utils.extract_index(filename)
        time = step_exact * self.dt
        
        dataframe = io_utils.parse_spc_file(filename)
        return Spectrum(field, dataframe["Bin"], dataframe["En_sum"], parent=self, binscale='linear', time=time)

    def get_spectrum(self, time, field='B'):
        # retrieve flow analysis file that is closest to "time":
        if len(self.flow_analysis_files) == 0:
            raise RuntimeError("No flow analysis files found in simulation directory.")
        step = round(time/self.dt)
        if step < 0 or step >= len(self.flow_analysis_files):
            raise ValueError(f"Time {time} out of range (0 to {self.dt * len(self.files)})")
        filename = min(self.flow_analysis_files, key=lambda f: abs(io_utils.extract_index(f) - step))
        # get the exact time from the snapshot index:
        step_exact = io_utils.extract_index(filename)
        time = step_exact * self.dt
        # Extract Spectral content:
        return io_utils.get_spectrum(filename, field=field, time=time, parent=self)
    
    def get_energy_transfer(self, time):
        # Currently, this assumes that the fiels contain the transfer terms ''BB', 'UU' and 'UBTb'. 
        # retrieve transfer analysis file that is closest to "time":

        # Currently, this is not very clean, as the transfer analysis file naming conventions are messy. 
        # fix later. 

        step = int(time / self.dt)
        if step < 0 or step >= len(self.files):
            raise IndexError(f"Time {time} out of range (0 to {self.dt * len(self.files)})")

        # Load pkl file:
        filename = min(self.transfer_analysis_files, key=lambda f: abs(io_utils.extract_index(f) - step))
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        types = ['BB', 'UU', 'UBTb']
        transfer_data = {}

        for type_ in types:

            # Extract transfer array and bins:
            try:
                transfer_dict = data['WW'][type_]['AnyToAny']
            except KeyError:
                raise KeyError(f"Transfer type {type_} not found in transfer analysis data.")
            transfer_array, k_labels, _ = io_utils.dict_to_array(transfer_dict)
            if type_ == 'UBTb':
                type_ = 'UBT'  # rename for consistency
            transfer_data[type_] = transfer_array            
        k_edges = np.array([tuple(map(float, s.split('-'))) for s in k_labels])
        k_geom = np.sqrt(k_edges[:, 0] * k_edges[:, 1])

        # get the exact time from the snapshot index:
        step_exact = io_utils.extract_index(filename)
        time = step_exact * self.dt

        energy_transfer = EnergyTransfer(transfer_data, k_geom, k_edges, parent_simulation=self, time=time)
        return energy_transfer