"""
BSD 2-Clause License
Author: Lisandro Dalcin and Mikael Mortensen
Contact:    dalcinl@gmail.com or mikaem@math.uio.no

Copyright (c) 2017, Lisandro Dalcin and Mikael Mortensen. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

# from
# https://bitbucket.org/mpi4py/mpi4py-fft/raw/67dfed980115108c76abb7e865860b5da98674f9/examples/spectral_dns_solver.py
def get_local_wavenumbermesh(FFT, L):
    """Returns local wavenumber mesh."""
    s = FFT.local_slice()
    N = FFT.global_shape()
    # Set wavenumbers in grid
    k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N[:-1]]
    k.append(np.fft.rfftfreq(N[-1], 1./N[-1]).astype(int))
    K = [ki[si] for ki, si in zip(k, s)]
    Ks = np.meshgrid(*K, indexing='ij', sparse=True)
    Lp = 2*np.pi/L
    for i in range(3):
        Ks[i] = (Ks[i]*Lp[i]).astype(float)
    return [np.broadcast_to(k, FFT.shape(True)) for k in Ks]
