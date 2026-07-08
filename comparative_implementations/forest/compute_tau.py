"""Compute and store line-of-sight optical-depth spectra from ray data.

This script:
1. Reads atomic transition parameters from a fixed-width line list file.
2. Computes Voigt-profile optical depth grids for selected simulation snapshots.
3. Converts optical depth to transmitted flux and writes wavelength/flux groups
   into each snapshot HDF5 ray file.
4. Provides helper utilities for observational post-processing steps
   (trimming, continuum normalization, filtering, and rebinning).

Notes
-----
- Most calculations are performed in CGS units.
- The script is written as an executable module (work is done at import/runtime
  in the snapshot loop near the end of the file).
- Several helper functions at the bottom are not called in the main loop but
  are intended for follow-up observational-style processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import simloader as sl
import mycosmo as mc
from astropy import constants as const
from astropy import units
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import binned_statistic

# Constants
Msun = const.M_sun.cgs.value   # Solar mass [g]
c = const.c.cgs.value          # Speed of light [cm/s]
km = units.km.in_units('cm')   # Units: 1 km  = 1e5  cm
pc = units.pc.in_units('cm')   # Units: 1 pc  = 3e18 cm
kpc = units.kpc.in_units('cm') # Units: 1 kpc = 3e21 cm
Mpc = units.Mpc.in_units('cm') # Units: 1 Mpc = 3e24 cm
kB = const.k_B.cgs.value       # Boltzmann's constant [g cm^2/s^2/K]
mp = const.m_p.cgs.value       # Mass of hydrogen atom (g)
me = const.m_e.cgs.value       # Electron mass [g]
mH = mp+me
ee = const.e.esu.value      # Electron charge [g^(1/2) cm^(3/2) / s]
X  = 0.76                   # Primordial hydrogen mass fraction
GAMMA = 5. / 3.             # Adiabatic index of simulated gas
GAMMA_MINUS1 = GAMMA - 1.   # For convenience




def read_line_parameters(line_list_file='line_list.txt'):
    """Parse spectral-line metadata and build a lookup dictionary.

    Parameters
    ----------
    line_list_file : str, optional
        Path to the line list text file. Each non-comment line is expected to
        follow a fixed-width layout:
        - columns [:10]: species label
        - columns [10:-12]: wavelength, damping constant, oscillator strength
        - columns [-12:-1]: alternate short name used as dictionary key

    Returns
    -------
    dict
        Mapping from alternate line name (e.g., ``'Ly a'``) to a dictionary:
        - ``lambda`` : rest wavelength [cm]
        - ``nu``     : rest frequency [Hz]
        - ``f``      : oscillator strength [dimensionless]
        - ``Gamma``  : damping constant [1/s]
        - ``tau``    : natural lifetime [s], defined as ``1/Gamma``
        - ``m_atom`` : atomic mass in atomic-mass-unit scale used by this file

    Notes
    -----
    Embedded abundance/atomic tables are used to infer atomic masses from the
    line naming convention. The returned dictionary only includes entries with
    non-empty alternate names.
    """
    with open(line_list_file, 'r') as f:
        lines = f.readlines()
    species, wavelength, gamma, osc_str, alt_name = [], [], [], [], []
    for line in lines:
        if line=='\n' or line[0]=="#":
            continue

        species.append( line[:10].strip() )
        alt_name.append( line[-12:-1].strip() )
        vals = line[10:-12].strip().split()
        wavelength.append( float(vals[0]) )
        gamma.append( float(vals[1]) )
        osc_str.append( float(vals[2]) )
    
    #arrange lines in a dict
    # Taken from Cloudy documentation.
    solar_abundance = {
    'H' : 1.00e+00, 'He': 1.00e-01, 'Li': 2.04e-09,
    'Be': 2.63e-11, 'B' : 6.17e-10, 'C' : 2.45e-04,
    'N' : 8.51e-05, 'O' : 4.90e-04, 'F' : 3.02e-08,
    'Ne': 1.00e-04, 'Na': 2.14e-06, 'Mg': 3.47e-05,
    'Al': 2.95e-06, 'Si': 3.47e-05, 'P' : 3.20e-07,
    'S' : 1.84e-05, 'Cl': 1.91e-07, 'Ar': 2.51e-06,
    'K' : 1.32e-07, 'Ca': 2.29e-06, 'Sc': 1.48e-09,
    'Ti': 1.05e-07, 'V' : 1.00e-08, 'Cr': 4.68e-07,
    'Mn': 2.88e-07, 'Fe': 2.82e-05, 'Co': 8.32e-08,
    'Ni': 1.78e-06, 'Cu': 1.62e-08, 'Zn': 3.98e-08}

    atomic_mass = {
    'H' : 1.00794,   'He': 4.002602,  'Li': 6.941,
    'Be': 9.012182,  'B' : 10.811,    'C' : 12.0107,
    'N' : 14.0067,   'O' : 15.9994,   'F' : 18.9984032,
    'Ne': 20.1797,   'Na': 22.989770, 'Mg': 24.3050,
    'Al': 26.981538, 'Si': 28.0855,   'P' : 30.973761,
    'S' : 32.065,    'Cl': 35.453,    'Ar': 39.948,
    'K' : 39.0983,   'Ca': 40.078,    'Sc': 44.955910,
    'Ti': 47.867,    'V' : 50.9415,   'Cr': 51.9961,
    'Mn': 54.938049, 'Fe': 55.845,    'Co': 58.933200,
    'Ni': 58.6934,   'Cu': 63.546,    'Zn': 65.409}

    atomic_number = {
    'H' : 1,  'He': 2,  'Li': 3,
    'Be': 4,  'B' : 5,  'C' : 6,
    'N' : 7,  'O' : 8,  'F' : 9,
    'Ne': 10, 'Na': 11, 'Mg': 12,
    'Al': 13, 'Si': 14, 'P' : 15,
    'S' : 16, 'Cl': 17, 'Ar': 18,
    'K' : 19, 'Ca': 20, 'Sc': 21,
    'Ti': 22, 'V' : 23, 'Cr': 24,
    'Mn': 25, 'Fe': 26, 'Co': 27,
    'Ni': 28, 'Cu': 29, 'Zn': 30}


    def line_name_to_matom(name):
        if name[:3]=='Ly ' or name[2]=='H ':
            return atomic_mass['H']
        else:
            return atomic_mass[name[:2].strip()]

    lines_parameter = {}
    for i in range(len(alt_name)):
        if len(alt_name[i])>0:
            lines_parameter[alt_name[i]] = \
                    { 'lambda': wavelength[i]*1e-8, #cm
                      'nu': c/(wavelength[i]*1e-8), #Hz
                      'f': osc_str[i], 
                      'Gamma': gamma[i], #1/s
                      'tau': 1.0/gamma[i], #s
                      'm_atom': line_name_to_matom(alt_name[i])
                    }

    return lines_parameter

#lines_parameter= { 'Lya': { 'lambda': 1215.67e-8, #cm
#                            'nu': 2.466e15, #Hz
#                            'f': 0.41641, #osc. strength
#                            'Gamma': 6.2648e8, #1/s
#                            'tau': 1.5962e-9, #s
#                            'm_atom': 1}, #mH
#                   'Lyb': { 'lambda': 1025.7220e-8, #cm
#                             'nu': 2.923e15, #Hz
#                             'f': 0.0791,
#                             'Gamma': 5.57e+07, #1/s
#                             'tau': 1.795e-08, #s
#                             'm_atom': 1}, #mH
#                
#                # template entry
#                #   ''   : { 'lambda': ,
#                #             'nu': , 
#                #             'f': ,
#                #             'Gamma': ,
#                #             'tau': ,
#                #             'm_atom'},
#                 }

def Voigt(a, u):
    """Evaluate an approximate Voigt profile for small damping parameter.

    Parameters
    ----------
    a : float
        Damping parameter of the line profile for one gas element.
    u : ndarray
        Dimensionless frequency/velocity offset array.

    Returns
    -------
    ndarray
        Approximate Voigt profile values at ``u``.

    Notes
    -----
    Uses a piecewise approximation for ``a << 1`` following references noted
    in the code comments (Harris 1948; Garcia 2006). The implementation treats
    three ``u^2`` regimes: near-core, intermediate, and far-wing.
    """
    #approx. for a<<1 (Harris 1948, Garcia 2006)
    voigt = np.empty(len(u))

    u2 = u*u

    w = (u2>=1e-4)&(u2<=25.0)

    exp_u2 = np.exp(-u2[w])

    H0 = exp_u2
    H1 = -2/np.sqrt(np.pi) * 0.5*exp_u2/u2[w] * ( (4*u2[w]+3)*(u2[w]+1)*exp_u2 - (2*u2[w]+3)/u2[w] * np.sinh(u2[w]))
    H2 = (1-2*u2[w])*exp_u2

    voigt[w] = H0 + a*H1 + a*a*H2
    voigt[u2<1e-4] = 1
    voigt[u2>25.0] = a/u2[u2>25.0]/np.sqrt(np.pi)*(1.5/u2[u2>25.0]+1-a*a)

    return voigt 


def calculate_tau_line(data, min_Dshift, max_Dshift, n_freq, line, static=False, only_rays=None, verbose=False):
    """Compute optical-depth spectra along one or more rays for one transition.

    Parameters
    ----------
    data : dict-like
        Ray container returned by ``simloader.readColtRay``. Expected keys
        include cosmology metadata (``a``, ``Omega0``, ``HubbleParam``),
        ``NumRays``, and a ``rays`` sequence with per-cell thermodynamic and
        kinematic fields in CGS units.
    min_Dshift : float
        Minimum Doppler-shift bound in km/s for the frequency grid.
    max_Dshift : float
        Maximum Doppler-shift bound in km/s for the frequency grid.
    n_freq : int
        Number of frequency bins on the Doppler-shift grid.
    line : str
        Line key present in global ``lines_parameter`` (e.g., ``'Ly a'``).
    static : bool, optional
        If ``True``, excludes peculiar-velocity term and uses only Hubble-flow
        shifts. If ``False``, includes cell peculiar velocities.
    only_rays : array-like or None, optional
        Subset of ray indices to process. ``None`` means all rays.
    verbose : bool, optional
        If ``True``, prints progress over processed rays.

    Returns
    -------
    tuple
        ``(Dv, tau)`` where:
        - ``Dv``  : 1D Doppler-shift grid [cm/s]
        - ``tau`` : 2D optical depth array with shape
          ``(len(only_rays), n_freq)``

    Notes
    -----
    The implementation follows the standard decomposition:
    - thermal broadening scale ``b_th``
    - dimensionless frequency offset ``u``
    - damping parameter ``a``
    - per-cell optical-depth prefactor multiplied by the Voigt profile
    """

    #b_th = b_th_factor * T^(1/2)
    # => u = u_factor * T^(-1/2) * Delta_v[cm/s]
    #    a = a_factor * T^(-1/2)
    #    tau = tau_factor * nHI[1/cm3] * dl[cm] * a * Voigt(u, a)
    b_th_factor = np.sqrt(2*kB/(lines_parameter[line]['m_atom']*mH))
    u_factor = 1/b_th_factor
    a_factor = lines_parameter[line]['Gamma']*0.25/np.pi/b_th_factor/lines_parameter[line]['nu']*c
    tau_factor = 4 * np.pi**1.5 * ee**2 * lines_parameter[line]['f'] / ( me * c * lines_parameter[line]['Gamma'] )

    if only_rays is None:
        only_rays = np.arange(data['NumRays'])  #all of them

    #create velocity grid
    Dv = np.linspace(min_Dshift*km, max_Dshift*km, n_freq) # Rest-frame frequency grid [cm/s]
    
    tau = np.zeros((len(only_rays), n_freq))
    for iray, ray in enumerate(only_rays):
        if verbose:
            print(iray, '/', len(only_rays)-1, end='\r')
        hz = mc.HubbleParam(data['a'], OmegaM=data['Omega0'], OmegaL=1-data['Omega0'], Hubble0=data['HubbleParam'])
        Dv_Hubble_flow = 100 * hz  * km / Mpc * (data['rays'][ray].midpoints_cgs - 0.5 * data['rays'][ray].segments_cgs) # Hubble velocity offset (left edges relative to systemic) [cm/s]
        a = a_factor / np.sqrt(data['rays'][ray].temperature)
        mu = 4. / (1. + 3.*X + 4.*X * data['rays'][ray].electron_abundance)
        K_tau = tau_factor * data['rays'][ray].density_cgs * data['rays'][ray].xHI / (mu*mH) * data['rays'][ray].segments_cgs * a 

        for i in range(len(a)):
            if static:
                u = (-Dv + Dv_Hubble_flow[i]                                    ) * u_factor / np.sqrt(data['rays'][ray].temperature[i]) # Doppler frequency
            else:
                u = (-Dv + Dv_Hubble_flow[i] + data['rays'][ray].velocity_cgs[i]) * u_factor / np.sqrt(data['rays'][ray].temperature[i]) # Doppler frequency
            
            this_tau = K_tau[i] * Voigt(a[i], u)
            tau[iray, :] += this_tau

    return Dv, tau


def get_length_kms_from_cMpch(L_in_cMpch, redshift, OmegaM=0.3, OmegaL=0.7, h=0.7):
    """Convert comoving length (cMpc/h) to an equivalent velocity span (km/s).

    Parameters
    ----------
    L_in_cMpch : float
        Comoving box/path length in cMpc/h.
    redshift : float
        Redshift at which to evaluate the Hubble conversion.
    OmegaM : float, optional
        Matter density parameter.
    OmegaL : float, optional
        Dark-energy density parameter.
    h : float, optional
        Reduced Hubble constant (H0 / 100 km/s/Mpc).

    Returns
    -------
    float
        Velocity-space span in km/s corresponding to the provided length.
    """
    return L_in_cMpch / (1+redshift) * 100 * h * (OmegaM*(1+redshift)**3+OmegaL)**0.5


def Doppler_shift_to_wavelength(dv, line, redshift):
    """Map Doppler shifts to observed-frame wavelength for a transition.

    Parameters
    ----------
    dv : ndarray
        Doppler-shift grid [cm/s].
    line : str
        Line key present in global ``lines_parameter``.
    redshift : float
        Cosmological redshift.

    Returns
    -------
    ndarray
        Observed-frame wavelength array in cm.
    """
    return (1 - dv/c) * lines_parameter[line]['lambda'] * (1+redshift) #cm



# Main production loop: compute spectra for selected snapshots and persist the
# resulting wavelength/flux arrays into the corresponding ray HDF5 files.
for snap in [54, 58, 61, 64, 70, 75, 80]:

    filename = f'ray_files/rays_{snap:03d}.hdf5'
    spec_num = 0
    data = sl.readColtRay(filename)
    redshift = 1/data['a']-1

    lines_parameter = read_line_parameters()

    spectrum_length_in_kms = get_length_kms_from_cMpch(100, redshift, OmegaM=data['Omega0'], OmegaL=1-data['Omega0'], h=data['HubbleParam'])

    spectral_resolution_kms = 1
    num_freq_bins = int(spectrum_length_in_kms / spectral_resolution_kms)

    doppler_shift, taus = calculate_tau_line(data, 0, spectrum_length_in_kms, num_freq_bins, 'Ly a', static=True, verbose=True)

    wavelength_in_cm = Doppler_shift_to_wavelength(doppler_shift, 'Ly a', redshift)
    wavelength_in_Ang = wavelength_in_cm * 1e8
    flux = np.exp(-taus[spec_num, :])

    with h5py.File(filename, 'r+') as f:
        if '/wavelength' in f: del f['wavelength']
        gr_wl = f.create_group('wavelength')
        
        if '/flux' in f: del f['flux']
        gr_fl = f.create_group('flux')
        
        for ray in range(data['NumRays']):
            gr_wl.create_dataset(str(ray), data=wavelength_in_cm)
            gr_fl.create_dataset(str(ray), data=np.exp(-taus[ray, :]))


