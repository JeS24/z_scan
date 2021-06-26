"""
Utilities for Z-Scan experiment

"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from io import StringIO
from pathlib import Path, PurePath
from scipy.optimize import curve_fit


# General constants
lmd = 532e-9  # m | Wavelength
L = 12.2  # cm | Total length traversed by the sample
zR = 0  # m | Rayleigh or Diffraction Length | JUST INITIALIZATION

EXPORT_WIDTH = 960
EXPORT_HEIGHT = 540


def preproc(file_path, z0=0., OA=False, SA=False):
    """
    Preprocessed raw data from the sensors.
    Strips whitespaces, tabs and metadata.

    Parameters
    ----------
    file_path: pathlib.Path.PurePath or str
        Path to logfile, produced by StarLab (in .txt format)
    z0: float
        Location of focus
        Defaults to ``0.``
    OA: bool
        Whether the data corresponds to Open Aperture scan
        If True, the data is centered around minimum power
        Defaults to ``False``
    SA: bool
        If Saturable Absorption is present

    Returns
    -------
    out_path: str
        Path, where output .csv file is stored
    mult: pandas.DataFrame
        Pandas DataFrame, containing data from the processed .csv file

    Optionally Returns
    ------------------
    z0: float
        Calculated Rayleigh length for Open Aperture scans only

    Notes
    -----
    Return order is (out_path, mult) for CA scans and (out_path, z0, mult) for OA scans

    """
    if not isinstance(file_path, PurePath):
        file_path = Path(file_path)

    file_name = file_path.stem

    with open(file_path) as file:
        print(f"Reading {file_path}...")
        mult = "\n".join(line.strip().replace("\t", ",") for line in file)
        mult = mult[mult.find("Timestamp"):]
        print("Removing whitespaces (tabs and spaces)...")

    out_path = file_path / f"../{file_name}Proc.csv"

    mult = StringIO(mult)
    mult = pd.read_csv(mult)
    mult = mult.dropna()
    mult.columns = mult.columns.str.replace(' ', '')

    # Conversion factor from time to length
    t2l = L / mult["Timestamp"].iloc[-1]  # cm/s
    # Average power | Normalization factor
    P_avg = mult["ChannelB"].mean()  # W

    mult["Length"] = (mult["Timestamp"] * t2l)

    if OA:
        P_avg = (mult["ChannelB"].iloc[0] + mult["ChannelB"].iloc[-1]) / 2
        z0 = mult["Length"].iloc[mult["ChannelB"].argmin()]

    if SA:
        z0 = mult["Length"].iloc[mult["ChannelB"].argmax()]

    mult["Transmittance"] = mult["ChannelB"] / P_avg
    mult["Length"] -= z0

    # Saving to .csv
    mult.to_csv(out_path, index=False)
    print("Done!")

    if OA:
        return out_path, z0, mult

    return out_path, mult


def fit_and_plot(mult, func, w0):
    """
    Fits the data and plots the result

    Parameters
    ----------
    mult: pandas.DataFrame
        DataFrame, containing all the StarLab data (after running ``preproc()`` on it)
    func: callable
        Function to fit against
    w0: float
        Beam waist
        This is a variable fit parameter

    Returns
    -------
    popt: numpy.ndarray
        Array, containing calculated values of fit parameter
    p_sigma: numpy.ndarray
        Array, containing p_sigma errors in fit parameters
    fit: plotly.graph_objects.Figure
        Interactive fit, containing the original data and the fit

    """
    global zR
    zR = (np.pi / lmd) * (w0 ** 2)  # m | Rayleigh or Diffraction Length

    length = mult["Length"].to_numpy()
    irrad = mult["Transmittance"].to_numpy()

    popt, pcov = curve_fit(func, length, irrad)
    p_sigma = np.sqrt(np.diag(pcov))

    # Plotly plot for analysis
    # Create traces
    fit = go.Figure()
    fit.add_trace(go.Scatter(x=length, y=irrad, mode='markers', name="Experimental Data"))
    fit.add_trace(go.Scatter(x=length, y=func(length, *popt), mode='lines', name="Theoretical Fit"))
    fit.update_layout(
        xaxis_title=r"$\text{Length, }z\text{ (in cm)}$",
        yaxis_title=r"$\text{Normalized Transmittance, }T$",
        legend=dict(
            yanchor="top",
            y=1.,
            xanchor="left",
            x=0.,
            bgcolor="rgba(0, 0, 0, 0)"
        ),
        font=dict(
            family="Georgia",
            size=16
        ),
        template="simple_white",
    )
    fit.update_xaxes(
        showgrid=True,
        mirror=True,
        ticks='outside',
        showline=True
    )
    fit.update_yaxes(
        showgrid=True,
        mirror=True,
        ticks='outside',
        showline=True
    )
    fit.show()

    print("Fit parameters with error (p_sigma):")
    print(f"{abs(popt[0]):.4f} \pm {abs(p_sigma[0]):.4f} ({abs(100 * p_sigma[0] / popt[0]):.4f} % Error)")

    return popt, p_sigma, fit


# Open Aperture
def TO(z, Q):
    x = z / (zR * 100)  # Converting zR to cm
    x2 = x ** 2

    return 1 - (Q / (x2 + 1))


# Closed Aperture - Two-Parameter Fit
def TC2a(z, del_phi0, Q):
    x = z / (zR * 100)  # Converting zR to cm
    x2 = x ** 2

    return (1 + ((4 * del_phi0 * x) / ((x2 + 9) * (x2 + 1))) - (Q / (x2 + 1)))


# Nguyen et al, 2014
# Closed Aperture: NA & PNR
def TC2b(z, del_phi0, Q):
    x = z / (zR * 100)  # Converting zR to cm
    x2 = x ** 2

    return (1 + (4 * del_phi0 * x) / ((x2 + 9) * (x2 + 1))) * (1 - Q / (x2 + 1))


# n2 | del_phi0 = k n_2 I_0 L_eff
def n2(del_phi0, I0):
    """
    Returns Third Order Non-Linear Refractive Index in units of "cm^2 / W"

    Parameters
    ----------
    del_phi0: float
        Non-linear phase shift | Fitting parameter from Closed Aperture Z-Scan Fit
    I0: float
        Peak on-axis irradiance at focus

    Returns
    -------
    float
        Third Order Non-Linear Refractive Index in units of "cm^2 / W"

    """
    k = (2 * np.pi) / lmd
    L_eff = 1e-3  # Effective length = 1 mm

    return del_phi0 / (k * I0 * L_eff)


# Q = (beta I_0 L_eff) / (2 ** 1.5)
def beta(Q, I0):
    """
    Returns Third Order Non-Linear Absorption Coefficient in units of "cm / W"

    Parameters
    ----------
    Q: float
        Fitting parameter from Open Aperture Z-Scan Fit
    I0: float
        Peak on-axis irradiance at focus

    Returns
    -------
    float
        Third Order Non-Linear Absorption Coefficient in units of "cm / W"

    """
    L_eff = 1e-3  # Effective length = 1 mm

    return (2 ** 1.5 * Q) / (I0 * L_eff)


def del_Tpv(del_phi0):
    """
    Returns the difference between the normalized 
    peak and valley transmittance, using
    $T_{pv} = 0.406|\del\phi_0|$ for $S \approx 0$
    if $\del\phi_0 <= \pi$.

    Parameters
    ----------
    del_phi0: float
        Small phase distortion, obtained from Z-Scan fit

    Returns
    -------
    float
        Difference between normalized peak and valley transmittance

    """
    if abs(del_phi0) >= np.pi:
        print("Condition not satisfied.\n")

    return 0.406 * abs(del_phi0)


def stddev(q0, q_list, del_list):
    """
    Returns standard deviation / error

    Parameters
    ----------
    q0: float
        Quantity, whose standard deviation / error is to be calculated
    q_list: array_like
        List of quantities, that have uncertainties
    del_list: array_like
        List of uncertainties in quantities in ``q_list``

    Returns
    -------
    float
        Standard Deviation or Error in ``q0``

    """
    sum_ = 0
    for i, j in zip(del_list, q_list):
        sum_ += (i / j) ** 2

    return abs(q0 * np.sqrt(sum_))
