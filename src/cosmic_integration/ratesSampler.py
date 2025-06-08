#! /usr/bin/env python3
# pyright: ignore-file
import sys
import os
import numpy as np
import h5py
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy.stats import norm as NormDist
import astropy.units as units
from astropy.cosmology import WMAP9 as cosmology
import csv
import argparse
import time
from tqdm.auto import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))

# defaults

M1_MINIMUM             = 5.0
M1_MAXIMUM             = 150.0
M2_MINIMUM             = 0.1
BINARY_FRACTION        = 1.0

MAX_CHIRPMASS          = 125.0
MIN_CHIRPMASS          = 0.5
McBIN_WIDTH_PERCENT    = 5.0

MAX_FORMATION_REDSHIFT = 10.0
MAX_DETECTION_REDSHIFT = 1.5
REDSHIFT_STEP          = 0.1
NUM_REDSHIFT_BINS      = int(MAX_DETECTION_REDSHIFT / REDSHIFT_STEP)

COMPAS_HDF5_FILE_PATH  = '.'
COMPAS_HDF5_FILE_NAME  = 'COMPAS_Output.h5'

SNR_NOISE_FILE_PATH    = HERE
SNR_NOISE_FILE_NAME    = '/SNR_Grid_IMRPhenomPv2_FD_all_noise.hdf5'
SNR_SENSITIVITY        = 'O1'
SNR_THRESHOLD          = 8.0

Mc_STEP                = 0.1
ETA_STEP               = 0.01
SNR_STEP               = 0.1

# for calculating the distribution of metallicities at different redshifts
# see CosmicIntegration::FindZdistribution()
MINIMUM_LOG_Z          = -12.0
MAXIMUM_LOG_Z          = 0.0
Z_SCALE                = 0.0
SKEW                   = 0.0 
LOG_Z_STEP             = 0.01

# Neijssel+19 Eq.7-9
NEIJSSEL_Z0 = 0.035
NEIJSSEL_ALPHA = -0.23
NEIJSSEL_SIGMA = 0.39
NEIJSSEL_SFR_A = 0.01
NEIJSSEL_SFR_D = 4.7


SAMPLE_COUNT = 10



ALPHA_VALUES = [-0.500, -0.400, -0.300, -0.200, -0.100, -0.001]
SIGMA_VALUES = [ 0.100,  0.200,  0.300,  0.400,  0.500,  0.600]
SFR_A_VALUES = [ 0.005,  0.007,  0.009,  0.011,  0.013,  0.015]
SFR_D_VALUES = [ 4.200,  4.400,  4.600,  4.800,  5.000,  5.200]


# globals

verbose = False


def Stop(p_ErrStr = 'An error was encountered'):
    print(p_ErrStr)
    print('Terminating')
    sys.exit()    



class SelectionEffects:

    def __init__(self, 
                 p_SNRfilePath    = SNR_NOISE_FILE_PATH,
                 p_SNRfileName    = SNR_NOISE_FILE_NAME,
                 p_SNRsensitivity = SNR_SENSITIVITY,
                 p_SNRthreshold   = SNR_THRESHOLD):

        self.SNRfilePath                 = p_SNRfilePath
        self.SNRfileName                 = p_SNRfileName

        self.SNRsensitivity              = p_SNRsensitivity
        self.SNRthreshold                = p_SNRthreshold

        self.SNRthetas                   = self.GenerateSNRthetas()

        self.SNRmassAxis, \
        self.SNRgrid                     = self.ReadSNRdata()

        self.SNRinterpolator             = scipy.interpolate.RectBivariateSpline(np.log(self.SNRmassAxis), np.log(self.SNRmassAxis), self.SNRgrid)


        self.SNRgridAt1Mpc, \
        self.detectionProbabilityFromSNR = self.ComputeSNRandDetectionGrids(self.SNRsensitivity, self.SNRthreshold)


    def GenerateSNRthetas(self, p_NumThetas = 1E6, p_MinThetas = 1E4):

        p_NumThetas = int(p_NumThetas)

        if p_NumThetas < p_MinThetas:
            Stop('SelectionEffects::GenerateSNRthetas(): Number of thetas requested ({:s}) < minumum required {:s}'.format(p_NumThetas, p_MinThetas))

        cosThetas = np.random.uniform(low = -1, high = 1, size = p_NumThetas)
        cosIncs   = np.random.uniform(low = -1, high = 1, size = p_NumThetas)
        phis      = np.random.uniform(low =  0, high = 2 * np.pi, size = p_NumThetas)
        zetas     = np.random.uniform(low =  0, high = 2 * np.pi, size = p_NumThetas)

        Fps = (0.5 * np.cos(2 * zetas) * (1 + cosThetas**2) * np.cos(2 * phis) - np.sin(2 * zetas) * cosThetas * np.sin(2 * phis))
        Fxs = (0.5 * np.sin(2 * zetas) * (1 + cosThetas**2) * np.cos(2 * phis) + np.cos(2 * zetas) * cosThetas * np.sin(2 * phis))

        thetas = np.sqrt(0.25 * Fps**2 * (1.0 + cosIncs**2)**2 + Fxs**2 * cosIncs**2)

        return np.sort(thetas)


    def ReadSNRdata(self):

        SNRmassAxis = None
        SNRgrid     = None

        # set sensitivity
        if self.SNRsensitivity == 'design': hdfDatasetName = 'SimNoisePSDaLIGODesignSensitivityP1200087'
        elif self.SNRsensitivity == 'O1'  : hdfDatasetName = 'P1500238_GW150914_H1-GDS-CALIB_STRAIN.txt'
        elif self.SNRsensitivity == 'O3'  : hdfDatasetName = 'SimNoisePSDaLIGOMidHighSensitivityP1200087'
        elif self.SNRsensitivity == 'ET'  : hdfDatasetName = 'ET_D.txt'
        else                              : Stop('SelectionEffects::GetSNRdata(): Unknown sensitivity: {:s}'.format(self.SNRsensitivity))

        # open HDF5 file
        fqFilename = self.SNRfilePath + '/' + self.SNRfileName
        if not os.path.isfile(fqFilename):
            Stop('SelectionEffects::GetSNRdata(): SNR HDF5 file not found: {:s}'.format(fqFilename))
        
        with h5py.File(fqFilename, 'r') as SNRfile:

            grouplist = list(SNRfile.keys())

            if 'mass_axis' not in grouplist:
                Stop('SelectionEffects::GetSNRdata(): Group \'mass_axis\' not found in SNR file {:s}'.format(fqFilename))

            SNRmassAxis = SNRfile['mass_axis'][...]

            if 'snr_values' not in grouplist:
                Stop('SelectionEffects::GetSNRdata(): Group \'snr_values\' not found in SNR file {:s}'.format(fqFilename))

            SNRdatasets = SNRfile['snr_values']

            if hdfDatasetName not in SNRdatasets:
                Stop('SelectionEffects::GetSNRdata(): Group \'{:s}\' for sensitivity \'{:s}\' not found in SNR file {:s}'.format(hdfDatasetName, self.SNRsensitivity, fqFilename))

            SNRgrid = SNRdatasets[hdfDatasetName][...]

        return SNRmassAxis, SNRgrid


    def GetDetectionProbabilityFromSNR(self, p_SNRvalues, p_SNRthreshold = None):

        if p_SNRthreshold is None: p_SNRthreshold = self.SNRthreshold

        thetaMin = p_SNRthreshold / p_SNRvalues
    
        detectionProbability = np.zeros_like(thetaMin)
        detectionProbability[thetaMin <= 1.0] = 1.0 - ((np.digitize(thetaMin[thetaMin <= 1.0], self.SNRthetas) - 1.0) / float(self.SNRthetas.shape[0]))
 
        return detectionProbability


    """
    Compute a grid of SNRs and detection probabilities for a range of masses and SNRs

    These grids are computed to allow for interpolating the values of the snr and detection probability. This function
    combined with find_detection_probability() could be replaced by something like
        for i in range(n_binaries):
            detection_probability = selection_effects.detection_probability(COMPAS.mass1[i],COMPAS.mass2[i],
                                    redshifts, distances, GWdetector_snr_threshold, GWdetector_sensitivity)
    if runtime was not important.

    Args:
        p_SNRsensitivity [string] Which detector sensitivity to use: one of ["design", "O1", "O3"]
        p_SNRthreshold   [float]  What SNR threshold required for a detection
        p_McMax          [float]  Maximum chirp mass in grid
        p_McStep         [float]  Step in chirp mass to use in grid
        p_ETAmax         [float]  Maximum symmetric mass ratio in grid
        p_ETAstep        [float]  Step in symmetric mass ratio to use in grid
        p_SNRmax         [float]  Maximum snr in grid
        p_SNRstep        [float]  Step in snr to use in grid

    Returns:
        SNRgridAt1Mpc               [2D float array] The snr of a binary with masses (Mc, eta) at a distance of 1 Mpc
        detectionProbabilityFromSNR [list of floats] A list of detection probabilities for different SNRs
    """
    def ComputeSNRandDetectionGrids(self, 
                                    p_SNRsensitivity = None,
                                    p_SNRthreshold   = None,
                                    p_McMax          = 300.0,
                                    p_McStep         = 0.1,
                                    p_ETAmax         = 0.25,
                                    p_ETAstep        = 0.01,
                                    p_SNRmax         = 1000.0,
                                    p_SNRstep        = 0.1):

        if p_SNRsensitivity is None: p_SNRsensitivity = self.SNRsensitivity
        if p_SNRthreshold is None  : p_SNRthreshold   = self.SNRthreshold

        # create chirp mass and eta arrays
        Mc  = np.arange(p_McStep, p_McMax + p_McStep, p_McStep)
        eta = np.arange(p_ETAstep, p_ETAmax + p_ETAstep, p_ETAstep)

        # convert to total, primary and secondary mass arrays
        Mt = Mc / eta[:,np.newaxis]**0.6
        M1 = Mt * 0.5 * (1.0 + np.sqrt(1.0 - 4.0 * eta[:,np.newaxis]))
        M2 = Mt - M1

        # interpolate to get snr values if binary was at 1Mpc
        SNRgridAt1Mpc = self.SNRinterpolator(np.log(M1), np.log(M2), grid = False)

        # precompute a grid of detection probabilities as a function of snr
        SNR = np.arange(p_SNRstep, p_SNRmax + p_SNRstep, p_SNRstep)

        detectionProbabilityFromSNR = self.GetDetectionProbabilityFromSNR(SNR, p_SNRthreshold)

        return SNRgridAt1Mpc, detectionProbabilityFromSNR



class COMPAS:       

    # constants

    # Kroupa IMF is a broken power law with three slopes
    # There are two breaks in the Kroupa power law -- they occur here (in solar masses)
    # We define upper and lower bounds (KROUPA_BREAK_3 and KROUPA_BREAK_0 respectively)
    KROUPA_BREAK_0 = 0.01
    KROUPA_BREAK_1 = 0.08
    KROUPA_BREAK_2 = 0.5
    KROUPA_BREAK_3 = 200.0

    # There are three line segments, each with a specific slope
    KROUPA_SLOPE_1 = 0.3
    KROUPA_SLOPE_2 = 1.3
    KROUPA_SLOPE_3 = 2.3


    def __init__(self,
                 p_COMPASfilePath = COMPAS_HDF5_FILE_PATH,
                 p_COMPASfileName = COMPAS_HDF5_FILE_NAME,
                 p_m1Minimum      = M1_MINIMUM,
                 p_m1Maximum      = M1_MAXIMUM,
                 p_m2Minimum      = M2_MINIMUM,
                 p_BinaryFraction = BINARY_FRACTION):

        self.COMPASfilePath       = p_COMPASfilePath
        self.COMPASfileName       = p_COMPASfileName

        self.m1Minimum            = p_m1Minimum
        self.m1Maximum            = p_m1Maximum
        self.m2Minimum            = p_m2Minimum
        self.binaryFraction       = p_BinaryFraction

        self.nSystems             = 0

        self.sysSeeds             = None
        self.zamsST1              = None
        self.zamsST2              = None
        self.zamsMass1            = None
        self.zamsMass2            = None
        self.zamsZ                = None

        self.dcoSeeds             = None
        self.st1                  = None
        self.st2                  = None
        self.mass1                = None
        self.mass2                = None
        self.formationTime        = None
        self.coalescenceTime      = None
        self.mergesInHubbleTime   = None

        self.ceeSeeds             = None
        self.immediateRLOF        = None
        self.optimisticCE         = None

        self.ReadData()

        self.DCOmask              = np.repeat(True, len(self.dcoSeeds))
        self.BBHmask              = np.repeat(True, len(self.dcoSeeds))
        self.BHNSmask             = np.repeat(True, len(self.dcoSeeds))
        self.DNSmask              = np.repeat(True, len(self.dcoSeeds))
        self.CHE_BBHmask          = np.repeat(True, len(self.dcoSeeds))
        self.NON_CHE_BBHmask      = np.repeat(True, len(self.dcoSeeds))
        self.ALL_TYPESmask        = np.repeat(True, len(self.dcoSeeds))
        self.OPTIMISTICmask       = np.repeat(True, len(self.dcoSeeds))

        self.Zsystems             = None
        self.delayTime            = None

        self.massEvolvedPerBinary = None


    def ReadData(self):

        # open HDF5 file
        fqFilename = self.COMPASfilePath + '/' + self.COMPASfileName
        if not os.path.isfile(fqFilename):
            Stop('GetCOMPASdata(): COMPAS HDF5 file not found: {:s}'.format(fqFilename))
        
        with h5py.File(fqFilename, 'r') as COMPASfile:

            groupList = list(COMPASfile.keys())


            # get system parameters info

            if 'BSE_System_Parameters' not in groupList:
                Stop('Group \'BSE_System_Parameters\' not found in COMPAS file {:s}'.format(fqFilename))

            sys = COMPASfile['BSE_System_Parameters']
            datasetList = list(sys.keys())

            if 'SEED' not in datasetList:
                Stop('Dataset \'SEED\' not found in \'BSE_System_Parameters\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Stellar_Type@ZAMS(1)' not in datasetList:
                Stop('Dataset \'Stellar_Type@ZAMS(1)\' not found in \'BSE_System_Parameters\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Stellar_Type@ZAMS(2)' not in datasetList:
                Stop('Dataset \'Stellar_Type@ZAMS(2)\' not found in \'BSE_System_Parameters\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Mass@ZAMS(1)' not in datasetList:
                Stop('Dataset \'Mass@ZAMS(1)\' not found in \'BSE_System_Parameters\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Mass@ZAMS(2)' not in datasetList:
                Stop('Dataset \'Mass@ZAMS(2)\' not found in \'BSE_System_Parameters\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Metallicity@ZAMS(1)' not in datasetList:
                Stop('Dataset \'Metallicity@ZAMS(1)\' not found in \'BSE_System_Parameters\' group in COMPAS file {:s}'.format(fqFilename))

            self.sysSeeds  = sys['SEED'][...]
            self.nSystems  = len(self.sysSeeds)
            self.zamsST1   = sys['Stellar_Type@ZAMS(1)'][...]
            self.zamsST2   = sys['Stellar_Type@ZAMS(2)'][...]
            self.zamsMass1 = sys['Mass@ZAMS(1)'][...]
            self.zamsMass2 = sys['Mass@ZAMS(2)'][...]
            self.zamsZ     = sys['Metallicity@ZAMS(1)'][...]

            # CHE info may not be available - just disable CHE DCO types if not
            
            if 'CH_on_MS(1)' in datasetList:
                self.CHEonMS1 = sys['CH_on_MS(1)'][...].astype(bool)
            else:
                print('Dataset \'CH_on_MS(1)\' not found in \'BSE_System_Parameters\' group in COMPAS file {:s}.  DCO types \'CHE_BBH\' and \'NON_CHE_BBH\' not available'.format(fqFilename))
                self.CHEonMS1 = None
            
            if 'CH_on_MS(2)' in datasetList:
                self.CHEonMS2 = sys['CH_on_MS(2)'][...].astype(bool)
            else:
                print('Dataset \'CH_on_MS(2)\' not found in \'BSE_System_Parameters\' group in COMPAS file {:s}.  DCO types \'CHE_BBH\' and \'NON_CHE_BBH\' not available'.format(fqFilename))
                self.CHEonMS2 = None


            # get double compact objects info

            if 'BSE_Double_Compact_Objects' not in groupList:
                Stop('Group \'BSE_Double_Compact_Objects\' not found in COMPAS file {:s}'.format(fqFilename))

            dco = COMPASfile['BSE_Double_Compact_Objects']
            datasetList = list(dco.keys())

            if 'SEED' not in datasetList:
                Stop('Dataset \'SEED\' not found in \'BSE_Double_Compact_Objects\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Stellar_Type(1)' not in datasetList:
                Stop('Dataset \'Stellar_Type(1)\' not found in \'BSE_Double_Compact_Objects\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Stellar_Type(2)' not in datasetList:
                Stop('Dataset \'Stellar_Type(2)\' not found in \'BSE_Double_Compact_Objects\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Mass(1)' not in datasetList:
                Stop('Dataset \'Mass(1)\' not found in \'BSE_Double_Compact_Objects\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Mass(2)' not in datasetList:
                Stop('Dataset \'Mass(2)\' not found in \'BSE_Double_Compact_Objects\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Time' not in datasetList:
                Stop('Dataset \'Time\' not found in \'BSE_Double_Compact_Objects\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Coalescence_Time' not in datasetList:
                Stop('Dataset \'Coalescence_Time\' not found in \'BSE_Double_Compact_Objects\' group in COMPAS file {:s}'.format(fqFilename))
            if 'Merges_Hubble_Time' not in datasetList:
                Stop('Dataset \'Merges_Hubble_Time\' not found in \'BSE_Double_Compact_Objects\' group in COMPAS file {:s}'.format(fqFilename))

            self.dcoSeeds           = dco['SEED'][...]
            self.st1                = dco['Stellar_Type(1)'][...]
            self.st2                = dco['Stellar_Type(2)'][...]
            self.mass1              = dco['Mass(1)'][...]
            self.mass2              = dco['Mass(2)'][...]
            self.formationTime      = dco['Time'][...]
            self.coalescenceTime    = dco['Coalescence_Time'][...]
            self.mergesInHubbleTime = dco['Merges_Hubble_Time'][...].astype(bool)


            # get common envelopes info

            if 'BSE_Common_Envelopes' not in groupList:
                Stop('Group \'BSE_Common_Envelopes\' not found in COMPAS file {:s}'.format(fqFilename))

            cee = COMPASfile['BSE_Common_Envelopes']
            datasetList = list(cee.keys())

            if 'SEED' in datasetList:
                self.ceeSeeds = cee['SEED'][...]
            else:
                print('Dataset \'SEED\' not found in \'BSE_Common_Envelopes\' group in COMPAS file {:s}.  DCO masks for common envelopes not available'.format(fqFilename))
                self.ceeSeeds = None

            if 'Immediate_RLOF>CE' in datasetList:
                self.immediateRLOF = cee['Immediate_RLOF>CE'][...].astype(bool)
            else:
                print('Dataset \'Immediate_RLOF>CE\' not found in \'BSE_Common_Envelopes\' group in COMPAS file {:s}.  DCO masks for common envelopes not available'.format(fqFilename))
                self.immediateRLOF = None

            if 'Optimistic_CE' in datasetList:
                self.optimisticCE = cee['Optimistic_CE'][...].astype(bool)
            else:
                print('Dataset \'Optimistic_CE\' not found in \'BSE_Common_Envelopes\' group in COMPAS file {:s}.  DCO masks for common envelopes not available'.format(fqFilename))
                self.optimisticCE = None


    def SetDCOmasks(self, p_DCOtypes = 'BBH', p_WithinHubbleTime = True, p_Pessimistic = True, p_NoRLOFafterCEE = True):

        # set all DCO type masks
        
        typeMasks = {
            'ALL' : np.repeat(True, len(self.dcoSeeds)),
            'BBH' : np.logical_and(self.st1 == 14, self.st2 == 14),
            'BHNS': np.logical_or(np.logical_and(self.st1  == 14, self.st2 == 13), np.logical_and(self.st1  == 13, self.st2 == 14)),
            'BNS' : np.logical_and(self.st1  == 13, self.st2 == 13),
        }
        typeMasks['CHE_BBH']     = np.repeat(False, len(self.dcoSeeds)) # for now - updated below
        typeMasks['NON_CHE_BBH'] = np.repeat(True, len(self.dcoSeeds))  # for now - updated below


        # set CHE type masks if required and if able

        if p_DCOtypes == 'CHE_BBH' or p_DCOtypes == 'NON_CHE_BBH':          # if required
            if self.CHonMS1 is not None and self.CHonMS2 is not None:       # if able
                mask     = np.logical_and.reduce((self.zamsST1 == 16, self.zamsST2 == 16, self.CHonMS1 == True, self.CHonMS2 == True))
                cheSeeds = self.sysSeeds[...][mask]
                mask     = np.in1d(self.dcoSeeds, cheSeeds)

                if p_DCOtypes == 'CHE_BBH'    : typeMasks['CHE_BBH']     = np.logical_and(mask, type_masks['BBH'])
                if p_DCOtypes == 'NON_CHE_BBH': typeMasks['NON_CHE_BBH'] = np.logical_and(np.logical_not(mask), type_masks['BBH'])


        # set merges in a hubble time mask

        hubbleMask = self.mergesInHubbleTime if p_WithinHubbleTime else np.repeat(True, len(self.dcoSeeds))


        # set no RLOF after CE and optimistic/pessimistic CE masks if required

        rlofMask = np.repeat(True, len(self.dcoSeeds))
        pessimisticMask = np.repeat(True, len(self.dcoSeeds))
        if p_NoRLOFafterCEE or p_Pessimistic:                               # if required
            if self.ceeSeeds is not None and self.immediateRLOF is not None and self.optimisticCE is not None: # if able

                dcoFromCE = np.in1d(self.ceeSeeds, self.dcoSeeds)
                dcoCEseeds = self.ceeSeeds[dcoFromCE]


                # set no RLOF after CE if required

                if p_NoRLOFafterCEE:                                            # if required
                    rlofSeeds = np.unique(dcoCEseeds[self.immediateRLOF[dcoFromCE]])
                    rlofMask = np.logical_not(np.in1d(self.dcoSeeds, rlofSeeds))
                else:
                    rlofMask = np.repeat(True, len(self.dcoSeeds))


                # set pessimistic/optimistic CE mask if required

                if p_Pessimistic:                                               # if required
                    pessimisticSeeds = np.unique(dcoCEseeds[self.optimisticCE[dcoFromCE]])
                    pessimisticMask = np.logical_not(np.in1d(self.dcoSeeds, pessimisticSeeds))
                else:
                    pessimisticMask = np.repeat(True, len(self.dcoSeeds))


        # set class member variables for all DCO masks

        self.DCOmask         = typeMasks[p_DCOtypes]    * hubbleMask * rlofMask * pessimisticMask
        self.BBHmask         = typeMasks['BBH']         * hubbleMask * rlofMask * pessimisticMask
        self.BHNSmask        = typeMasks['BHNS']        * hubbleMask * rlofMask * pessimisticMask
        self.DNSmask         = typeMasks['BNS']         * hubbleMask * rlofMask * pessimisticMask
        self.CHE_BBHmask     = typeMasks['CHE_BBH']     * hubbleMask * rlofMask * pessimisticMask
        self.NON_CHE_BBHmask = typeMasks['NON_CHE_BBH'] * hubbleMask * rlofMask * pessimisticMask
        self.ALL_TYPESmask   = typeMasks['ALL']         * hubbleMask * rlofMask * pessimisticMask
        self.OPTIMISTICmask  = pessimisticMask


    def CalculatePopulationValues(self):

        self.dcoSeeds            = self.dcoSeeds[self.DCOmask]

        self.mass1                = self.mass1[self.DCOmask]
        self.mass2                = self.mass2[self.DCOmask]
        self.formationTime        = self.formationTime[self.DCOmask]
        self.coalescenceTime      = self.coalescenceTime[self.DCOmask]
        self.mergesInHubbleTime   = self.mergesInHubbleTime[self.DCOmask]

        self.Zsystems             = self.zamsZ[np.in1d(self.sysSeeds, self.dcoSeeds)]
        self.delayTime            = np.add(self.formationTime, self.coalescenceTime)

        self.massEvolvedPerBinary = self.CalculateStarFormingMassPerBinary(p_Samples        = 20000000, 
                                                                           p_m1Minimum      = self.m1Minimum, 
                                                                           p_m1Maximum      = self.m1Maximum, 
                                                                           p_m2Minimum      = self.m2Minimum, 
                                                                           p_BinaryFraction = self.binaryFraction)


    """
    Calculate the fraction of stellar mass between 0 and m for a three part broken power law.
    Default values follow Kroupa (2001)
        F(m) ~ int_0^m zeta(m) dm
        
    Args:
        p_Masses [float, list of floats] mass or masses at which to evaluate
        p_Mi     [float]                 masses at which to transition the slope
        p_Sij    [float]                 slope of the IMF between mi and mj
            
    Returns:
        CDF(m) [float/list of floats] value or values of the IMF
    """
    def CDF_IMF(self, 
                p_Masses, 
                p_M1  = KROUPA_BREAK_0, 
                p_M2  = KROUPA_BREAK_1, 
                p_M3  = KROUPA_BREAK_2, 
                p_M4  = KROUPA_BREAK_3, 
                p_S12 = KROUPA_SLOPE_1, 
                p_S23 = KROUPA_SLOPE_2, 
                p_S34 = KROUPA_SLOPE_3):

        # calculate normalisation constants that ensure the IMF is continuous
        b1 = 1.0 / ((p_M2**(1 - p_S12) - p_M1**(1 - p_S12)) / (1 - p_S12) \
                   + p_M2**(-(p_S12 - p_S23)) * (p_M3**(1 - p_S23) - p_M2**(1 - p_S23)) / (1 - p_S23) \
                   + p_M2**(-(p_S12 - p_S23)) * p_M3**(-(p_S23 - p_S34)) * (p_M4**(1 - p_S34) - p_M3**(1 - p_S34)) / (1 - p_S34))
        b2 = b1 * p_M2**(-(p_S12 - p_S23))
        b3 = b2 * p_M3**(-(p_S23 - p_S34))

        if isinstance(p_Masses, float):
            if p_Masses <= p_M1:   CDF = 0.0
            elif p_Masses <= p_M2: CDF = b1 / (1 - p_S12) * (p_Masses**(1 - p_S12) - p_M1**(1 - p_S12))
            elif p_Masses <= p_M3: CDF = self.CDF_IMF(p_M2) + b2 / (1 - p_S23) * (p_Masses**(1 - p_S23) - p_M2**(1 - p_S23))
            elif p_Masses <= p_M4: CDF = self.CDF_IMF(p_M3) + b3 / (1 - p_S34) * (p_Masses**(1 - p_S34) - p_M3**(1 - p_S34))
            else:                  CDF = 0.0
        else:
            CDF = np.zeros(len(p_Masses))
            CDF[np.logical_and(p_Masses >= p_M1, p_Masses < p_M2)] = b1 / (1 - p_S12) * (p_Masses[np.logical_and(p_Masses >= p_M1, p_Masses < p_M2)]**(1 - p_S12) - p_M1**(1 - p_S12))
            CDF[np.logical_and(p_Masses >= p_M2, p_Masses < p_M3)] = self.CDF_IMF(p_M2) + b2 / (1 - p_S23) * (p_Masses[np.logical_and(p_Masses >= p_M2, p_Masses < p_M3)]**(1 - p_S23) - p_M2**(1 - p_S23))
            CDF[np.logical_and(p_Masses >= p_M3, p_Masses < p_M4)] = self.CDF_IMF(p_M3) + b3 / (1 - p_S34) * (p_Masses[np.logical_and(p_Masses >= p_M3, p_Masses < p_M4)]**(1 - p_S34) - p_M3**(1 - p_S34))
            CDF[p_Masses >= p_M4] = np.ones(len(p_Masses[p_Masses >= p_M4]))
    
        return CDF


    """ 
    Calculate the inverse CDF for a three part broken power law.
    Default values follow Kroupa (2001)
        
    Args:
        p_Samples [float, list of floats] A uniform random variable on [0, 1]
        p_Mi      [float]                 masses at which to transition the slope
        p_Sij     [float]                 slope of the IMF between mi and mj
            
    Returns:
        masses(m) [list of floats] values of m
    """
    def SampleInitialMass(self, 
                          p_Samples, 
                          p_M1  = KROUPA_BREAK_0, 
                          p_M2  = KROUPA_BREAK_1, 
                          p_M3  = KROUPA_BREAK_2, 
                          p_M4  = KROUPA_BREAK_3, 
                          p_S12 = KROUPA_SLOPE_1, 
                          p_S23 = KROUPA_SLOPE_2, 
                          p_S34 = KROUPA_SLOPE_3):
        # calculate normalisation constants that ensure the IMF is continuous
        b1 = 1.0 / ((p_M2**(1 - p_S12) - p_M1**(1 - p_S12)) / (1 - p_S12) \
                   + p_M2**(-(p_S12 - p_S23)) * (p_M3**(1 - p_S23) - p_M2**(1 - p_S23)) / (1 - p_S23) \
                   + p_M2**(-(p_S12 - p_S23)) * p_M3**(-(p_S23 - p_S34)) * (p_M4**(1 - p_S34) - p_M3**(1 - p_S34)) / (1 - p_S34))
        b2 = b1 * p_M2**(-(p_S12 - p_S23))
        b3 = b2 * p_M3**(-(p_S23 - p_S34))

        # find the probabilities at which the gradient changes
        F1, F2, F3, F4 = self.CDF_IMF(np.array([p_M1, p_M2, p_M3, p_M4]), p_M1 = p_M1, p_M2 = p_M2, p_M3 = p_M3, p_M4 = p_M4, p_S12 = p_S12, p_S23 = p_S23, p_S34 = p_S34)

        masses = np.zeros(len(p_Samples))
        masses[np.logical_and(p_Samples > F1, p_Samples <= F2)] = np.power((1 - p_S12) / b1 * (p_Samples[np.logical_and(p_Samples > F1, p_Samples <= F2)] - F1) + p_M1**(1 - p_S12), 1 / (1 - p_S12))
        masses[np.logical_and(p_Samples > F2, p_Samples <= F3)] = np.power((1 - p_S23) / b2 * (p_Samples[np.logical_and(p_Samples > F2, p_Samples <= F3)] - F2) + p_M2**(1 - p_S23), 1 / (1 - p_S23))
        masses[np.logical_and(p_Samples > F3, p_Samples <= F4)] = np.power((1 - p_S34) / b3 * (p_Samples[np.logical_and(p_Samples > F3, p_Samples <= F4)] - F3) + p_M3**(1 - p_S34), 1 / (1 - p_S34))
    
        return masses


    def CalculateStarFormingMassPerBinary(self, 
                                          p_Samples        = 20000000,
                                          p_m1Minimum      = M1_MINIMUM,
                                          p_m1Maximum      = M1_MAXIMUM,
                                          p_m2Minimum      = M2_MINIMUM,
                                          p_BinaryFraction = BINARY_FRACTION):

        primaryMass      = self.SampleInitialMass(np.random.rand(p_Samples))                        # primary mass samples
        totalPrimaryMass = np.sum(primaryMass)                                                      # total mass of all primaries     

        q = np.random.rand(p_Samples)                                                               # random mass ratios 0 <= q < 1

        mask                = np.zeros(p_Samples) < p_BinaryFraction                                # only p_BinaryFraction stars have a companion
        secondaryMass       = np.zeros(p_Samples)                                                   # sampled secondary masses
        secondaryMass[mask] = primaryMass[mask] * q[mask]                                           # mask for binaries
        totalSecondaryMass  = np.sum(secondaryMass)                                                 # total mass of all secondaries

        totalMass = totalPrimaryMass + totalSecondaryMass;                                          # total population mass

        mask1 = np.logical_and(primaryMass >= p_m1Minimum, primaryMass <= p_m1Maximum)              # mask primary mass for COMPAS mass range
        mask2 = secondaryMass >= p_m2Minimum                                                        # mask secondary mass for COMPAS mass range
        mask  = np.logical_and(mask1, mask2)                                                        # combined mask for COMPAS mass range
        
        totalMassCOMPAS = np.sum(primaryMass[mask]) + np.sum(secondaryMass[mask])                   # total COMPAS mass

        fractionMassCOMPAS = totalMassCOMPAS / totalMass                                            # fraction of total mass sampled by COMPAS
        averageMassCOMPAS  = totalMassCOMPAS / float(np.sum(mask))                                  # average stellar mass sampled by COMPAS
        
        return averageMassCOMPAS / fractionMassCOMPAS                                               # COMPAS average star forming mass evolved per binary in the Universe


class CosmicIntegration:




    def __init__(self, 
                 p_COMPAS,
                 p_SE,
                 p_MaxRedshift          = MAX_FORMATION_REDSHIFT, 
                 p_MaxRedshiftDetection = MAX_FORMATION_REDSHIFT, 
                 p_RedshiftStep         = REDSHIFT_STEP):

        self.compas               = p_COMPAS
        self.SE                   = p_SE

        self.maxRedshift          = p_MaxRedshift 
        self.maxRedshiftDetection = p_MaxRedshiftDetection
        self.redshiftStep         = p_RedshiftStep

        self.redshifts            = None
        self.nRedshiftsDetection  = None 
        self.times                = None
        self.distances            = None
        self.shellVolumes         = None


    """ 
    Given limits on the redshift, create an array of redshifts, times, distances and volumes

    Args:
        p_COMPAS               [instance] COMPAS instance
        p_SE                   [instance] SelectionEffects instance
        p_MaxRedshift          [float]    Maximum redshift to use in array
        p_MaxRedshiftDetection [float]    Maximum redshift to calculate detection rates (must be <= max_redshift)
        p_RedshiftStep         [float]    size of step to take in redshift

    Returns:
        redshifts           [list of floats] List of redshifts between limits supplied
        nRedshiftsDetection [int]            Number of redshifts in list that should be used to calculate detection rates
        times               [list of floats] Equivalent of redshifts but converted to age of Universe
        distances           [list of floats] Equivalent of redshifts but converted to luminosity distances
        shellVolumes        [list of floats] Equivalent of redshifts but converted to shell volumes
    """
    def CalculateRedshiftRelatedParams(self, p_MaxRedshift = None, p_MaxRedshiftDetection = None, p_RedshiftStep = None):

        if p_MaxRedshift          is None: p_MaxRedshift          = self.maxRedshift
        if p_MaxRedshiftDetection is None: p_MaxRedshiftDetection = self.maxRedshiftDetection
        if p_RedshiftStep         is None: p_RedshiftStep         = self.redshiftStep

        # create a list of redshifts and record lengths
        self.redshifts           = np.arange(0.0, p_MaxRedshift + p_RedshiftStep, p_RedshiftStep)
        self.nRedshiftsDetection = int(p_MaxRedshiftDetection / p_RedshiftStep)

        # convert redshifts to times and ensure all times are in Myr
        self.times = cosmology.age(self.redshifts).to(units.Myr).value

        # convert redshifts to distances and ensure all distances are in Mpc (also avoid D=0 because division by 0)
        self.distances    = cosmology.luminosity_distance(self.redshifts).to(units.Mpc).value
        self.distances[0] = 0.001

        # convert redshifts to volumnes and ensure all volumes are in Gpc^3
        self.volumes = cosmology.comoving_volume(self.redshifts).to(units.Gpc**3).value

        # split volumes into shells and duplicate last shell to keep same length
        self.shellVolumes = np.diff(self.volumes)
        self.shellVolumes = np.append(self.shellVolumes, self.shellVolumes[-1])

        return self.redshifts, self.nRedshiftsDetection, self.times, self.distances, self.shellVolumes


    """
    Calculate the star forming mass per unit volume per year using
    Neijssel+19 Eq. 6

    Args:
        p_Redshifts [list of floats] List of redshifts at which to evaluate the sfr
        p_SFRa      [float]          SFR a
        p_SFRd      [float]          SFR d

    Returns:
        sfr [list of floats] Star forming mass per unit volume per year for each redshift
    """
    def CalculateSFR(self, p_Redshifts = None, p_SFRa = NEIJSSEL_SFR_A, p_SFRd = NEIJSSEL_SFR_D):

        if p_Redshifts is None: p_Redshifts = self.redshifts

        # use Neijssel+19 to get value in mass per year per cubic Mpc and convert to per cubic Gpc then return
        sfr = p_SFRa * ((1.0 + p_Redshifts)**2.77) / (1.0 + ((1.0 + p_Redshifts)/2.9)**p_SFRd) * units.Msun / units.yr / units.Mpc**3
        return sfr.to(units.Msun / units.yr / units.Gpc**3).value


    """
    Calculate the distribution of metallicities at different redshifts using a log skew normal distribution
    the log-normal distribution is a special case of this log skew normal distribution distribution, and is retrieved by setting 
    the skewness to zero (alpha = 0). 
    Based on the method in Neijssel+19. Default values of mu0=0.035, muz=-0.23, sigma_0=0.39, sigma_z=0.0, alpha =0.0, 
    retrieve the dP/dZ distribution used in Neijssel+19

    NOTE: This assumes that metallicities in COMPAS are drawn from a flat in log distribution!

        NOTE: This assumes that metallicities in COMPAS are drawn from a flat in log distribution

        Args:
            p_Redshifts      [list of floats] List of redshifts at which to calculate things
            p_MinLogZ_COMPAS [float]          Minimum logZ value that COMPAS samples
            p_MaxLogZ_COMPAS [float]          Maximum logZ value that COMPAS samples
            p_MinLogZ        [float]          Minimum logZ at which to calculate dPdlogZ (influences normalization) 
            p_MaxLogZ        [float]          Maximum logZ at which to calculate dPdlogZ (influences normalization)
            p_Z0             [float]          Parameter used for calculating mean metallicity
            p_Alpha          [float]          Parameter used for calculating mean metallicity
            p_Sigma          [float]          Parameter used for calculating mean metallicity
            p_zScale         [float]          Redshift scaling of the scale (variance in normal, 0 in Neijssel+19)
            p_Skew           [float]          Shape (skewness, p_Skew = 0 retrieves normal dist as in Neijssel+19)
            p_StepLogZ       [float]          Size of logZ steps to take in finding a Z range

        Returns:
            dPdlogZ [2D float array] Probability of getting a particular logZ at a certain redshift
            Z       [list of floats] Metallicities at which dPdlogZ is evaluated
            pDrawZ  [float]          Probability of drawing a certain metallicity in COMPAS (float because assuming uniform)
    """
    def FindZdistribution(self, 
                          p_Redshifts,
                          p_MinLogZ_COMPAS,
                          p_MaxLogZ_COMPAS,
                          p_MinLogZ  = MINIMUM_LOG_Z,
                          p_MaxLogZ  = MAXIMUM_LOG_Z,
                          p_Z0       = NEIJSSEL_Z0, 
                          p_Alpha    = NEIJSSEL_ALPHA, 
                          p_Sigma    = NEIJSSEL_SIGMA,
                          p_zScale   = Z_SCALE,
                          p_Skew     = SKEW, 
                          p_StepLogZ = LOG_Z_STEP):

        sigma = p_Sigma * 10**(p_zScale * p_Redshifts)     # Log-Linear redshift dependence of sigma
        meanZ = p_Z0 * 10**(p_Alpha * p_Redshifts)         # Follow Langer & Norman 2007? in assuming that mean metallicities evolve in z
        
        # Now we re-write the expected value of our log-skew-normal to retrieve mu
        beta = p_Skew / (np.sqrt(1.0 + (p_Skew * p_Skew)))
        phi  = NormDist.cdf(beta * sigma) 
        muZ  = np.log(meanZ / 2.0 * 1.0 / (np.exp(0.5 * (sigma * sigma)) * phi)) 

        # create a range of metallicities (the x-values, or random variables)
        logZ = np.arange(p_MinLogZ, p_MaxLogZ + p_StepLogZ, p_StepLogZ)
        Z    = np.exp(logZ)

        # probabilities of log-skew-normal (without the factor of 1/Z since this is dp/dlogZ not dp/dZ)
        normPDF = NormDist.pdf((logZ - muZ[:,np.newaxis]) / sigma[:,np.newaxis])
        normCDF = NormDist.cdf(p_Skew * (logZ - muZ[:,np.newaxis]) / sigma[:,np.newaxis])
        dPdlogZ = 2.0 / (sigma[:,np.newaxis]) * normPDF * normCDF

        # normalise the distribution over all metallicities
        norm    = dPdlogZ.sum(axis = -1) * p_StepLogZ
        dPdlogZ = dPdlogZ / norm[:,np.newaxis]

        # assume a flat in log distribution in metallicity to find probability of drawing Z in COMPAS
        pDrawZ = 1.0 / (p_MaxLogZ_COMPAS - p_MinLogZ_COMPAS)
    
        return dPdlogZ, Z, pDrawZ


    """
    Find both the formation and merger rates for each binary at each redshift

    Args:
        p_nBinaries        [int]            Number of DCO binaries in the arrays
        p_Redshifts        [list of floats] Redshifts at which to evaluate the rates
        p_Times            [list of floats] Equivalent of the redshifts in terms of age of the Universe
        p_nFormed          [float]          Binary formation rate (number of binaries formed per year per cubic Gpc) represented by each simulated COMPAS binary
        p_dPdlogZ          [2D float array] Probability of getting a particular logZ at a certain redshift
        p_Z                [list of floats] Metallicities at which dPdlogZ is evaluated
        p_pDrawZ           [float]          Probability of drawing a certain metallicity in COMPAS (float because assuming uniform)
        p_COMPAS_Z         [list of floats] Metallicity of each binary in COMPAS data
        p_COMPASdelayTimes [list of floats] Delay time of each binary in COMPAS data

    Returns:
        formationRate [2D float array] Formation rate for each binary at each redshift
        mergerRate    [2D float array] Merger rate for each binary at each redshift
    """
    def FindFormationAndMergerRates(self, p_nBinaries, p_Redshifts, p_Times, p_nFormed, p_dPdlogZ, p_Z, p_pDrawZ, p_COMPAS_Z, p_COMPASdelayTimes):

        # initalise rates to zero
        nRedshifts    = len(p_Redshifts)
        redshiftStep  = p_Redshifts[1] - p_Redshifts[0]
        formationRate = np.zeros(shape=(p_nBinaries, nRedshifts))
        mergerRate    = np.zeros(shape=(p_nBinaries, nRedshifts))

        # interpolate times and redshifts for conversion
        timesToRedshifts = interp1d(p_Times, p_Redshifts)

        # make note of the first time at which star formation occured
        ageFirstSFR = np.min(p_Times)

        # go through each binary in the COMPAS data
        for i in range(p_nBinaries):
            # calculate formation rate (see Neijssel+19 Section 4) - note this uses p_dPdlogZ for *closest* metallicity

            formationRate[i, :] = p_nFormed * p_dPdlogZ[:, np.digitize(p_COMPAS_Z[i], p_Z)] / p_pDrawZ

            # calculate the time at which the binary formed if it merges at this redshift
            formationTime = p_Times - p_COMPASdelayTimes[i]

            # we have only calculated formation rate up to z=max(p_Redshifts), so we need to only find merger rates for formation times at z<max(p_Redshifts)
            # first locate the index above which the binary would have formed before z=max(p_Redshifts)
            firstTooEarlyIndex = np.digitize(ageFirstSFR, formationTime)

            # include the whole array if digitize returns end of array and subtract one so we don't include the time past the limit
            firstTooEarlyIndex = firstTooEarlyIndex + 1 if firstTooEarlyIndex == nRedshifts else firstTooEarlyIndex

            # as long as that doesn't preclude the whole range
            if firstTooEarlyIndex > 0:
                # work out the redshift at the time of formation
                zOfFormation = timesToRedshifts(formationTime[:firstTooEarlyIndex - 1])
                # calculate which index in the redshift array these redshifts correspond to
                zOfFormationIndex = np.ceil(zOfFormation / redshiftStep).astype(int) if nRedshifts > 1 else 0
                # set the merger rate at z (with z<10) to the formation rate at z_form
                mergerRate[i, :firstTooEarlyIndex - 1] = formationRate[i, zOfFormationIndex]

        return formationRate, mergerRate


    """
    Compute the detection probability given a grid of SNRs and detection probabilities with masses

    Args:
        p_Mc                          [list of floats] Chirp mass of binaries in COMPAS
        p_ETA                         [list of floats] Symmetric mass ratios of binaries in COMPAS
        p_Redshifts                   [list of floats] List of redshifts
        p_Distances                   [list of floats] List of distances corresponding to redshifts
        p_nRedshiftsDetection         [int]            Index (in redshifts) to which we evaluate detection probability
        p_nBinaries                   [int]            Number of merging binaries in the COMPAS file
        p_SNRgridAt1Mpc               [2D float array] The snr of a binary with masses (Mc, eta) at a distance of 1 Mpc
        p_DetectionProbabilityFromSNR [list of floats] A list of detection probabilities for different SNRs
        p_McStep                      [float]          Step in chirp mass to use in grid (default 0.1)
        p_ETAstep                     [float]          Step in symmetric mass ratio to use in grid (default 0.01)
        p_SNRstep                     [float]          Step in snr to use in grid (default 0.1)

    Returns:
        detectionProbability [2D float array] Detection probabilities
    """
    def FindDetectionProbability(self,
                                 p_Mc,
                                 p_ETA,
                                 p_Redshifts,
                                 p_Distances,
                                 p_nRedshiftsDetection,
                                 p_nBinaries,
                                 p_SNRgridAt1Mpc,
                                 p_DetectionProbabilityFromSNR,
                                 p_McStep  = Mc_STEP,
                                 p_ETAstep = ETA_STEP,
                                 p_SNRstep = SNR_STEP):

        # by default, set detection probability to one
        detectionProbability = np.ones(shape=(p_nBinaries, p_nRedshiftsDetection))

        # for each binary in the COMPAS file
        for i in range(p_nBinaries):
            # shift frames for the chirp mass
            McShifted = p_Mc[i] * (1 + p_Redshifts[:p_nRedshiftsDetection])

            # work out the closest index to the given values of eta and Mc
            etaIndex = np.round(p_ETA[i] / p_ETAstep).astype(int) - 1
            McIndex  = np.round(McShifted / p_McStep).astype(int) - 1

            # lookup values for the snr (but make sure you don't go over the top of the array)
            SNRs             = np.ones(p_nRedshiftsDetection) * 0.00001
            McBelowMax       = McIndex < p_SNRgridAt1Mpc.shape[1]
            SNRs[McBelowMax] = p_SNRgridAt1Mpc[etaIndex, McIndex[McBelowMax]]

            # convert these snr values to the correct distances
            SNRs = SNRs / p_Distances[:p_nRedshiftsDetection]

            # lookup values for the detection probability (but make sure you don't go over the top of the array)
            detectionListIndex = np.round(SNRs / p_SNRstep).astype(int) - 1
            SNRbelowMax = detectionListIndex < len(p_DetectionProbabilityFromSNR)
            SNRbelowMin = detectionListIndex < 0

            # remember we set probability = 1 by default? Because if we don't set it here, we have snr > max snr
            # which is 1000 by default, meaning very detectable
            detectionProbability[i, SNRbelowMax] = p_DetectionProbabilityFromSNR[detectionListIndex[SNRbelowMax]]
            #on the other hand, if SNR is too low, the detection probability is effectively zero
            detectionProbability[i, SNRbelowMin] = 0

        return detectionProbability


    """
    Find the detection rate, formation rate and merger rate for each binary in a COMPAS file at a series of redshifts
    defined by intput. Also returns relevant COMPAS data.

    NOTE: This code assumes that assumes that metallicities in COMPAS are drawn from a flat in log distribution

    Args:
        p_MaxRedshift          [float] Maximum redshift to use in array
        p_MaxRedshiftDetection [float] Maximum redshift to calculate detection rates (must be <= max_redshift)
        p_RedshiftStep         [float] Size of step to take in redshift
        p_M1min                [float] Minimum primary mass sampled by COMPAS
        p_M1max                [float] Maximum primary mass sampled by COMPAS
        p_M2min                [float] Minimum secondary mass sampled by COMPAS
        p_BinaryFraction       [float] Binary fraction used by COMPAS
        p_Z0                   [float] Parameter used for calculating mean metallicity (see Neijssel+19 Eq.7-9)
        p_Alpha                [float] Parameter used for calculating mean metallicity (see Neijssel+19 Eq.7-9)
        p_Sigma                [float] Parameter used for calculating mean metallicity (see Neijssel+19 Eq.7-9)
        p_SFRa                 [float] Parameter used for calculating mean metallicity (see Neijssel+19 Eq.7-9)
        p_SFRd                 [float] Parameter used for calculating mean metallicity (see Neijssel+19 Eq.7-9)
        p_StepLogZ             [float] Size of logZ steps to take in finding a Z range
        p_McStep               [float] Step in chirp mass to use in grid (default 0.1)
        p_ETAstep              [float] Step in symmetric mass ratio to use in grid (default 0.01)
        p_SNRstep              [float] Step in snr to use in grid (default 0.1)

    Returns:
        detectionRate [2D float array] Detection rate for each binary at each redshift in 1/yr
        chirpMasses   [float array]    Chirpmasses
    """

    def FindDetectionRate(self, 
                          p_MaxRedshift          = None, 
                          p_MaxRedshiftDetection = None, 
                          p_RedshiftStep         = None,
                          p_M1min                = M1_MINIMUM, 
                          p_M1max                = M1_MAXIMUM, 
                          p_M2min                = M2_MINIMUM, 
                          p_BinaryFraction       = BINARY_FRACTION,
                          p_Z0                   = NEIJSSEL_Z0,                        
                          p_Alpha                = NEIJSSEL_ALPHA, 
                          p_Sigma                = NEIJSSEL_SIGMA,
                          p_SFRa                 = NEIJSSEL_SFR_A,
                          p_SFRd                 = NEIJSSEL_SFR_D,
                          p_StepLogZ             = LOG_Z_STEP,
                          p_McStep               = Mc_STEP,
                          p_ETAstep              = ETA_STEP,
                          p_SNRstep              = SNR_STEP):


        if p_MaxRedshift          is None: p_MaxRedshift          = self.maxRedshift
        if p_MaxRedshiftDetection is None: p_MaxRedshiftDetection = self.maxRedshiftDetection
        if p_RedshiftStep         is None: p_RedshiftStep         = self.redshiftStep


        # compute the chirp masses and symmetric mass ratios only for systems of interest
        chirpMasses = (self.compas.mass1 * self.compas.mass2)**(3.0 / 5.0) / (self.compas.mass1 + self.compas.mass2)**(1.0 / 5.0)    
        ETAs        = self.compas.mass1 * self.compas.mass2 / (self.compas.mass1 + self.compas.mass2)**2
        nBinaries   = len(chirpMasses)

        # calculate the redshifts array and its equivalents
        self.CalculateRedshiftRelatedParams(p_MaxRedshift = p_MaxRedshift, p_MaxRedshiftDetection = p_MaxRedshiftDetection, p_RedshiftStep = p_RedshiftStep)

        # find the star forming mass per year per Gpc^3 and convert to total number formed per year per Gpc^3
        sfr     = self.CalculateSFR(self.redshifts, p_SFRa, p_SFRd)
        nFormed = sfr / (self.compas.massEvolvedPerBinary * self.compas.nSystems)

        # work out the metallicity distribution at each redshift and probability of drawing each metallicity in COMPAS
        dPdlogZ, Z, pDrawZ = self.FindZdistribution(self.redshifts, np.log(np.min(self.compas.zamsZ)), np.log(np.max(self.compas.zamsZ)), p_Z0 = p_Z0, p_Alpha = p_Alpha, p_Sigma = p_Sigma, p_StepLogZ = p_StepLogZ)

        # calculate the formation and merger rates using what we computed above
        formationRate, mergerRate = self.FindFormationAndMergerRates(nBinaries, self.redshifts, self.times, nFormed, dPdlogZ, Z, pDrawZ, self.compas.Zsystems, self.compas.delayTime)

        # use lookup tables to find the probability of detecting each binary at each redshift
        detectionProbability = self.FindDetectionProbability(chirpMasses, ETAs, self.redshifts, self.distances, self.nRedshiftsDetection, nBinaries, self.SE.SNRgridAt1Mpc, self.SE.detectionProbabilityFromSNR, p_McStep, p_ETAstep, p_SNRstep)

        # finally, compute the detection rate using Neijssel+19 Eq. 2
        detectionRate = np.zeros(shape=(nBinaries, self.nRedshiftsDetection))
        detectionRate = mergerRate[:, :self.nRedshiftsDetection] * detectionProbability * self.shellVolumes[:self.nRedshiftsDetection] / (1.0 + self.redshifts[:self.nRedshiftsDetection])

        return detectionRate, chirpMasses


    @classmethod
    def from_compas_h5(cls, inputPath:str, inputName:str):

        SE = SelectionEffects(p_SNRfilePath=SNR_NOISE_FILE_PATH,
                              p_SNRfileName=SNR_NOISE_FILE_NAME,
                              p_SNRsensitivity='O3')

        compas = COMPAS(p_COMPASfilePath=inputPath, p_COMPASfileName=inputName)
        compas.SetDCOmasks(p_DCOtypes='ALL', p_WithinHubbleTime=True, p_Pessimistic=True, p_NoRLOFafterCEE=True)
        compas.CalculatePopulationValues()
        return cls(
            p_COMPAS=compas,
            p_SE=SE,
            p_MaxRedshift=MAX_FORMATION_REDSHIFT,
            p_MaxRedshiftDetection=MAX_DETECTION_REDSHIFT,
            p_RedshiftStep=REDSHIFT_STEP
        )


class BinnedCosmicIntegrator(CosmicIntegration):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defaultChirpMassBins, self.defualtChirpMassBinWidths = MakeChirpMassBins(minChirpMass=MIN_CHIRPMASS, maxChirpMass=MAX_CHIRPMASS, binWidthPercent=McBIN_WIDTH_PERCENT)

    def FindBinnedDetectionRate(self,
                          # use default values for the other detection rate parameters
                          p_Alpha                = NEIJSSEL_ALPHA,
                          p_Sigma                = NEIJSSEL_SIGMA,
                          p_SFRa                 = NEIJSSEL_SFR_A,
                          p_SFRd                 = NEIJSSEL_SFR_D,
                          p_ChirpMassBins        = None,
                          ):
        if p_ChirpMassBins is None:
            p_ChirpMassBins = self.defaultChirpMassBins

        numChirpMassBins = len(p_ChirpMassBins) + 1
        detectionRate, chirpMasses = self.FindDetectionRate(p_BinaryFraction=0.7, p_Alpha=p_Alpha, p_Sigma=p_Sigma,
                                                            p_SFRa=p_SFRa, p_SFRd=p_SFRd)

        numRows = detectionRate.shape[1]
        numColumns = detectionRate.shape[0]

        # bin the detection rates
        binnedDetectionRate = np.zeros((numChirpMassBins, numRows), dtype=float)
        for Mc in range(numColumns):
            c = np.random.randint(0, numColumns)
            McBin = ChirpMassBin(chirpMasses[c], p_ChirpMassBins)
            for zBin in range(numRows):
                binnedDetectionRate[McBin][zBin] += detectionRate[c][zBin]

        return binnedDetectionRate




# create variable width chirpmass bins
# returns:
#   list of doubles: bin right edges
#   list of doubles: bin widths
def MakeChirpMassBins(minChirpMass = MIN_CHIRPMASS, maxChirpMass = MAX_CHIRPMASS, binWidthPercent = McBIN_WIDTH_PERCENT):

    # first bin is 0..minChirpMass
    binLeftEdge   = 0.0
    thisChirpMass = minChirpMass / 2.0
    binHalfWidth  = thisChirpMass
    binRightEdge  = [minChirpMass]
    binWidth      = [minChirpMass]

    # remaining bins are each binWidthPercent around a chirpmass, from minChirpMass
    while thisChirpMass < maxChirpMass:

        binLeftEdge   = binRightEdge[len(binRightEdge) - 1]
        thisChirpMass = 100.0 * binLeftEdge / (100.0 - (binWidthPercent / 2.0))
        binHalfWidth  = thisChirpMass - binLeftEdge
        binRightEdge.append(thisChirpMass + binHalfWidth)
        binWidth.append(thisChirpMass + binHalfWidth - binLeftEdge)
    
    return binRightEdge, binWidth


# find chirpMass bin in chirpMassBins
# allows for variable width bins
def ChirpMassBin(chirpMass, chirpMassBins):

    bin = 0
    while chirpMass >= chirpMassBins[bin]:
        bin += 1
        if bin >= len(chirpMassBins): break

    return bin


# sample from COMPAS data

def Sample(CSVwriter, p_CI:BinnedCosmicIntegrator, p_NumSamples, p_AlphaVector = ALPHA_VALUES, p_SigmaVector = SIGMA_VALUES, p_SFRaVector = SFR_A_VALUES, p_SFRdVector = SFR_D_VALUES):
    
    global verbose

    ntotal = len(p_AlphaVector) * len(p_SigmaVector) * len(p_SFRaVector) * len(p_SFRdVector) * p_NumSamples
    pbar = tqdm(total=ntotal, desc='Sampling', unit='sample', unit_scale=True)


    # create data for each sigma required
    for _, alpha in enumerate(p_AlphaVector):
        for _, sigma in enumerate(p_SigmaVector):
            for _, SFRa in enumerate(p_SFRaVector):
                for _, SFRd in enumerate(p_SFRdVector):

                    for sample in range(p_NumSamples):

                        pbar.desc = f'Sampling alp,sig,sfa,sfd=[{alpha, sigma, SFRa, SFRd}]'

                        print('\nSampling sample ', sample, ', alpha =', alpha, ', sigma =', sigma, ', SFRa =', SFRa, ', SFRd =', SFRd)

                        if verbose:
                            print('Get detection rate matrix')
                            t = time.process_time()

                        binnedDetectionRate = p_CI.FindBinnedDetectionRate(alpha, sigma, SFRa, SFRd)
                        numChirpMassBins, numZBins = binnedDetectionRate.shape

                        if verbose:
                            print('Have detection rate matrix after', time.process_time() - t, 'seconds')

                        # write binned detection rates to output file
                        row = [alpha, sigma, SFRa, SFRd, numChirpMassBins, numZBins]
                        for xBin in range(numChirpMassBins):
                            for yBin in range(NUM_REDSHIFT_BINS):
                                row.append(str(binnedDetectionRate[xBin][yBin]))

                        CSVwriter.writerow(row)

                        if verbose: print('\nDetection rates written to output file: #McBins =', numChirpMassBins, ', #zBins =', numZBins)

                        pbar.update(1)


# convert string to bool (mainly for arg parser)
def str2bool(v):
    if isinstance(v, bool): return v

    if   v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected!')


def main():

    print("STARTING DETECTION RATES SAMPLER")
    global verbose

    # setup argument parser
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position = 4, width = 90)
    parser    = argparse.ArgumentParser(description = 'Detection rates sampler.', formatter_class = formatter)

    # define arguments
    parser.add_argument('outputFilename',     metavar = 'output',     type = str,      nargs = 1,                                                                  help = 'output file name')
    parser.add_argument('-i', '--inputFilename', dest = 'inputName',  type = str,                 action = 'store',               default = COMPAS_HDF5_FILE_NAME, help = 'COMPAS HDF5 file name (def = ' + COMPAS_HDF5_FILE_NAME + ')')
    parser.add_argument('-p', '--inputFilepath', dest = 'inputPath',  type = str,                 action = 'store',               default = COMPAS_HDF5_FILE_PATH, help = 'COMPAS HDF5 file path (def = ' + COMPAS_HDF5_FILE_PATH + ')')
    parser.add_argument('-v', '--verbose',       dest = 'verbose',    type = str2bool, nargs='?', action = 'store', const = True, default = False,                 help = 'verbose flag (def = True)')
    parser.add_argument('-n', '--numSamples',    dest = 'numSamples', type = int,                 action = 'store',               default = SAMPLE_COUNT,          help = 'Number of samples (def = ' + str(SAMPLE_COUNT) + ')')
    parser.add_argument('-a', '--alpha',         dest = 'fAlpha',     type = float,               action = 'store',               default = None,                  help = 'alpha')
    parser.add_argument('-s', '--sigma',         dest = 'fSigma',     type = float,               action = 'store',               default = None,                  help = 'sigma')
    parser.add_argument('-A', '--sfrA',          dest = 'fsfrA',      type = float,               action = 'store',               default = None,                  help = 'sfrA')
    parser.add_argument('-D', '--sfrD',          dest = 'fsfrD',      type = float,               action = 'store',               default = None,                  help = 'sfrD')

    # parse arguments
    args = parser.parse_args()

    if len(args.outputFilename) < 1 or len(args.outputFilename) > 1:
        print('Expected single output filename!')
        sys.exit()

    verbose = args.verbose

    # set parameters ranges if not user supplied
    fAlpha = ALPHA_VALUES if args.fAlpha is None else [args.fAlpha]
    fSigma = SIGMA_VALUES if args.fSigma is None else [args.fSigma]
    fsfrA  = SFR_A_VALUES if args.fsfrA  is None else [args.fsfrA]
    fsfrD  = SFR_D_VALUES if args.fsfrD  is None else [args.fsfrD]

    # seed random number generator
    np.random.seed(0) # AVI SET TO 0 FOR REPRODUCIBILITY

    # initialise Cosmic Integrator
    if verbose:
        print('Start CI initialisation')
        t = time.process_time()

    CI = BinnedCosmicIntegrator.from_compas_h5(inputPath = args.inputPath, inputName = args.inputName)


    if verbose:
        print('CI initialisation done after', time.process_time() - t, 'seconds')


    # open csv file - overwrite any existing file
    with open(args.outputFilename[0] + '.csv', 'w', newline = '') as csvFile:
        writer = csv.writer(csvFile)

        # get and write samples
        Sample(writer, CI, args.numSamples, fAlpha, fSigma, fsfrA, fsfrD)


if __name__ == "__main__":
    main()   
