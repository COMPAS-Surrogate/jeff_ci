from .ratesSampler import CosmicIntegration, NEIJSSEL_ALPHA, NEIJSSEL_SIGMA, NEIJSSEL_SFR_A, NEIJSSEL_SFR_D, MAX_CHIRPMASS, MIN_CHIRPMASS, McBIN_WIDTH_PERCENT
from .ratesSampler import MAX_CHIRPMASS, MIN_CHIRPMASS, MAX_DETECTION_REDSHIFT, REDSHIFT_STEP, NUM_REDSHIFT_BINS, McBIN_WIDTH_PERCENT
import numpy as np
import os



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
            c = np.random.randint(0, numColumns) #TODO: why is this random?
            McBin = ChirpMassBin(chirpMasses[c], p_ChirpMassBins)
            for zBin in range(numRows):
                binnedDetectionRate[McBin][zBin] += detectionRate[c][zBin]

        return binnedDetectionRate

    @classmethod
    def from_compas_fpath(cls, fpath):
        """
        Create a BinnedCosmicIntegrator from a COMPAS file path.
        """
        inputPath, inputName = os.path.split(fpath)
        return cls.from_compas_h5(inputPath=inputPath, inputName=inputName)


# create variable width chirpmass bins
# returns:
#   list of doubles: bin right edges
#   list of doubles: bin widths
def MakeChirpMassBins(minChirpMass=MIN_CHIRPMASS, maxChirpMass=MAX_CHIRPMASS, binWidthPercent=McBIN_WIDTH_PERCENT):
    # first bin is 0..minChirpMass
    binLeftEdge = 0.0
    thisChirpMass = minChirpMass / 2.0
    binHalfWidth = thisChirpMass
    binRightEdge = [minChirpMass]
    binWidth = [minChirpMass]

    # remaining bins are each binWidthPercent around a chirpmass, from minChirpMass
    while thisChirpMass < maxChirpMass:
        binLeftEdge = binRightEdge[len(binRightEdge) - 1]
        thisChirpMass = 100.0 * binLeftEdge / (100.0 - (binWidthPercent / 2.0))
        binHalfWidth = thisChirpMass - binLeftEdge
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



def get_default_mc_z_bins():
    z_bin_right_edges = np.arange(0.0, MAX_DETECTION_REDSHIFT + REDSHIFT_STEP, REDSHIFT_STEP)[:NUM_REDSHIFT_BINS]
    mc_bin_left_edges, mc_bin_widths = MakeChirpMassBins(minChirpMass=MIN_CHIRPMASS, maxChirpMass=MAX_CHIRPMASS, binWidthPercent=McBIN_WIDTH_PERCENT)
    return np.array(mc_bin_left_edges), np.array(mc_bin_widths), np.array(z_bin_right_edges)