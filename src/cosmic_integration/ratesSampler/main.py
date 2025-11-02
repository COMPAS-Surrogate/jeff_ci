import argparse
import csv
import sys
import time
import numpy as np
from tqdm import tqdm

from .binned_cosmic_integrator import BinnedCosmicIntegrator
from .ratesSampler import (
    ALPHA_VALUES,
    SFR_A_VALUES,
    SFR_D_VALUES,
    SIGMA_VALUES,
    SAMPLE_COUNT,
    COMPAS_HDF5_FILE_NAME,
    COMPAS_HDF5_FILE_PATH,
    NUM_REDSHIFT_BINS,
)


# sample from COMPAS data

def Sample(CSVwriter, p_CI: BinnedCosmicIntegrator, p_NumSamples, p_AlphaVector=ALPHA_VALUES,
           p_SigmaVector=SIGMA_VALUES, p_SFRaVector=SFR_A_VALUES, p_SFRdVector=SFR_D_VALUES):
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

                        print('\nSampling sample ', sample, ', alpha =', alpha, ', sigma =', sigma, ', SFRa =', SFRa,
                              ', SFRd =', SFRd)

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

                        if verbose: print('\nDetection rates written to output file: #McBins =', numChirpMassBins,
                                          ', #zBins =', numZBins)

                        pbar.update(1)


# convert string to bool (mainly for arg parser)
def str2bool(v):
    if isinstance(v, bool): return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected!')


def main():
    print("STARTING DETECTION RATES SAMPLER")
    global verbose

    # setup argument parser
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=4, width=90)
    parser = argparse.ArgumentParser(description='Detection rates sampler.', formatter_class=formatter)

    # define arguments
    parser.add_argument('outputFilename', metavar='output', type=str, nargs=1, help='output file name')
    parser.add_argument('-i', '--inputFilename', dest='inputName', type=str, action='store',
                        default=COMPAS_HDF5_FILE_NAME,
                        help='COMPAS HDF5 file name (def = ' + COMPAS_HDF5_FILE_NAME + ')')
    parser.add_argument('-p', '--inputFilepath', dest='inputPath', type=str, action='store',
                        default=COMPAS_HDF5_FILE_PATH,
                        help='COMPAS HDF5 file path (def = ' + COMPAS_HDF5_FILE_PATH + ')')
    parser.add_argument('-v', '--verbose', dest='verbose', type=str2bool, nargs='?', action='store', const=True,
                        default=False, help='verbose flag (def = True)')
    parser.add_argument('-n', '--numSamples', dest='numSamples', type=int, action='store', default=SAMPLE_COUNT,
                        help='Number of samples (def = ' + str(SAMPLE_COUNT) + ')')
    parser.add_argument('-a', '--alpha', dest='fAlpha', type=float, action='store', default=None, help='alpha')
    parser.add_argument('-s', '--sigma', dest='fSigma', type=float, action='store', default=None, help='sigma')
    parser.add_argument('-A', '--sfrA', dest='fsfrA', type=float, action='store', default=None, help='sfrA')
    parser.add_argument('-D', '--sfrD', dest='fsfrD', type=float, action='store', default=None, help='sfrD')

    # parse arguments
    args = parser.parse_args()

    if len(args.outputFilename) < 1 or len(args.outputFilename) > 1:
        print('Expected single output filename!')
        sys.exit()

    verbose = args.verbose

    # set parameters ranges if not user supplied
    fAlpha = ALPHA_VALUES if args.fAlpha is None else [args.fAlpha]
    fSigma = SIGMA_VALUES if args.fSigma is None else [args.fSigma]
    fsfrA = SFR_A_VALUES if args.fsfrA is None else [args.fsfrA]
    fsfrD = SFR_D_VALUES if args.fsfrD is None else [args.fsfrD]

    # seed random number generator
    np.random.seed(0)  # AVI SET TO 0 FOR REPRODUCIBILITY

    # initialise Cosmic Integrator
    if verbose:
        print('Start CI initialisation')
        t = time.process_time()

    CI = BinnedCosmicIntegrator.from_compas_h5(inputPath=args.inputPath, inputName=args.inputName)

    if verbose:
        print('CI initialisation done after', time.process_time() - t, 'seconds')

    # open csv file - overwrite any existing file
    with open(args.outputFilename[0] + '.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)

        # get and write samples
        Sample(writer, CI, args.numSamples, fAlpha, fSigma, fsfrA, fsfrD)


if __name__ == "__main__":
    main()
