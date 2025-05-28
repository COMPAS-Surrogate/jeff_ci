# Jeff's Cosmic integration code

Author: Jeff Riley

**Install:**
```bash
pip install -e .
```

**Usage:**
```bash
usage: run_cosmic_integration [-h] [-i INPUTNAME] [-p INPUTPATH] [-v [VERBOSE]]
                              [-n NUMSAMPLES] [-a FALPHA] [-s FSIGMA] [-A FSFRA]
                              [-D FSFRD]
                              output

Detection rates sampler.

positional arguments:
  output
    output file name

optional arguments:
  -h, --help
    show this help message and exit
  -i INPUTNAME, --inputFilename INPUTNAME
    COMPAS HDF5 file name (def = COMPAS_Output.h5)
  -p INPUTPATH, --inputFilepath INPUTPATH
    COMPAS HDF5 file path (def = .)
  -v [VERBOSE], --verbose [VERBOSE]
    verbose flag (def = True)
  -n NUMSAMPLES, --numSamples NUMSAMPLES
    Number of samples (def = 10)
  -a FALPHA, --alpha FALPHA
    alpha
  -s FSIGMA, --sigma FSIGMA
    sigma
  -A FSFRA, --sfrA FSFRA
    sfrA
  -D FSFRD, --sfrD FSFRD
    sfrD

```
