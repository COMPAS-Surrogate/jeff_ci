# COMPAS cosmic integration GP surrogate

This code uses (most of) Jeff's rateSampler (cosmic integration tool), and adds a bayesian-optimisation layer to it.
The goal of the bayesian-optimisation is to build an LnL surrogate (given some observed data), and thereby put some constraints on 
cosmic-integration parameters. 




 

## Jeff's Cosmic integration code

Author: Jeff Riley

**Install:**
```bash
pip install -e .
```

**Repo hygiene (local artifacts):**
- Clean large, gitignored artifacts (dry-run): `python3 scripts/clean_local_artifacts.py`
- Actually delete: `python3 scripts/clean_local_artifacts.py --yes`
- Include simulation outputs: `python3 scripts/clean_local_artifacts.py --include-simulation-outputs --yes`
- Include large HDF5 datasets: `python3 scripts/clean_local_artifacts.py --include-datasets --yes`
- Override test output dir (avoid `tests/out` bloat): set `COSMIC_INTEGRATION_TEST_OUTDIR=/path/to/dir`
- Point tests at an external COMPAS HDF5 (avoid copying `h5out_5M.h5` into the repo): set `COSMIC_INTEGRATION_COMPAS_H5=/path/to/file.h5`

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
