from ..lnl_computer import LnLComputer, Observation
import click



@click.command()
@click.argument("observation_file", type=click.Path(exists=True))
@click.argument("compas_h5", type=click.Path(exists=True))
@click.argument("params", type=str)
def main(observation_file, compas_h5, params):
    """
    Run the LnLComputer to compute and plot the log likelihood for given parameters.
    :param observation_file: Path to the observation file
    :param compas_h5: Path to the COMPAS HDF5 file
    :param params: String of parameters in the format "alp:0.5 sig:0.3 sfA:1.0 sfD:2.0"
    """
    if params is None:
        obs = Observation.from_ilya(observation_file)
        params = obs.params
    else:
        params = paramstr_to_param(params)
    lnl_computer = LnLComputer.load(
        observation_file=observation_file,
        compas_h5=compas_h5,
    )
    fname = lnl_computer.plot(
        params=params,
        outdir="."
    )
    print(f"Plot saved to {fname}")


def paramstr_to_param(paramstr):
    """
    Convert a parameter string to a dictionary of parameters.

    :param paramstr: String of parameters in the format "alp:0.5 sig:0.3 sfA:1.0 sfD:2.0"
    :return: List of parameters
    """
    params = {}
    for p in paramstr.split():
        key, value = p.split(":")
        params[key] = float(value)
    print(f"Parameters: {params}")
    return [params['alp'], params['sig'], params['sfA'], params['sfD']]