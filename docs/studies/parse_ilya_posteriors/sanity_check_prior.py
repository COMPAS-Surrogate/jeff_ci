import bilby
from bilby.core.prior import Constraint, PowerLaw
from bilby.gw.prior import UniformInComponentsChirpMass
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


prior = bilby.prior.PriorDict(
    dict(
        chirp_mass = UniformInComponentsChirpMass(name='chirp_mass', minimum=1, maximum=200),
        mass_1 = Constraint(name='mass_1', minimum=2, maximum=2000),
        mass_2 = Constraint(name='mass_2', minimum=0.1, maximum=2000),
        redshift = PowerLaw(name='redshift', alpha=2, minimum=0.01, maximum=1.5)
    )
)



def test_prior():
    """
    Test the prior by sampling from it and printing the samples.
    """

    import matplotlib.pyplot as plt


    chirp_mass_samples = np.loadtxt("posteriors/Mcprior.dat")
    redshift_samples = np.loadtxt("posteriors/zprior.dat")



    mcs = np.linspace(1, 200, 1000)
    z = np.linspace(0.01, 1.5, 1000)
    mc_pdf = prior['chirp_mass'].prob(mcs)
    z_pdf = prior['redshift'].prob(z)



    plt.figure(figsize=(10, 5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))




    axes[0].hist(redshift_samples, bins=50, density=True, alpha=0.7, label='Redshift samples', color='orange')
    axes[0].plot(z, z_pdf, label='Redshift Prior PDF', color='blue')

    axes[0].set_xlabel('Redshift (z)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Redshift Prior Histogram')

    axes[1].hist(chirp_mass_samples, bins=50, density=True, alpha=0.7, label='Chirp Mass Samples', color='orange')
    axes[1].plot(mcs, mc_pdf, label='Chirp Mass Prior PDF', color='blue')

    axes[1].set_xlabel('Chirp Mass (Mc)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Chirp Mass Prior Histogram')
    plt.tight_layout()
    plt.savefig("prior_hist.png")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plot_empirical_vs_model_cdf(chirp_mass_samples, mcs, mc_pdf, 'chirp_mass')

    plt.subplot(1, 2, 2)
    plot_empirical_vs_model_cdf(redshift_samples, z, z_pdf, 'redshift')

    plt.tight_layout()
    plt.savefig("prior_cdf_comparison.png")


    # quantitative checks to say that the prior samples come from the prior PDF
    # from scipy.stats import kstest
    #
    # # Compute CDF from the prior
    # from scipy.interpolate import interp1d
    # mc_cdf = np.cumsum(mc_pdf)
    # mc_cdf /= mc_cdf[-1]
    # mc_cdf_func = interp1d(mcs, mc_cdf, bounds_error=False, fill_value=(0, 1))
    #
    # z_cdf = np.cumsum(z_pdf)
    # z_cdf /= z_cdf[-1]
    # z_cdf_func = interp1d(z, z_cdf, bounds_error=False, fill_value=(0, 1))
    #
    # # Perform KS test
    # ks_mc = kstest(chirp_mass_samples, mc_cdf_func)
    # ks_z = kstest(redshift_samples, z_cdf_func)
    #
    # print(f"KS test for chirp_mass: statistic = {ks_mc.statistic:.4f}, p-value = {ks_mc.pvalue:.4f}")
    # print(f"KS test for redshift:   statistic = {ks_z.statistic:.4f}, p-value = {ks_z.pvalue:.4f}")
    #


def plot_empirical_vs_model_cdf(samples, model_x, model_pdf, label):
    empirical_cdf_x = np.sort(samples)
    empirical_cdf_y = np.linspace(0, 1, len(samples), endpoint=False)

    model_cdf = np.cumsum(model_pdf)
    model_cdf /= model_cdf[-1]
    model_cdf_func = interp1d(model_x, model_cdf, bounds_error=False, fill_value=(0, 1))

    plt.plot(empirical_cdf_x, empirical_cdf_y, label='Empirical')
    plt.plot(model_x, model_cdf_func(model_x), label='Model CDF')
    plt.xlabel(label)
    plt.ylabel('CDF')
    plt.legend()
    plt.title(f"CDF Comparison for {label}")



