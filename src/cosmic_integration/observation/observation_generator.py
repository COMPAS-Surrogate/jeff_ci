import matplotlib.pyplot as plt
import numpy as np


def generate_posterior_samples(
        rates: np.ndarray,
        chirp_masses: np.ndarray,
        output_dir: str,
        n_samples=int(1e6),
        n_posterior_samples=int(1e4)
):
    """
    Process gravitational wave population synthesis data to generate posterior samples.

    Parameters:
    -----------
    rates_file_path : str
        Path to the rates CSV file
    chirp_masses_file_path : str
        Path to the chirp masses text file
    output_dir : str
        Directory to save posterior files
    n_samples : int
        Number of samples for prior generation (default: 1e6)
    n_posterior_samples : int
        Maximum number of posterior samples per event (default: 1e4)
    """

    # rates = np.array(rates) / np.sum(rates)  # normalize to sum to 1
    print(f"Total rate sum: {np.sum(rates)}")
    print(f"Maximum rate: {np.max(rates)}")  # should be << 1
    if np.max(rates) >= 1:
        warnings.warn(f"Rates should be probabilities (max < 1), but max={np.max(rates)} Check the rates file.")

    n_mc, n_z = rates.shape
    mc = chirp_masses
    print(f"Chirp mass array length: {len(mc)}, Rate matrix Mc dimension: {n_mc}")

    # Sample events based on rates
    mc_found = []
    z_found = []

    for i in range(n_z):
        r = np.random.rand(n_mc)
        detected_indices = np.where(r < rates[:, i])[0]
        mc_found.extend(mc[detected_indices])
        z = (i) * 0.1 + 0.05  # redshift bin center
        z_found.extend([z] * len(detected_indices))

    mc_found = np.array(mc_found)
    z_found = np.array(z_found)

    print(f"Rate sum check: {np.sum(rates)}, Number of detected events: {len(mc_found)}")

    # Generate chirp mass prior
    m1 = 1 + 999 * np.random.rand(n_samples)
    m2 = 1 + 999 * np.random.rand(n_samples)
    q = m2 / m1
    mc_prior = (m1 ** 0.6 * m2 ** 0.6) / (m1 + m2) ** 0.2

    # Apply cuts for physical binary systems
    mc_cut = (q > 0.05) & (q < 1) & (mc_prior > 1) & (mc_prior < 200)

    # Create chirp mass prior histogram
    mc_edges = np.arange(0.1, 200.1, 0.1)
    mc_prior_counts, _ = np.histogram(mc_prior[mc_cut], bins=mc_edges)
    mc_prior_counts = mc_prior_counts / np.max(mc_prior_counts)

    # Generate redshift prior
    z_prior = 1.5 * np.random.rand(int(1e6)) ** (1 / 3)
    z_edges = np.arange(0.01, 1.51, 0.01)
    z_prior_counts, _ = np.histogram(z_prior, bins=z_edges)
    z_prior_counts = z_prior_counts / np.max(z_prior_counts)

    # Generate posteriors for each detected event
    for i in range(len(mc_found)):
        # SNR sampling (CDF ~ SNR^{-3})
        rho = 12 * np.random.rand() ** (-1 / 3)

        # Chirp mass measurement uncertainty (Powell+ 2019, eq. 4)
        r0_mc = np.random.randn()
        r_mc = np.random.randn(n_samples)
        mc_out = mc_found[i] * (1 + 0.03 * 12 / rho * (r0_mc + r_mc))
        mc_out = mc_out[(mc_out > 0.1) & (mc_out < 199.9)]

        # Redshift measurement uncertainty (analogous to chirp mass)
        r0_z = np.random.randn()
        r_z = np.random.randn(n_samples)
        z_out = z_found[i] * (1 + 0.3 * 12 / rho * (r0_z + r_z))
        z_out = z_out[(z_out > 0) & (z_out < 1.49)]

        # Apply priors via rejection sampling
        r_mc = np.random.rand(len(mc_out))
        mc_indices = np.clip(np.ceil(mc_out * 10).astype(int) - 1, 0, len(mc_prior_counts) - 1)
        mc_out_post = mc_out[r_mc < mc_prior_counts[mc_indices]]

        r_z = np.random.rand(len(z_out))
        z_indices = np.clip(np.ceil(z_out * 100).astype(int) - 1, 0, len(z_prior_counts) - 1)
        z_out_post = z_out[r_z < z_prior_counts[z_indices]]

        # Combine and limit output length
        out_length = min(len(mc_out_post), len(z_out_post), n_posterior_samples)

        if out_length > 0:
            posterior_samples = np.column_stack([
                mc_out_post[:out_length],
                z_out_post[:out_length]
            ])

            # Save posterior samples
            output_file = os.path.join(output_dir, f'posterior-{i + 1}.dat')
            np.savetxt(output_file, posterior_samples, delimiter='\t')

    # Save priors
    mc_prior_sample = mc_prior[mc_cut][:n_posterior_samples]
    np.savetxt(os.path.join(output_dir, 'Mcprior.dat'), mc_prior_sample)

    z_prior_sample = z_prior[:n_posterior_samples]
    np.savetxt(os.path.join(output_dir, 'zprior.dat'), z_prior_sample)


def plot_gw_posteriors(output_dir, figsize=(8, 6), save_plots=True, show_plots=False, dpi=300):
    """
    Create simplified 2D plots for GW posterior analysis.

    Parameters:
    -----------
    output_dir : str
        Directory containing saved posterior files and priors
    figsize : tuple
        Figure size (width, height) in inches
    save_plots : bool
        Whether to save plots to files (default: True)
    show_plots : bool
        Whether to display plots interactively (default: False)
    dpi : int
        Resolution for saved plots (default: 300)
    """

    # Load priors
    mc_prior = np.loadtxt(os.path.join(output_dir, 'Mcprior.dat'))
    z_prior = np.loadtxt(os.path.join(output_dir, 'zprior.dat'))

    # Load all posterior files
    all_mc_post = []
    all_z_post = []
    individual_posteriors = []

    posterior_files = sorted([f for f in os.listdir(output_dir)
                              if f.startswith('posterior-') and f.endswith('.dat')])

    print(f"Found {len(posterior_files)} posterior files")

    for filename in posterior_files:
        data = np.loadtxt(os.path.join(output_dir, filename))
        if data.size > 0:
            if data.ndim == 1:
                data = data.reshape(1, -1)
            mc_samples = data[:, 0]
            z_samples = data[:, 1]

            all_mc_post.extend(mc_samples)
            all_z_post.extend(z_samples)

            event_id = int(filename.split('-')[1].split('.')[0])
            individual_posteriors.append({
                'mc': mc_samples,
                'z': z_samples,
                'event_id': event_id
            })

    all_mc_post = np.array(all_mc_post)
    all_z_post = np.array(all_z_post)

    print(f"Loaded {len(individual_posteriors)} events with {len(all_mc_post)} total samples")

    # 1. Prior 2D plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create 2D histogram of prior samples
    H_prior, z_edges, mc_edges = np.histogram2d(z_prior, mc_prior, bins=50)
    extent = [z_edges[0], z_edges[-1], mc_edges[0], mc_edges[-1]]

    im = ax.imshow(H_prior.T, aspect='auto', origin='lower', extent=extent,
                   cmap='Greys', interpolation='gaussian')

    ax.set_xlabel('Redshift')
    ax.set_ylabel('Chirp Mass [M☉]')
    ax.set_title('Prior Distribution')

    plt.colorbar(im, ax=ax, label='Number of Samples')
    plt.tight_layout()

    if save_plots:
        plt.savefig(os.path.join(output_dir, 'prior_2d.png'),
                    dpi=dpi, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 2. Individual posterior plots (one file per event)
    for event in individual_posteriors:
        if len(event['mc']) > 0 and len(event['z']) > 0:
            fig, ax = plt.subplots(figsize=figsize)

            # Create 2D histogram for this event
            H, z_edges, mc_edges = np.histogram2d(event['z'], event['mc'], bins=30)
            extent = [z_edges[0], z_edges[-1], mc_edges[0], mc_edges[-1]]

            im = ax.imshow(H.T, aspect='auto', origin='lower', extent=extent,
                           cmap='Blues', interpolation='gaussian')

            ax.set_xlabel('Redshift')
            ax.set_ylabel('Chirp Mass [M☉]')
            ax.set_title(f'Event {event["event_id"]} Posterior')

            plt.colorbar(im, ax=ax, label='Number of Samples')
            plt.tight_layout()

            if save_plots:
                plt.savefig(os.path.join(output_dir, f'posterior_event_{event["event_id"]}.png'),
                            dpi=dpi, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()

    # 3. Population plot (all events combined)
    if len(all_mc_post) > 0:
        fig, ax = plt.subplots(figsize=figsize)

        # Create 2D histogram of all posterior samples
        H_pop, z_edges, mc_edges = np.histogram2d(all_z_post, all_mc_post, bins=50)
        extent = [z_edges[0], z_edges[-1], mc_edges[0], mc_edges[-1]]

        im = ax.imshow(H_pop.T, aspect='auto', origin='lower', extent=extent,
                       cmap='Reds', interpolation='gaussian')

        ax.set_xlabel('Redshift')
        ax.set_ylabel('Chirp Mass [M☉]')
        ax.set_title(f'Population Posterior ({len(individual_posteriors)} events)')

        plt.colorbar(im, ax=ax, label='Number of Samples')
        plt.tight_layout()

        if save_plots:
            plt.savefig(os.path.join(output_dir, 'population_posterior.png'),
                        dpi=dpi, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if save_plots:
        print(f"Plots saved to {output_dir}:")
        print(f"- prior_2d.png")
        print(f"- posterior_event_X.png (for {len(individual_posteriors)} events)")
        print(f"- population_posterior.png")

    return len(individual_posteriors)
