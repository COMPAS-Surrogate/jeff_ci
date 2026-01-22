# Simulation study

We recreate the study from the Jeff Riley's paper (https://arxiv.org/pdf/2303.00508), but using OUR active-learning/GP method. 




> 3.2.1. Method validation: inference on perfect measurements
In Section 2.5 we outlined how we validate our
method. Here we present the results of that validation.
Using COMPAS post-processing tools and the synthesised population of 512 million binaries, we created
a detection rate matrix, binned as described in Section 2.3, for a known value of λ: λ(α, σ, aSF , dSF ) =
(−0.325, 0.213, 0.012, 4.253). Using that detection rate
> matrix and assuming an observing time of either 0.1
year or 1 year, we created mock data sets D with perfect parameter measurement accuracy for each of the “detected” sources (58 in the 0.1 year data set, 578 in
the 1 year data set).
We carried out a Markov Chain Monte Carlo (MCMC)
search (e.g. Andrieu et al. (2003)), using the emcee
Python package (Foreman-Mackey et al. 2013), over the
λ = {α, σ, aSF , dSF } parameter space, assuming flat priors on λ. This search calculates the likelihood L(D|λi)
at each λi visited by the MCMC algorithm using the surrogate model (see Section 2.5.1 for the likelihood calculation). To confirm that our inference step was performing
adequately, we also performed a fine-grained grid search
consisting of 51 equidistant values for each of α, σ, and
aSF , and 101 equidistant values for dSF - for a total
of 51 × 51 × 51 × 101 ≈ 13.4 million points evaluated,
followed by a na¨ıve hill-climbing search (e.g. Russell &
Norvig (1995)), using the results of the grid search as
a starting point. We confirmed that the searches found
the same maximum likelihoods (within expected sampling variations).
The MCMC search posteriors are shown in Figure 5.
The true value is found within 68% credible intervals
for most MSSFR parameters under study, and within
95% credible intervals for all parameters. As expected,
the larger data set produces more precise inference on
MSSFR parameters, with the ratio of posterior width for
α approaching a factor of √
10 narrower on the data set
with 10 times more data (right panel), though the improvement is much smaller for poorly measured parameters such as dSF whose posteriors are prior-dominated.
We considered surrogate models trained on data sets
with 10 bootstrapped samples per λ value and 100
bootstrapped samples and found that they are not significantly different, which is consistent with the accuracy of the surrogate model shown in Figure 3. We
show results for surrogate models trained with 100 bootstrapped samples per λ from here on.

> 3.2.2. Method validation: inference on uncertain measurements
>
> We used the same detection rate matrix as in the previous subsection, corresponding to λ(α, σ, aSF , dSF ) = (−0.325, 0.213, 0.012, 4.253), to validate inference on  mock observations with measurement uncertainties. Using that detection rate matrix, assuming an observing  time of one year, we created a dataset of mock LVK data containing 578 events. We then randomly sampled 58 events from that dataset, creating a new dataset with  an observing time of 0.1 year.
We then replaced each of the chosen events with samples from an associated mock posterior. To create these 
samples, we used a mock model of the LVK prior. We
built the source-frame chirp mass prior by assuming
that the component masses m1 > m2 are uniformly
drawn from the range [1, 1000]Mf, with additional cuts
m2 ∈ [0.05m1, m1] and M ≡ m0.6
1 m0.6
2
(m1 + m2)
−0.2 ∈
[1, 200]Mf. For the mock redshift prior, we used π(z) ∝
z
2 on z ∈ [0.01, 1.5]. We then weighed mock chirp mass
and redshift samples taken from
M ∼ MT
(1 + 0.03
12
ρ
(r0 + r)); (11)
z ∼ z
T
(1 + 0.3
12
ρ
(r0 + r))
by these priors. In Eq. 11, which follows Powell et al.
(2019), the superscript T denotes the true value, ρ is
the signal-to-noise ratio sampled from p(ρ) ∝ ρ
−4 with
a minimum of ρ ≥ 12, r0 is a normal random variable
which stochastically shifts the peak of the posterior away
from the truth, and r is a vector of normal random variables which provides the spread of the posterior. Thus,
the small and large data sets are composed of 58 and
578 events, each represented by a set of mock posterior
samples that contain mock measurement uncertainties
on the observed parameters M and z.
Using each of those datasets as “true” data, we used
our surrogate model to infer posteriors on λ assuming
flat priors. The MCMC search statistics are shown in
Figure 6.
Figure 6 shows that, as expected, inference provides
consistent credible intervals, with most “true” values
falling within the 1-σ credible interval, and the remainder within the 2-σ credible interval. As before, the larger
data set produces more precise inference on MSSFR parameters. Comparing Figure 6 with Figure 5, we see
that mock measurement uncertainties have limited impact on MSSFR inference for the smaller data set, where
inference is predominantly limited by the total number
of events. For the larger data set, incorporating measurement uncertainty does lead to a moderate deterioration in the accuracy of inference.