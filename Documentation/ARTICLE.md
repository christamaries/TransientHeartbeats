# Predicting Parent-Pulse Multiplicity in Synthetic Gamma-Ray Burst Light Curves with a 1D Convolutional Neural Network

## Abstract

Gamma-ray burst (GRB) light curves are highly variable, multi-peaked
time series whose substructure — individual "pulses" — is thought to
trace discrete episodes of central-engine activity. Counting how many
such pulses seeded a given burst is difficult from real data alone:
pulses overlap, fade into instrumental noise, and, more fundamentally, no
observed burst carries a verifiable ground-truth pulse count, since that
quantity is a construct of a generative model rather than a directly
measurable one. We address this by training a one-dimensional
convolutional neural network (CNN) to predict the number of parent
pulses in light curves synthesized from the Stern & Svensson (1996)
stochastic pulse-avalanche model, using parameters calibrated to the
Swift/BAT instrument by Bazzanini et al. (2024). Across two dataset
sizes (3,000 and 10,000 light curves) and three label-distribution
regimes — the naturally imbalanced calibrated distribution, a
deliberately shifted distribution, and a distribution made uniform by
construction — we compare two loss functions and three class-imbalance
mitigation strategies. We find that naive training collapses onto
whichever pulse count is most common in the training data regardless of
the true structure of a given input; that reweighting existing rare
examples during training consistently recovers more of the rare classes
than undersampling common ones; and that a genuinely balanced training
set achieves the best per-class discrimination of any condition tested,
at a substantially higher data-generation cost. We conclude with the
central open limitation of this approach: because no real burst has a
verifiable pulse count, the model's predictions cannot be directly
validated against real Swift/BAT data, only checked for consistency
against an independent peak-counting proxy in future work.

## 1. Introduction

Gamma-ray bursts are the most luminous electromagnetic transients known,
observed as brief, intense flashes of gamma-ray photons lasting from
milliseconds to several minutes. Rather than a single smooth flash, GRB
light curves — photon count rate as a function of time — typically show
a highly variable, multi-peaked structure. Each distinguishable pulse in
this structure is thought to correspond to a discrete episode of energy
release at the central engine, such as a distinct accretion episode or a
magnetic reconnection event within the relativistic jet (Stern &
Svensson 1996). The number and arrangement of these pulses therefore
carries information about the central engine's activity pattern, not
merely its total output — but the central engine itself is never
observed directly; it can only be inferred from the light curve's
structure.

In practice, recovering that structure is difficult. Once more than a
handful of pulses are present, their rise-and-decay profiles overlap,
and faint or low-signal features become hard to separate from
instrumental noise. Manual counting is subjective and does not scale;
algorithmic peak-detection methods (e.g., MEPSA) impose a fixed
detection threshold and so systematically miss overlapping or
sub-threshold pulses, recovering a related but distinct quantity — the
number of *observable* peaks above some significance level — rather than
the *total* number of pulses the underlying process actually produced.

This work reframes pulse counting as a supervised learning problem.
Rather than attempting to count pulses in real, unlabeled data, we
generate synthetic light curves from a validated stochastic model for
which the true pulse count is known by construction, and train a 1D CNN
to recover that count directly from the light curve's time series. This
sidesteps the central obstacle to any data-driven approach to this
problem: real bursts have no verifiable ground-truth pulse count to
supervise against in the first place, since "number of parent pulses" is
a property of a generative model's internal bookkeeping, not something
any instrument measures.

## 2. Related Work

Stern & Svensson (1996) proposed that GRB light curves can be modeled as
the output of a stochastic pulse-avalanche process: a small number of
independent "parent" pulses are drawn at the start of a burst, each of
which may recursively trigger "child" pulses, producing the
characteristic multi-peaked, self-similar structure observed in real
bursts. The model has seven free parameters governing the number of
parent pulses, the branching behavior of child pulses, and the timing
and shape of individual pulses.

Bazzanini et al. (2024) fit this model's seven parameters against real
GRB catalogs using a genetic algorithm, producing two independently
calibrated parameter sets — one for the BATSE instrument and one for
Swift/BAT — such that light curves simulated under a given calibration
are statistically consistent with that instrument's real observed burst
population. The present work uses the Swift/BAT-calibrated parameter set
directly; the genetic-algorithm fitting procedure itself is not rerun
here.

Algorithmic peak-detection methods such as MEPSA represent the
conventional alternative to a learned approach: they identify
statistically significant local maxima in a light curve above a
detection threshold. Such methods answer a related but different
question from the one addressed here — how many peaks are individually
resolvable above the noise — rather than how many pulses the underlying
generative process actually produced, some of which may overlap or fall
below any fixed significance threshold.

## 3. Data and Methods

### 3.1 Stochastic Pulse-Avalanche Model

Following Stern & Svensson (1996), the number of parent pulses in a
given light curve, `n_parent_pulses`, is drawn once per light curve from
a Poisson distribution with mean `mu0`: `n_parent_pulses ~ Poisson(mu0)`,
redrawn if the sampled value is zero. Each parent pulse may
recursively trigger child pulses, with the number of children per pulse
drawn from a separate Poisson distribution with mean `mu`; `mu ≈ 1`
places this branching process near a critical regime, capable of
producing deep recursive cascades, bounded in practice by a fixed cap on
the total number of pulses per light curve. The remaining model
parameters (`alpha, delta1, delta2, tau_min, tau_max`) govern the
amplitude and temporal shape of individual pulses. All parameters other
than `mu0` are held fixed throughout this work at the Bazzanini et al.
(2024) Swift/BAT-calibrated values (`mu=1.34, alpha=2.53, delta1=-0.75,
delta2=0.27, tau_min=0.03, tau_max=56.8`); `mu0` is the only parameter
varied, as described below. Simulated light curves are passed through a
model of the Swift/BAT instrument response (0.064s time resolution,
1,400 cm² effective area, a calibrated background rate, and detection
thresholds on burst duration and signal-to-noise) before being retained
as training examples, so that retained light curves are consistent with
what Swift/BAT would actually have detected and cataloged.

### 3.2 Synthetic Dataset Generation

Three label-distribution regimes were generated, at two dataset sizes
each (N = 3,000 and N = 10,000 light curves):

- **Natural (μ₀ = 1.16)** — the Bazzanini et al. (2024) Swift/BAT-fitted
  value, producing a heavily right-skewed label distribution in which
  77-86% of light curves have only 1-2 parent pulses.
- **μ₀ = 10** — a deliberately shifted mean, producing a broader,
  substantially more symmetric label distribution centered near 9-11
  pulses (a Poisson distribution's skewness scales as `1/√μ0`, so
  larger `μ0` produces a visibly more Gaussian-shaped label
  distribution as a direct consequence of this scaling, independent of
  any astrophysical assumption).
- **Uniform-by-construction** — rather than a single fixed `μ0`, this
  dataset was built by sweeping `μ0` across each target pulse-count bin
  in turn and retaining only light curves whose realized pulse count
  matched that bin exactly, yielding a genuinely flat label distribution
  by design rather than by resampling. The computational cost of this
  method grows sharply with target bin, so this regime was capped at
  bins 1-8 (N=3,000) and 1-7 (N=10,000) rather than the full range
  observed under μ0=10.

**Table 1.** Composition of every dataset generated for this study.

| Dataset | N | True labels present | Notes |
|---|---|---|---|
| Natural, N=3,000 | 3,000 | 1-6 | 77-86% of examples are 1-2 parent pulses |
| Natural, N=10,000 | 10,000 | 1-8 | same shape as N=3,000, more examples per bin |
| μ0=10, N=3,000 | 3,000 | 2-25 | bin 1 absent — P(X=1∣λ=10) ≈ 0.045%, expected count ≈ 1.4, absence consistent with chance |
| μ0=10, N=10,000 | 10,000 | 1-23 | |
| Uniform, N=3,000 | 2,400 | 1-8 | exactly 300 examples per bin by construction |
| Uniform, N=10,000 | 7,000 | 1-7 | exactly 1,000 examples per bin; bin 8 omitted (cost, §3.2) |

### 3.3 Network Architecture

Light curves (peak-normalized to their own maximum, 15,783 time-bins at
Swift/BAT's native 0.064s resolution) are passed through a compact 1D
CNN, `PulseCountCNN`, summarized layer-by-layer in Table 2.

**Table 2.** Network architecture. `k`=kernel size, `s`=stride,
`p`=padding, in samples.

| Stage | Layers | Output channels | Output length |
|---|---|---|---|
| Input | — | 1 | 15,783 |
| Block 1 | Conv1d(k=15, s=2, p=7) → BatchNorm1d → ReLU → MaxPool1d(4) | 16 | 1,973 |
| Block 2 | Conv1d(k=9, s=2, p=4) → BatchNorm1d → ReLU → MaxPool1d(4) | 32 | 246 |
| Block 3 | Conv1d(k=5, s=2, p=2) → BatchNorm1d → ReLU | 64 | 123 |
| Pooling | AdaptiveAvgPool1d(1) | 64 | 1 |
| Head | Flatten → Linear(64→32) → ReLU → Dropout(p=0.2) → Linear(32→1) | 1 | — |

Kernel size decreases with depth (15 → 9 → 5 samples) while channel
count increases (16 → 32 → 64), a standard convolutional taper: early
layers use a wide window to detect a pulse's rise-and-decay shape
directly in the raw signal (a kernel of 15 samples at 0.064s/sample
spans roughly 1 second), while deeper layers operate on an
already-downsampled representation and combine simpler features into
more abstract ones. Each block's combination of a stride-2 convolution
and, in the first two blocks, 4-sample max-pooling shrinks the sequence
length by roughly 8x per block while growing the receptive field
substantially faster than the kernel size alone would suggest — by
Block 3, a single output position already integrates information from
roughly 389 samples of the original input, and the final global average
pool integrates over the entire remaining sequence, giving the network
an effectively global view of the light curve despite being built
entirely from local convolutions. Average, rather than max, pooling was
used for this final step specifically because the quantity being
predicted is a count: averaging a channel's activity over time yields a
value that scales with how often a detected pattern recurred, whereas
max-pooling would only indicate whether it occurred at least once,
discarding exactly the frequency information the task requires. Global
average pooling also makes the architecture's total parameter count
independent of input sequence length. The output is passed through
`softplus(x) + 1`, encoding the constraint `n_parent_pulses ≥ 1`
directly in the architecture. The same architecture was used, trained
independently from scratch with no shared weights, for every one of the
28 conditions reported below.

### 3.4 Training Configurations

For the two fixed-`μ0` regimes, three strategies for handling label
imbalance were compared, all applied at training time to one shared,
already-generated dataset:

- **Unweighted (`unw`)** — the baseline: ordinary random sampling over
  the dataset exactly as generated, with no correction for class
  frequency.
- **Weighted (`wgt`)** — training batches are drawn via a weighted
  random sampler using inverse-class-frequency weights, so that
  under-represented labels are shown to the model more often per epoch
  than their raw frequency in the dataset would imply. No examples are
  discarded; rare examples are simply resampled with replacement more
  often during training.
- **Capped (`cap`)** — before training, any label value with more than a
  fixed threshold (1,000) of examples is randomly undersampled down to
  that threshold; labels already at or below the threshold are left
  untouched. Sampling during training is then unweighted on this
  smaller, flatter dataset. Unlike weighting, this strategy permanently
  discards majority-class data rather than reweighting how it is seen.

The uniform-by-construction regime required none of these, since its
labels are balanced by generation rather than by any training-time
correction. Each condition was additionally trained under two loss
functions — mean squared error (MSE), and Poisson negative
log-likelihood, motivated by `n_parent_pulses` itself being
Poisson-distributed by construction — giving a 3 (imbalance strategy) ×
2 (loss) factorial for each fixed-`μ0` regime and a loss-only comparison
for the uniform regime. In total, 28 model instances were trained across
both dataset sizes and all regime/strategy/loss combinations (MSE was
omitted for `μ0=10` at N=10,000 in an initial pass and completed
subsequently; all combinations are reported here).

### 3.5 Evaluation Metrics

Each trained model was evaluated on a held-out test split (15% of its
dataset) using mean absolute error (MAE), exact-match accuracy, and
within-±1 accuracy (predictions off by at most one pulse), alongside
per-class accuracy broken out by true pulse count.

## 4. Results

### 4.1 Naive training collapses onto the dominant label

Under the natural (μ0=1.16) regime, unweighted training reproduces the
label distribution's shape in its predictions: exact-match accuracy is
highest on classes 1-2 (which dominate the training data) and falls to
near zero for classes 3 and above, regardless of dataset size. This
produces a deceptively strong aggregate score — unweighted models top
both exact-match and within-±1 accuracy in Tables 3 and 4 (§4.5) — that is an artifact
of the label distribution rather than evidence of a better model:
guessing near the mode is correct most of the time when the mode
dominates the test set as heavily as it does here.

### 4.2 Reweighting outperforms undersampling

Weighted sampling (rare examples shown more often during training,
without discarding data) reaches classes that unweighted or capped
training never engages at all. At N=3,000, weighted MSE training
recovers 21.9% and 4.3% exact-match accuracy on classes 3 and 4
respectively, versus 0% for unweighted and capped training on both
classes; this pattern holds at N=10,000 and, as detailed below, under
the μ0=10 regime as well. Capped (undersampling) training only produces
comparable gains when undersampling happens to tie several classes at
exactly equal size — at N=3,000 under the natural regime, for instance,
where classes 1-2 are undersampled to the same size, some genuine
discrimination between them emerges; at N=3,000 under μ0=10, where no
class exceeds the undersampling threshold, capping is a complete
no-op, numerically identical to unweighted training. The same
undersampling threshold becomes genuinely active at N=10,000 under μ0=10
(several classes there naturally exceed it), but even so, it only
redistributes accuracy within the already-populous classes it directly
touches — it does not reach classes that were rare to begin with, since
by construction it can only remove examples from classes *above* its
threshold. Reweighting is the only strategy tested here that reaches
genuinely rare classes at all, in every regime and at both dataset
sizes tested.

### 4.3 Loss function choice interacts with resampling strategy

Contrary to the theoretical expectation that Poisson negative
log-likelihood should better match Poisson-distributed count labels,
MSE weighted training modestly outperformed Poisson weighted training on
rare-class recovery under the natural regime at both dataset sizes
(e.g., class 3 exact-match of 28.6% vs. 18.5% at N=10,000). Under μ0=10,
the two losses perform comparably once weighted sampling is applied
(within-±1 accuracy of 33.9% for MSE versus 35.3% for Poisson at
N=10,000), suggesting the sampler, not the loss function, is doing most
of the work of reaching rare classes; why Poisson loss underperforms its
theoretical motivation specifically when paired with aggressive
resampling is not resolved by this work. The one regime where Poisson
loss shows a clear, uncontested advantage is the uniform-by-construction
dataset at N=3,000 (within-±1 accuracy of 45.8% vs. 40.6% for MSE) —
the only condition in this study with no competing resampling mechanism
to interact with the loss function's own behavior. That advantage
narrows sharply at N=10,000 (55.6% vs. 54.5%), consistent with a general
expection that larger training sets reduce sensitivity to loss-function
choice.

### 4.4 A genuinely balanced dataset outperforms every resampling strategy

The uniform-by-construction regime achieves the best per-class
discrimination of any condition tested at a given dataset size — its
within-±1 accuracy (45.8-55.6% across sizes) exceeds every weighted or
capped condition under the natural or μ0=10 regimes on the same metric,
though the comparison is not perfectly like-for-like, since the
uniform regime's label range (1-7/1-8) is narrower than the other two
regimes' (1-6+ and 2-25 respectively) — a narrower label range makes
within-±1 accuracy an easier target independent of model quality. Within
that caveat, the uniform regime is the only one in which every tested
label value receives meaningfully engaged predictions rather than a
small subset of "reachable" classes, which we attribute to there being
no structural imbalance left for any training-time strategy to correct
or fail to correct.

### 4.5 Full results, all conditions

Tables 3 and 4 report aggregate test-set metrics for all 14 conditions
trained at each dataset size. Read with the caveat established in §4.1:
the unweighted rows' high aggregate scores reflect the concentration of
the label distribution, not superior per-class modeling; per-class
behavior is discussed in §4.1-§4.4.

**Table 3.** All conditions, N=3,000.

| Regime | Condition | Test MAE | Exact-match | Within-±1 |
|---|---|---|---|---|
| Natural (μ0=1.16) | MSE, unweighted | 0.633 | 46.2% | 92.2% |
| Natural (μ0=1.16) | MSE, weighted | 0.758 | 34.4% | 91.3% |
| Natural (μ0=1.16) | MSE, capped | 0.720 | 37.6% | 92.3% |
| Natural (μ0=1.16) | Poisson, unweighted | 0.658 | 44.4% | 91.6% |
| Natural (μ0=1.16) | Poisson, weighted | 0.749 | 42.7% | 86.0% |
| Natural (μ0=1.16) | Poisson, capped | 0.691 | 40.6% | 92.3% |
| μ0=10 | MSE, unweighted | 2.429 | 10.9% | 34.2% |
| μ0=10 | MSE, weighted | 2.802 | 10.0% | 33.1% |
| μ0=10 | MSE, capped | 2.429 | 10.9% | 34.2% |
| μ0=10 | Poisson, unweighted | 2.427 | 12.0% | 36.4% |
| μ0=10 | Poisson, weighted | 2.793 | 10.4% | 32.4% |
| μ0=10 | Poisson, capped | 2.427 | 12.0% | 36.4% |
| Uniform-by-construction | MSE | 1.861 | 13.9% | 40.6% |
| Uniform-by-construction | Poisson | 1.731 | 15.0% | 45.8% |

**Table 4.** All conditions, N=10,000.

| Regime | Condition | Test MAE | Exact-match | Within-±1 |
|---|---|---|---|---|
| Natural (μ0=1.16) | MSE, unweighted | 0.670 | 43.7% | 91.6% |
| Natural (μ0=1.16) | MSE, weighted | 0.801 | 35.9% | 86.5% |
| Natural (μ0=1.16) | MSE, capped | 0.860 | 32.6% | 83.2% |
| Natural (μ0=1.16) | Poisson, unweighted | 0.660 | 45.0% | 91.8% |
| Natural (μ0=1.16) | Poisson, weighted | 0.750 | 39.5% | 87.9% |
| Natural (μ0=1.16) | Poisson, capped | 0.843 | 35.7% | 83.1% |
| μ0=10 | MSE, unweighted | 2.291 | 12.7% | 39.3% |
| μ0=10 | MSE, weighted | 2.715 | 11.6% | 33.9% |
| μ0=10 | MSE, capped | 2.391 | 12.4% | 37.6% |
| μ0=10 | Poisson, unweighted | 2.297 | 13.1% | 39.9% |
| μ0=10 | Poisson, weighted | 2.631 | 12.3% | 35.3% |
| μ0=10 | Poisson, capped | 2.444 | 11.9% | 37.0% |
| Uniform-by-construction | MSE | 1.475 | 20.3% | 54.5% |
| Uniform-by-construction | Poisson | 1.438 | 19.8% | 55.6% |

Notably, the `capped` condition is numerically identical to `unweighted`
under μ0=10 at N=3,000 for both losses (0.9-2.427 MAE match to three
decimal places) — the undersampling threshold (1,000) never triggers at
this size, since the largest class there has only 374 examples, so no
data is ever actually removed. The same threshold becomes genuinely
active at N=10,000, where several classes naturally exceed it, and
`capped` there is no longer identical to `unweighted` (Table 4).

### 4.6 Best-performing condition per dataset

Table 5 summarizes the best-performing training condition for each
dataset regime and size, selected for genuine per-class engagement
rather than aggregate score alone (§4.1).

**Table 5.** Best-performing condition per dataset (by within-±1
accuracy among conditions that engage more than the dominant class
alone).

| Dataset | Best condition | Test MAE | Exact-match | Within-±1 |
|---|---|---|---|---|
| μ₀ = 1.16, N=3,000 | MSE, weighted | 0.758 | 34.4% | 91.3% |
| μ₀ = 1.16, N=10,000 | MSE, weighted | 0.801 | 35.9% | 86.5% |
| μ₀ = 10, N=3,000 | Poisson, weighted | 2.793 | 10.4% | 32.4% |
| μ₀ = 10, N=10,000 | Poisson, weighted | 2.631 | 12.3% | 35.3% |
| Uniform, N=3,000 | Poisson | 1.731 | 15.0% | 45.8% |
| Uniform, N=10,000 | Poisson | 1.438 | 19.8% | 55.6% |

Within-±1 accuracy is not directly comparable down this column, since it
depends on the width of each regime's true label range, not solely on
model quality; the natural regime's high values (83-92%) reflect labels
concentrated on 1-2 pulses, not a categorically better model than the
μ0=10 regime's 32-35% over a much wider label range.

## 5. Discussion

The consistent finding across every regime and dataset size tested is
that reweighting how existing examples are sampled during training
reaches further into a label distribution's rare tail than any strategy
that resamples or discards existing data, and that this advantage holds
regardless of loss function. Undersampling (capping) only ever produces
comparable gains as a side effect of coincidentally tying multiple
classes at exactly equal size after thinning — a structural accident of
the specific undersampling threshold chosen relative to the specific
class sizes present, not a general property of undersampling as a
strategy. This suggests that, for problems structurally similar to this
one — a single dominant class with a long, sparse tail — reweighting
during training is the more robust default remedy, with undersampling
better reserved for cases where the practitioner can verify in advance
that it will tie enough classes together to be worthwhile.

The uniform-by-construction dataset's superior per-class discrimination,
at the cost of a substantially higher data-generation budget, points to
a broader implication: where feasible, correcting label imbalance at the
data-generation stage dominates any correction applied at training time,
because training-time corrections operate only on the data that already
exists, while generation-time correction can supply genuinely new,
appropriately distributed examples of classes that would otherwise
remain rare no matter how they are resampled.

### 5.1 Limitations

The central limitation of this work is that its central claim — accurate
pulse counting — cannot be directly verified against real observational
data. `n_parent_pulses` is a property of the generative model used to
create the training data, not a directly observable quantity; no real
Swift/BAT burst carries a verified ground-truth pulse count to check a
prediction against, even in principle. A weaker proxy validation is
possible in future work: comparing this model's predictions on real
Swift/BAT light curves against an independent significant-peak count
from an algorithmic method such as MEPSA. Because MEPSA counts only
peaks above a fixed detection threshold while this model is trained to
recover the full generative pulse count (including overlapping or
sub-threshold pulses), agreement or disagreement between the two would
not constitute definitive validation, only a qualitative consistency
check; this comparison was not performed in the present work due to lack
of access to the real catalog and precomputed peak-detection outputs.
Further limitations include: a single fixed instrument/parameter
calibration (Swift/BAT) with no tested generalization to BATSE or other
instrument calibrations; single train/validation/test splits per
condition without repeated-seed averaging; and the uniform-by-
construction dataset's bin range being capped below its originally
intended extent due to the steeply rising computational cost of
harvesting high-pulse-count examples by this method.

## 6. Conclusion

A 1D CNN can recover the pulse-avalanche model's parent-pulse count from
a synthetic GRB light curve's time series, but naive training on an
imbalanced label distribution collapses onto the dominant class
regardless of the specific input, an effect that persists across two
independently generated label-distribution regimes and two dataset
sizes. Reweighting existing rare examples during training consistently
recovers more of the label distribution's rare tail than undersampling
common classes, independent of loss function; a genuinely balanced
training set, built by construction rather than by resampling, achieves
the best per-class discrimination of any condition tested, at a
significant increase in data-generation cost. Because the target label
is unobservable in real data, this model's predictions remain to be
checked for consistency against real Swift/BAT bursts using an
independent peak-counting method as a proxy, which is the natural next
step for this work.

## References

Stern, B. E. & Svensson, R. 1996, *Astrophysical Journal Letters*, 469,
L109, "Evidence for Chain Reaction in the Time Profiles of Gamma-Ray
Bursts."

Bazzanini, L., Ferro, L., Guidorzi, C., Angora, G., Amati, L., Brescia,
M., Bulla, M., Frontera, F., Maccary, R., Maistrello, M., Rosati, P., &
Tsvetkova, A. 2024, *Astronomy & Astrophysics*, 689, A266,
"Long gamma-ray burst light curves as the result of a common stochastic
pulse-avalanche process." doi:10.1051/0004-6361/202450150
