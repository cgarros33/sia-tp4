# Kohonen sweep analysis (working notes)

Date: 2026-10-23

Scope: exercise 1.2, archived 27-variable run and current 10-component run.

This note is kept in English as originally written; the rest of `docs/` is in
Spanish.

## Sources

- Archived sweeps: `resultados/ej1/ej1.2-full/kohonen_sweep/sweep1/` and
  `sweep2/`.
- Archived fixed-k sweeps: `resultados/ej1/ej1.2-full/kohonen_sweep/sweep3/`,
  `sweep4/` and `sweep5/`.
- Current PC10 sweeps: `resultados/ej1/ej1.2-pca10/kohonen_sweep/sweep3/`,
  `sweep4/` and `sweep5/`.
- Current PC10 final runs: `resultados/ej1/ej1.2-pca10/kohonen_k3/`,
  `kohonen_k4/` and `kohonen_k5/`.
- Inputs: `datos/creadores.csv`,
  `resultados/ej1/pca_0.90/scores.csv`.
- Engine: `src/ej1/kohonen.py`. All in-memory experiments used seeds
  `[0, 1, 2]` and report mean ± sample standard deviation.

---

## Bottom line

- **A TE tie-breaker is not justified.** Exact QE ties exist, but TE is
  identical in all of them. In near-ties, the minimum-TE candidate always has
  *worse* QE.
- **The old huge gains are not caused by the 10 PCs.** The archived
  27-variable fixed-k sweeps are equally flat. The cause is the
  **map-capacity/neighborhood regime**: `sweep1`/`sweep2` ran with `k² > n`,
  where radius decay, eta and iterations let the map approach a degenerate
  one-neuron-per-sample solution. The same PC10 data at `k=10` reproduces the
  old QE collapse almost exactly.

---

## 1. TE ties: there are ties, but TE adds nothing

### Exact QE ties (new PC10 sweeps)

Only three steps have exact QE ties, all in `k=3`:

| Step | Candidates tied | QE (3 seeds) | TE (3 seeds) |
|---|---|---:|---:|
| 3 `inicializacion` | `uniforme`, `muestras` | 3.136935 | 0.0600 |
| 4 `radio_inicial_relativo` | `0.4`, `0.5` | 3.136935 | 0.0600 |
| 5 `decaimiento_radio` | `constante`, `lineal`, `exponencial`, `inversa` | 3.136935 | 0.0600 |

The per-seed QE values are identical to 8 decimals, not just equal on average.
**TE is also tied at 0.0600**, so a TE tie-breaker would change nothing.

Why so flat at `k=3`? The chosen radius is `R0 = 0.4 × 3 = 1.2`, and
`R_f = 1.0`; all decay schemes run almost the same radius range. The map is
tiny, so the final `R=1` neighborhood covers **40.7% of the grid on average**
(vs 4.6% at `k=10`). Neurons stay globally coupled and converge to the same
partition.

### Statistical near-ties

Using “within 1 standard error of the best” with 3 seeds, 17 of 24 steps have
near-ties. In **10 of those 17**, the minimum-TE candidate differs from the
minimum-QE candidate — and it always has higher QE:

| Step | QE winner | TE winner | QE penalty of TE choice |
|---|---|---|---:|
| `sweep4` step 2 `distancia` | `exponencial` 2.808934, TE 0.1067 | `euclidiana` TE 0.0533 | +0.006599 |
| `sweep4` step 3 `inicializacion` | `muestras` 2.808934, TE 0.1067 | `uniforme` TE 0.0933 | +0.026171 |
| `sweep5` step 4 `radio` | `0.3` 2.456067, TE 0.0467 | `0.4` TE 0.0400 | +0.029710 |
| `sweep5` step 5 `decaimiento_radio` | `lineal` 2.456067, TE 0.0467 | `exponencial` TE 0.0333 | +0.032612 |

TE is also coarse: 1/50 = 0.02 per seed, so the 3-seed mean moves in steps of
0.0067. A TE difference of 0.0067–0.02 is one sample; it is not a reliable
discriminator.

### Consequence for `k=3`

Because `seleccionar_mejor` uses `idxmin`, exact ties are resolved by candidate
list order. That is why `k=3` ended with `uniforme`, `radio 0.4` and
`decaimiento_radio=constante` — not because they were better, but because they
appear first. QE is identical.

If a tie-breaker is wanted, **prefer the neutral/current value**
(`muestras`, `0.5`, `lineal`) rather than TE. That is deterministic,
QE-neutral, and avoids arbitrary changes. TE cannot do this because it is tied.
TE should remain a reported diagnostic only.

---

## 2. Why `sweep1`/`sweep2` improved so much

### Controlled comparison: the 27-variable fixed-k runs are also flat

| Run | k | Neurons | k²/n | QE start→end (sequential) | Total | Iterations range | Final dead/occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sweep1` | 10 | 100 | 2.00 | 1.414 → 0.377 | **73.4%** | 36.3% | 58/100, 0.42 |
| `sweep2` | 14 | 196 | 3.92 | 0.407 → 0.000 | **100.0%** | 100.0% | 146/196, 0.25 |
| 27v `k=3` | 3 | 9 | 0.18 | 3.501 → 3.487 | 0.40% | 0.2% | 0/9, 1.00 |
| 27v `k=4` | 4 | 16 | 0.32 | 3.190 → 3.181 | 0.29% | 0.7% | 0.67/16, 0.96 |
| 27v `k=5` | 5 | 25 | 0.50 | 2.906 → 2.833 | 2.52% | 0.5% | 4.33/25, 0.83 |
| PC10 `k=3` | 3 | 9 | 0.18 | 3.137 → 3.128 | 0.29% | 0.5% | 0/9, 1.00 |
| PC10 `k=4` | 4 | 16 | 0.32 | 2.816 → 2.809 | 0.23% | 1.3% | 0.67/16, 0.96 |
| PC10 `k=5` | 5 | 25 | 0.50 | 2.491 → 2.451 | 1.61% | 1.2% | 4.33/25, 0.83 |

The 27-variable fixed-k runs — same representation as `sweep1`/`sweep2` — were
already almost flat. **So PCA is not the cause.**

### Direct proof on the PC10 data

Same data, same schedule, only `k` changes. Radius-decay effect
(`lineal` → `inversa`; excluding the pathological `constante`):

| k | k²/n | QE `lineal` → `inversa` | Drop |
|---:|---:|---:|---:|
| 5 | 0.50 | 2.491 → 2.468 | 0.9% |
| 6 | 0.72 | 2.284 → 2.172 | 4.9% |
| 7 | 0.98 | 1.973 → 1.743 | 11.7% |
| 8 | 1.28 | 1.733 → 1.402 | 19.1% |
| 10 | 2.00 | 1.412 → 0.562 | **60.2%** |

Including `constante`, the range at `k=10` is 2.975 → 0.562 (81.1%).

And the decisive reproduction: with the **same PC10 data** and `sweep1`’s
`k=10` configuration:

| Configuration | QE |
|---|---:|
| `sweep1` (27 variables, `k=10`, radio 0.5, inversa, eta 0.5, 10k) | 0.532888 |
| PC10 (same configuration) | 0.541376 |
| `sweep1` final (eta 0.9, 80k) | 0.376768 |
| PC10 (same configuration) | 0.414713 |

The representation barely matters. The regime does.

### The mechanism, beyond “k is smaller”

Two scale effects compound:

1. **Capacity:** `sweep1`/`sweep2` had `k² > n` (100 or 196 neurons for 50
   samples). The map can approach one-neuron-per-sample; QE then measures how
   fast neighborhoods decouple. At `k ≤ 5`, `k² ≤ 25 < 50`; each neuron must
   average two or more samples, and schedules only fine-tune the coarse
   partition.
2. **Fixed final radius:** `R_f = 1.0` while `R0 = radio_inicial_relativo × k`.
   The schedule’s shrink factor `R0/R_f` grows with `k`: at `k=3`, `R0=1.2 →
   1.0` (20%); at `k=10`, `5 → 1` (80%). Also, `R_f=1` covers 40.7% of the map
   at `k=3`, 25% at `k=4`, 16.8% at `k=5`, and only 4.6% at `k=10`. Small maps
   never decouple enough for radius decay to matter.

That is the missing reason: it is not “small k” as a knob, but the
**under-capacity, globally-coupled regime** that the fixed-k methodology
deliberately selected. The old gains were the approach to a degenerate,
mostly-dead map: `sweep1` ended with 58/100 dead neurons, `sweep2` with 146/196
and QE 0.

### Other explanations ruled out

- **Iteration cap:** the fixed-k configs only test up to 25k, while
  `sweep1`/`sweep2` tested 80k. Rerunning fixed-k with 80k gives QE =
  3.138 / 2.818 / 2.432 for `k=3/4/5`, versus 3.110 / 2.807 / 2.451 at 10k.
  Changes are within seed noise; `k=5` improves by less than 1%, `k=3/4` get
  slightly worse. The cap is not the cause. (TE does tend to improve at 80k,
  especially `k=5`, but QE does not.)
- **Code change:** the training engine is unchanged between the archived sweeps
  and the current ones; the only diffs after the initial motor are plotting
  changes. Not a regression.
- **PC1 dominance / PCA scaling:** final-map squared error is spread across all
  10 PCs (PC1 contributes only 9–16%), and codebook variance allocation is
  nearly identical between the 27-variable and PC10 maps at the same `k`. This
  is only a second-order effect.

### Secondary representation effect

PC10 is marginally flatter than the 27-variable fixed-k runs (`k=5`: 1.61% vs
2.52%). The effective dimension is 3.09 vs 3.68, PC1 carries 54.3% vs 49.7%,
and the k-means floor is lower (≈2.6/2.1/1.5 vs ≈3.0/2.4/1.75). This is real but
small; it does not explain the 30–300× difference with `sweep1`/`sweep2`.

---

## 3. Recommendation

1. **Do not add a TE tie-breaker to selection.** It cannot break the observed
   exact ties and systematically costs QE in near-ties. Report TE alongside QE
   instead.
2. **If arbitrary exact ties are the concern**, add a “prefer neutral/current
   value on exact QE ties” rule. QE stays identical, and `k=3` would keep
   `muestras`, `0.5`, `lineal` instead of the list-order choices. This requires a
   small selection-logic change and a rerun to see downstream effects, but it is
   more defensible than TE.
3. **For the report**, state that QE is not comparable across `k` regimes and
   that the fixed-k flatness is expected: the map is capacity-limited and
   globally coupled. The large historical improvements are a degenerate
   over-capacity effect, not evidence of a better segmentation.

---

## 4. Would testing lower iterations make sense?

Yes, but as a **convergence diagnostic**, not as a sweep candidate. The
selected configs were rerun in memory:

| Iterations | k=3 QE | k=4 QE | k=5 QE |
|---:|---:|---:|---:|
| 100 | 3.389583 | 3.128763 | 2.706011 |
| 250 | 3.194840 | 3.041214 | 2.563848 |
| 500 | 3.172753 | 2.987686 | 2.546534 |
| 1,000 | 3.158200 | 2.868429 | 2.504022 |
| 2,000 | 3.178452 | 2.901523 | 2.533618 |
| 3,000 | 3.128129 | 2.831431 | 2.488518 |
| 5,000 | 3.146275 | 2.805224 | 2.481403 |
| 10,000 | 3.109693 | 2.806541 | 2.450829 |

- At 100 iterations, QE is only **9–11% higher** than the 10k plateau.
- By 1,000 iterations it is within **1.5–2.2%**; by 3,000 within **0.6–1.5%**.
- The curve is noisy after ~3k: at `k=4`, 5k is marginally better than 10k; at
  `k=3`, 3k is worse than 10k. These differences are within seed noise.

So lower iterations do show a non-converged state, but the effect is modest at
fixed `k`. The flatness at 5k–25k is **not** because the network is
undertrained; it reaches its plateau quickly.

The contrast with the over-capacity regime is much stronger:

| PC10, k=10 | QE |
|---|---:|
| 100 iterations | 1.939241 |
| 1,000 | 1.152680 |
| 5,000 | 0.524347 |
| 10,000 | 0.466891 |
| 80,000 | 0.414713 |

At `k=10`, 100 iterations is ~4× worse than the plateau. That is the regime
where iterations genuinely matter.

**Recommendation:** do not add 100–1,000 to the sweep; it only adds runtime and
noise, and the optimum remains 10k. If a visual is wanted, do a separate
convergence check with 3 seeds and iterations
`[100, 250, 500, 1000, 2500, 5000, 10000]` for `k=3`, `k=5` and `k=10`. Save
the CSV if it goes in the report: the project rule requires figures to be
regenerable from saved results. Note that this is a training script, not an
analysis-only script.

---

## 5. Is low QE impossible with k² buckets?

Not impossible, but there is a large **topology penalty**. An unconstrained
k-means lower bound was computed with exactly `k²` free centroids:

| Grid k | Buckets | Samples/bucket | PC10 floor (mean L2) | 27-var floor | SOM QE (PC10) | Gap above floor |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 9 | 5.56 | ≈2.59–2.65 | ≈3.00–3.02 | 3.109693 | +17–20% |
| 4 | 16 | 3.12 | ≈2.08–2.09 | ≈2.42–2.43 | 2.806541 | +34–35% |
| 5 | 25 | 2.00 | ≈1.45–1.54 | ≈1.74–1.76 | 2.450829 | +59–69% |
| 6 | 36 | 1.39 | ≈0.78 | ≈0.96 | 2.18* | +179% |
| 7 | 49 | 1.02 | ≈0.03 | ≈0.04 | 1.72* | orders of magnitude |

\*Aggressive in-memory schedule (`radio=0.3`, `inversa`, `eta=0.9`, 80k). The
floors are approximate; k-means restarts vary by ±0.05 at `k=3–5`.

Interpretation:

- At `k=3/4/5`, the k-means floor is a **hard lower bound**, and the SOM is
  **17–69% above it**. So there is theoretical room, but not “much lower” —
  less than a factor of two.
- The data composition matters: 50 points, 10 continuous PCs, no obvious
  discrete clusters. The floor falls smoothly as buckets increase
  (2.65 → 2.09 → 1.45 → 0.78 → 0.03). There is no natural low-QE plateau at
  9–25 buckets; the bucket count sets the floor.
- But the gap is not mainly the data: it is the **SOM grid topology**. Free
  k-means can go much lower, especially as `k²` approaches `n=50`; the SOM
  cannot, because neighboring neurons are forced to remain similar. At `k=7`
  the free floor is ~0.03, while the SOM is 1.72. At `k=10` the free floor is 0
  (100 centroids for 50 points), while the SOM is ~0.41–0.54.
- So “impossible with k² buckets” is too strong. It is **bounded** by the
  bucket count, and the SOM’s topology keeps it well above that bound. Getting
  substantially lower QE requires more buckets (and accepting a fragmented or
  degenerate map) or dropping the grid topology (k-means), not a different
  learning schedule at fixed `k`.

This is consistent with the earlier conclusion: the huge `sweep1`/`sweep2`
gains were the over-capacity regime approaching free quantization, not a better
topology-preserving map.

---

## 6. Min–max `[0,1]` QE

Interpreted as **min–max scaling each input feature to `[0,1]` before
training**. Same selected configs and 3 seeds, in memory only.

### PC1…PC10, per-feature min–max

| k | Current raw QE | Min–max `[0,1]` QE | Raw QE / mean pairwise | Min–max QE / mean pairwise |
|---:|---:|---:|---:|---:|
| 3 | 3.109693 ± 0.015957 | **0.591406 ± 0.006542** | 0.466 | 0.573 |
| 4 | 2.806541 ± 0.022015 | **0.555464 ± 0.005203** | 0.420 | 0.538 |
| 5 | 2.450829 ± 0.022026 | **0.467391 ± 0.006499** | 0.367 | 0.452 |

The absolute QE drops because the data range is now roughly `[0,1]` instead of
about 13 units wide. It does **not** mean the map is better.

Normalized by the mean pairwise distance, min–max is **23–28% worse** than raw
PC scores with the same configs. Reason: min–max reweights each PC by its range
instead of its standard deviation.

| PC | Raw weight (std share) | Min–max weight (range share) |
|---|---:|---:|
| PC1 | 27.2% | 23.8% |
| PC3 | 10.2% | 11.5% |
| PC4 | 9.3% | 10.6% |
| PC9 | 6.0% | 6.8% |

PC1 has the smallest `range/std` ratio (3.64), so min–max downweights the most
informative component. PC9 has the largest (4.76) and gains weight. Also,
min–max ranges are driven by outliers, which compresses the bulk of the data.

At `k=4`, the selected distance is `exponencial`. Forcing `euclidiana` on the
min–max data gives QE `0.548971`, so the exponential-distance translation
effect is minor; the reweighting is the main issue.

### If “everything” means one global min–max

If instead the whole matrix is scaled by the global min/max (one common factor,
not per feature), the transformation is a pure scale change for Euclidean
distance:

| k | Global min–max QE |
|---:|---:|
| 3 | 0.233186 ± 0.001197 |
| 4 | 0.211201 ± 0.000630 |
| 5 | 0.183780 ± 0.001652 |

That is essentially `QE_raw / 13.3357` (`k=4` differs slightly because of the
exponential distance under the shift). Relative QE is unchanged from raw.

### 27 variables, per-feature min–max

For reference, using the archived 27-variable configs:

| k | Min–max QE | Min–max normalized | Z-score normalized |
|---:|---:|---:|---:|
| 3 | 1.002934 ± 0.001327 | 0.507 | 0.493 |
| 4 | 0.881936 ± 0.005820 | 0.446 | 0.450 |
| 5 | 0.779027 ± 0.011140 | 0.394 | 0.398 |

Here min–max and z-score are essentially equivalent (within ~3%), because the
27 variables already have similar distributions and equal-ish weights. The PC10
case is different because the raw PCs carry very different variances.

### Caveat

These QEs use the hyperparameters selected under the raw representation. A
min–max sweep could choose different `radio_inicial_relativo`,
`decaimiento_radio`, `eta`, and iterations, and improve the min–max result
somewhat. To claim a definitive min–max QE, the sweep should be rerun on the
min–max input and compared using normalized QE, not raw QE.

---

## Reproduction and scope

- The TE, capacity and min–max analyses read the saved sweep CSVs and
  `resultados/ej1/pca_0.90/scores.csv`; they do not alter `datos/` or the PCA
  outputs.
- The lower-iteration and k-means-floor numbers are exploratory in-memory
  experiments. If they are used in the presentation, save the corresponding
  CSV (or a training script that writes it) first.
- The k-means floors are unconstrained lower bounds, not SOM results. They are
  useful to quantify the topology penalty but are not achievable by a
  topology-preserving map at the same bucket count.
- The min–max results use the configs selected under the raw representation;
  they are a scale/geometry comparison, not a re-tuned min–max optimum.
- All reported means use three seeds and sample standard deviation.
