# Window Selection Algorithm via Fano Factor

## Overview
This algorithm processes a sequence of continuous event arrival times to identify an optimal operational time window from a user-specified list of candidate business operating periods (e.g., 30 minutes, 60 minutes, 8 hours, 24 hours). 

To ensure statistical validity, all time calculations are mapped internally to a consistent, uniform unit of time. If a candidate operating window does not contain a clean, integer number of base aggregation intervals, the algorithm dynamically adjusts the base interval downward to eliminate fractional periods. A strict sample size constraint ($n \ge 30$) is enforced to automatically prune windows that lack sufficient data. 

Rather than requiring a Fano factor of exactly 1.0, the algorithm calculates a dynamic, sample-size-dependent **Acceptable Range** that includes 1.0. The strictness of this range is controlled by an explicit **Design Parameter ($\gamma$)** representing the target confidence level, which defaults to **90%**. Any window whose Fano factor falls within this range is considered an acceptable fit for a Poisson assumption.

---

## 1. Mathematical Notation & Inputs

### Core Data Inputs
* **Event Log ($T$):** A sorted, one-dimensional array of raw arrival timestamps:
  $$T = [t_1, t_2, t_3, \dots, t_m]$$
* **Total Time Span ($T_{total}$):** The entire duration of data collection:
  $$T_{total} = t_m - t_1$$
* **Target Windows ($W$):** A user-supplied list of business operating periods to evaluate:
  $$W = [W_1, W_2, W_3, \dots, W_j]$$
* **Initial Aggregation Unit ($\Delta t_{init}$):** The baseline unit of time preferred by the user (e.g., 1 minute).

### Configurable Design Parameters
* **Confidence Level ($\gamma$):** The statistical confidence level used to build the acceptable envelope. 
  * *Default value:* $\gamma = 0.90$ (90% confidence)
  * The implementation exposes this parameter as `confidence`; pass a value between 0 and 1 to override the default.
* **Significance Level ($\alpha$):** Derived directly from the confidence level design parameter:
  $$\alpha = 1 - \gamma \quad (\text{Default: } \alpha = 0.10)$$

---

## 2. Dynamic Window Alignment & Bounding Rules

For each candidate target window $W_j$ in the list, the algorithm applies two checks before processing:

### A. Integral Alignment Rule (No Fractional Windows)
The target window must be an exact integer multiple ($k$) of the operational base aggregation window ($\Delta t_{base}$). If the initial window $\Delta t_{init}$ results in a remainder, the algorithm automatically scales the operational base window down to the nearest factor of $W_j$:

$$\text{If } \left( W_j \pmod{\Delta t_{init}} \neq 0 \right): \quad \Delta t_{base} = \frac{W_j}{\left\lceil W_j / \Delta t_{init} \right\rceil}$$
$$\text{Else}: \quad \Delta t_{base} = \Delta t_{init}$$

This ensures that the target window always comprises exactly $k$ complete, uniform aggregation periods:
$$k = \frac{W_j}{\Delta t_{base}} \quad \text{where } k \in \mathbb{Z}^+$$

### B. Sample Size Bounding Rule ($n \ge 30$)
To guarantee stable statistical estimates, the total timeline must yield a minimum sample size of **30 independent, non-overlapping windows** at the target scale.
$$n = \left\lfloor \frac{T_{total}}{W_j} \right\rfloor$$
* **Pruning Rule:** If $n < 30$ for a given $W_j$, the window is flagged as "insufficient data" and skipped. This automatically bounds your search space based on the size of your event log.

---

## 3. Step-by-Step Process Architecture

### Step 1: Iterate Through Target Windows
For each target window $W_j$ in the user-provided list $W$:
1. Calculate the maximum available non-overlapping windows: $n = \lfloor T_{total} / W_j \rfloor$.
2. If $n < 30$, reject $W_j$ due to insufficient data and proceed to the next target window.
3. Apply the **Integral Alignment Rule** to compute the uniform operational $\Delta t_{base}$ and its integer multiple $k$.

### Step 2: Segment and Discretize Timeline
Segment the continuous timeline into $n$ adjacent, non-overlapping intervals of length $W_j$. Map the continuous arrival timestamps into discrete, integral event counts ($N_i$) for each interval $i$ to generate a count array:
$$N = [N_1, N_2, N_3, \dots, N_n]$$
where $N_i$ represents the number of events satisfying: $(i-1)W_j \le t_m < i \cdot W_j$.

### Step 3: Compute Sample Fano Factor
Using the discrete count array $N$, compute the sample mean ($\bar{N}$) and unbiased sample variance ($S^2$):
$$\bar{N} = \frac{1}{n}\sum_{i=1}^{n}N_i \quad \text{and} \quad S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(N_i - \bar{N})^2$$
Calculate the sample Fano factor ($F$) for the target window:
$$F(W_j) = \frac{S^2}{\bar{N}}$$

### Step 4: Establish the Parameterized Acceptable Range
Calculate the mathematical bounds of the **Acceptable Range** using a Two-Sided Chi-Squared test parameterized by your chosen significance level $\alpha$ (derived from design parameter $\gamma$) and the current sample size $n$:

$$\text{Lower Limit}(W_j) = \frac{\chi^2_{\alpha/2, \; n-1}}{n-1}$$
$$\text{Upper Limit}(W_j) = \frac{\chi^2_{1-\alpha/2, \; n-1}}{n-1}$$

*Note: This mathematical interval is centered around 1.0. Adjusting the design parameter $\gamma$ down (e.g., from 95% to 90%) narrows these boundaries, enforcing a stricter test for true Poisson stationarity.*

### Step 5: Evaluate Acceptable Poisson Operating Windows
Compare the calculated sample Fano factor $F(W_j)$ against the Acceptable Range established in Step 4.

$$\text{Lower Limit}(W_j) \le F(W_j) \le \text{Upper Limit}(W_j)$$

* **The Verdict:** A target business window $W_j$ is classified as an **acceptable Poisson window** if its Fano factor sits completely inside the calculated limits. Within this range, any variation away from 1.0 is statistically insignificant at the configured design threshold, meaning the Poisson assumption is fully valid for that interval size.
