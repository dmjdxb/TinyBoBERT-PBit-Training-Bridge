"""
Convergence diagnostics for MCMC and thermodynamic sampling.

This module provides rigorous diagnostics to verify that sampling
has converged and results are reliable. Without these checks,
results from thermodynamic sampling cannot be trusted.

Key diagnostics:
- Gelman-Rubin statistic (R̂): Tests convergence across chains
- Effective Sample Size (ESS): Measures statistical efficiency
- Geweke diagnostic: Tests stationarity within chains
- Heidelberger-Welch: Tests stationarity and half-width
- Autocorrelation: Measures sample correlation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from scipy import stats, signal
from scipy.stats import chi2


@dataclass
class ConvergenceResult:
    """Results from convergence diagnostics."""
    converged: bool
    r_hat: Optional[float] = None  # Gelman-Rubin statistic
    ess: Optional[float] = None  # Effective sample size
    ess_per_second: Optional[float] = None  # Efficiency metric
    geweke_z: Optional[float] = None  # Geweke z-score
    heidelberger_passed: Optional[bool] = None
    autocorr_time: Optional[float] = None  # Integrated autocorrelation time
    mcse: Optional[float] = None  # Monte Carlo standard error
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class GelmanRubinDiagnostic:
    """
    Gelman-Rubin convergence diagnostic (potential scale reduction factor).

    Tests whether multiple MCMC chains have converged to the same distribution.
    R̂ < 1.1 indicates convergence (some use R̂ < 1.01 for stricter convergence).

    Reference:
        Gelman & Rubin (1992) "Inference from Iterative Simulation Using Multiple Sequences"
    """

    def __init__(self, threshold: float = 1.1):
        """
        Args:
            threshold: R̂ threshold for convergence (typically 1.01-1.1)
        """
        self.threshold = threshold

    def compute(self, chains: np.ndarray, warmup_frac: float = 0.5) -> Tuple[float, bool]:
        """
        Compute Gelman-Rubin statistic.

        Args:
            chains: Array of shape (n_chains, n_samples) or (n_chains, n_samples, n_params)
            warmup_frac: Fraction of samples to discard as warmup

        Returns:
            r_hat: The potential scale reduction factor
            converged: Whether R̂ < threshold
        """
        if chains.ndim == 2:
            chains = chains[:, :, np.newaxis]

        n_chains, n_samples, n_params = chains.shape

        if n_chains < 2:
            raise ValueError("Need at least 2 chains for Gelman-Rubin diagnostic")

        # Discard warmup
        warmup = int(warmup_frac * n_samples)
        chains = chains[:, warmup:, :]
        n_samples = chains.shape[1]

        if n_samples < 100:
            warnings.warn(f"Only {n_samples} samples after warmup - may be unreliable")

        # Calculate R̂ for each parameter
        r_hats = []

        for p in range(n_params):
            param_chains = chains[:, :, p]

            # Between-chain variance
            chain_means = np.mean(param_chains, axis=1)
            grand_mean = np.mean(chain_means)
            B = n_samples * np.var(chain_means, ddof=1)

            # Within-chain variance
            chain_vars = np.var(param_chains, axis=1, ddof=1)
            W = np.mean(chain_vars)

            # Posterior variance estimate
            var_plus = ((n_samples - 1) / n_samples) * W + B / n_samples

            # Potential scale reduction factor
            r_hat = np.sqrt(var_plus / W) if W > 0 else float('inf')
            r_hats.append(r_hat)

        # Return maximum R̂ across all parameters
        max_r_hat = max(r_hats)
        converged = max_r_hat < self.threshold

        return max_r_hat, converged


class EffectiveSampleSize:
    """
    Calculate effective sample size accounting for autocorrelation.

    ESS measures how many independent samples the chain contains.
    Low ESS indicates high autocorrelation and poor mixing.

    Reference:
        Gelman et al. (2013) "Bayesian Data Analysis", 3rd edition
    """

    def __init__(self, min_ess: int = 100):
        """
        Args:
            min_ess: Minimum ESS for adequate sampling
        """
        self.min_ess = min_ess

    def compute_autocorrelation(self, x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute autocorrelation function.

        Args:
            x: Time series data
            max_lag: Maximum lag to compute (default: len(x) // 4)

        Returns:
            Autocorrelation at each lag
        """
        if max_lag is None:
            max_lag = len(x) // 4

        x = x - np.mean(x)
        c0 = np.dot(x, x) / len(x)

        acf = np.zeros(max_lag)
        for k in range(max_lag):
            if k == 0:
                acf[k] = 1.0
            else:
                ck = np.dot(x[:-k], x[k:]) / (len(x) - k)
                acf[k] = ck / c0 if c0 > 0 else 0

        return acf

    def compute_integrated_autocorr_time(self, x: np.ndarray) -> float:
        """
        Compute integrated autocorrelation time using Geyer's initial monotone sequence.

        Args:
            x: Time series data

        Returns:
            Integrated autocorrelation time (τ)
        """
        acf = self.compute_autocorrelation(x)

        # Geyer's initial positive sequence estimator
        tau = 1.0  # acf[0] = 1

        for i in range(1, len(acf) - 1, 2):
            pair_sum = acf[i] + acf[i + 1]
            if pair_sum > 0:
                tau += 2 * pair_sum
            else:
                break

        return tau

    def compute(self, samples: np.ndarray) -> Tuple[float, bool]:
        """
        Compute effective sample size.

        Args:
            samples: MCMC samples (1D or 2D array)

        Returns:
            ess: Effective sample size
            adequate: Whether ESS >= min_ess
        """
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]

        n_samples, n_params = samples.shape

        # Compute ESS for each parameter
        ess_values = []

        for p in range(n_params):
            param_samples = samples[:, p]
            tau = self.compute_integrated_autocorr_time(param_samples)
            ess = n_samples / tau if tau > 0 else n_samples
            ess_values.append(ess)

        # Return minimum ESS across parameters
        min_ess_value = min(ess_values)
        adequate = min_ess_value >= self.min_ess

        return min_ess_value, adequate


class MCMCError:
    """
    Monte Carlo standard error estimation.

    MCSE quantifies the uncertainty in MCMC estimates due to
    finite sampling. Smaller MCSE indicates more precise estimates.
    """

    def compute_batch_means(self, x: np.ndarray, batch_size: Optional[int] = None) -> float:
        """
        Compute MCSE using batch means method.

        Args:
            x: MCMC samples
            batch_size: Size of batches (default: sqrt(n))

        Returns:
            Monte Carlo standard error
        """
        n = len(x)

        if batch_size is None:
            batch_size = int(np.sqrt(n))

        n_batches = n // batch_size

        if n_batches < 2:
            return np.std(x) / np.sqrt(n)

        # Create batches
        x_batched = x[:n_batches * batch_size].reshape(n_batches, batch_size)
        batch_means = np.mean(x_batched, axis=1)

        # MCSE from batch means
        mcse = np.std(batch_means) / np.sqrt(n_batches)

        return mcse

    def compute_spectral(self, x: np.ndarray, ess: float) -> float:
        """
        Compute MCSE using effective sample size.

        Args:
            x: MCMC samples
            ess: Effective sample size

        Returns:
            Monte Carlo standard error
        """
        return np.std(x) / np.sqrt(ess)


class GewekeDiagnostic:
    """
    Geweke convergence diagnostic.

    Tests whether the mean of the first portion of the chain
    equals the mean of the last portion. Under convergence,
    the z-score should be normally distributed.

    Reference:
        Geweke (1992) "Evaluating the accuracy of sampling-based approaches"
    """

    def __init__(self, first_frac: float = 0.1, last_frac: float = 0.5):
        """
        Args:
            first_frac: Fraction of chain to use from beginning
            last_frac: Fraction of chain to use from end
        """
        self.first_frac = first_frac
        self.last_frac = last_frac

    def compute(self, x: np.ndarray) -> Tuple[float, bool]:
        """
        Compute Geweke z-score.

        Args:
            x: MCMC samples

        Returns:
            z_score: Geweke statistic
            converged: Whether |z| < 2 (95% confidence)
        """
        n = len(x)

        # Split chain
        n_first = int(self.first_frac * n)
        n_last = int(self.last_frac * n)

        first_samples = x[:n_first]
        last_samples = x[-n_last:]

        # Compute means
        mean_first = np.mean(first_samples)
        mean_last = np.mean(last_samples)

        # Compute spectral density estimates at frequency 0
        # Using simple variance for now (could use more sophisticated spectral methods)
        var_first = np.var(first_samples)
        var_last = np.var(last_samples)

        # Standard error
        se = np.sqrt(var_first / n_first + var_last / n_last)

        # Z-score
        z_score = (mean_first - mean_last) / se if se > 0 else 0

        # Test at 95% confidence level
        converged = abs(z_score) < 1.96

        return z_score, converged


class HeidelbergerWelchTest:
    """
    Heidelberger-Welch stationarity and half-width test.

    Tests whether the chain has reached stationarity and whether
    the sample size is adequate for the desired precision.

    Reference:
        Heidelberger & Welch (1983) "Simulation run length control"
    """

    def __init__(self, alpha: float = 0.05, epsilon: float = 0.1):
        """
        Args:
            alpha: Significance level for stationarity test
            epsilon: Relative precision for half-width test
        """
        self.alpha = alpha
        self.epsilon = epsilon

    def cramer_von_mises_test(self, x: np.ndarray) -> bool:
        """
        Simplified Cramér-von Mises test for stationarity.

        Args:
            x: Time series

        Returns:
            Whether the series appears stationary
        """
        n = len(x)

        # Compute cumulative sums
        cumsum = np.cumsum(x - np.mean(x))

        # Brownian bridge transformation
        bridge = cumsum - np.linspace(0, cumsum[-1], n)

        # Test statistic (simplified)
        statistic = np.sum(bridge**2) / (n * np.var(x))

        # Critical value approximation
        critical_value = 0.461  # For alpha=0.05

        return statistic < critical_value

    def compute(self, x: np.ndarray) -> Tuple[bool, bool]:
        """
        Perform Heidelberger-Welch test.

        Args:
            x: MCMC samples

        Returns:
            stationary: Whether chain appears stationary
            halfwidth_passed: Whether precision criterion is met
        """
        # Stationarity test
        stationary = self.cramer_von_mises_test(x)

        # Half-width test
        mean_est = np.mean(x)
        mcse = np.std(x) / np.sqrt(len(x))
        halfwidth = 1.96 * mcse  # 95% confidence interval

        # Check relative precision
        halfwidth_passed = (halfwidth / abs(mean_est)) < self.epsilon if mean_est != 0 else True

        return stationary, halfwidth_passed


class ConvergenceDiagnostics:
    """
    Comprehensive convergence diagnostics for MCMC/thermodynamic sampling.

    This class combines multiple diagnostic tests to provide a thorough
    assessment of whether sampling has converged.
    """

    def __init__(
        self,
        r_hat_threshold: float = 1.1,
        min_ess: int = 100,
        geweke_alpha: float = 0.05,
        heidelberger_epsilon: float = 0.1
    ):
        """
        Args:
            r_hat_threshold: Gelman-Rubin threshold
            min_ess: Minimum effective sample size
            geweke_alpha: Significance level for Geweke test
            heidelberger_epsilon: Precision for Heidelberger-Welch test
        """
        self.gelman_rubin = GelmanRubinDiagnostic(r_hat_threshold)
        self.ess_calculator = EffectiveSampleSize(min_ess)
        self.mcse_calculator = MCMCError()
        self.geweke = GewekeDiagnostic()
        self.heidelberger = HeidelbergerWelchTest(geweke_alpha, heidelberger_epsilon)

    def diagnose_single_chain(
        self,
        samples: np.ndarray,
        runtime_seconds: Optional[float] = None
    ) -> ConvergenceResult:
        """
        Run diagnostics on a single chain.

        Args:
            samples: MCMC samples (1D or 2D)
            runtime_seconds: Time taken to generate samples

        Returns:
            ConvergenceResult with diagnostic information
        """
        result = ConvergenceResult(converged=True)

        if samples.ndim == 1:
            samples = samples[:, np.newaxis]

        # Effective sample size
        ess, ess_adequate = self.ess_calculator.compute(samples)
        result.ess = ess
        if runtime_seconds is not None:
            result.ess_per_second = ess / runtime_seconds

        if not ess_adequate:
            result.converged = False
            result.warnings.append(f"Low ESS: {ess:.1f} < {self.ess_calculator.min_ess}")

        # Autocorrelation time
        tau = len(samples) / ess if ess > 0 else float('inf')
        result.autocorr_time = tau

        # Monte Carlo standard error
        result.mcse = self.mcse_calculator.compute_spectral(samples[:, 0], ess)

        # Geweke diagnostic
        z_score, geweke_passed = self.geweke.compute(samples[:, 0])
        result.geweke_z = z_score
        if not geweke_passed:
            result.converged = False
            result.warnings.append(f"Geweke test failed: |z|={abs(z_score):.2f} > 1.96")

        # Heidelberger-Welch test
        stationary, halfwidth_passed = self.heidelberger.compute(samples[:, 0])
        result.heidelberger_passed = stationary and halfwidth_passed
        if not stationary:
            result.converged = False
            result.warnings.append("Heidelberger-Welch stationarity test failed")
        if not halfwidth_passed:
            result.warnings.append("Heidelberger-Welch half-width test failed")

        return result

    def diagnose_multiple_chains(
        self,
        chains: List[np.ndarray],
        runtime_seconds: Optional[float] = None
    ) -> ConvergenceResult:
        """
        Run diagnostics on multiple chains.

        Args:
            chains: List of MCMC sample arrays
            runtime_seconds: Total runtime for all chains

        Returns:
            ConvergenceResult with diagnostic information
        """
        result = ConvergenceResult(converged=True)

        # Stack chains
        max_len = max(len(c) for c in chains)
        chains_array = np.zeros((len(chains), max_len, chains[0].shape[-1] if chains[0].ndim > 1 else 1))
        for i, chain in enumerate(chains):
            if chain.ndim == 1:
                chain = chain[:, np.newaxis]
            chains_array[i, :len(chain), :] = chain

        # Gelman-Rubin diagnostic
        r_hat, gr_converged = self.gelman_rubin.compute(chains_array)
        result.r_hat = r_hat
        if not gr_converged:
            result.converged = False
            result.warnings.append(f"R̂={r_hat:.3f} > {self.gelman_rubin.threshold}")

        # Pool chains for other diagnostics
        pooled = np.vstack(chains)

        # ESS on pooled samples
        ess, ess_adequate = self.ess_calculator.compute(pooled)
        result.ess = ess
        if runtime_seconds is not None:
            result.ess_per_second = ess / runtime_seconds

        if not ess_adequate:
            result.converged = False
            result.warnings.append(f"Low ESS: {ess:.1f} < {self.ess_calculator.min_ess}")

        # MCSE on pooled samples
        result.mcse = self.mcse_calculator.compute_spectral(pooled[:, 0], ess)

        # Individual chain diagnostics
        for i, chain in enumerate(chains):
            chain_result = self.diagnose_single_chain(chain)
            if not chain_result.converged:
                result.converged = False
                result.warnings.append(f"Chain {i} failed convergence")

        return result

    def recommend_sampling_params(self, result: ConvergenceResult) -> Dict[str, Any]:
        """
        Recommend sampling parameters based on diagnostics.

        Args:
            result: Convergence diagnostic results

        Returns:
            Dictionary of recommended parameters
        """
        recommendations = {}

        if result.r_hat is not None and result.r_hat > 1.1:
            recommendations['increase_warmup'] = True
            recommendations['suggested_warmup_multiplier'] = 2.0

        if result.ess is not None and result.ess < self.ess_calculator.min_ess:
            increase_factor = self.ess_calculator.min_ess / result.ess
            recommendations['increase_samples'] = True
            recommendations['suggested_sample_multiplier'] = max(2.0, increase_factor)

        if result.autocorr_time is not None and result.autocorr_time > 100:
            recommendations['improve_mixing'] = True
            recommendations['suggested_thinning'] = int(result.autocorr_time / 10)

        if abs(result.geweke_z or 0) > 1.96:
            recommendations['check_stationarity'] = True
            recommendations['suggested_diagnostics'] = ['trace plots', 'running means']

        return recommendations


def quick_convergence_check(
    samples: Union[np.ndarray, List[np.ndarray]],
    verbose: bool = True
) -> bool:
    """
    Quick convergence check with default parameters.

    Args:
        samples: Single chain or list of chains
        verbose: Whether to print diagnostic information

    Returns:
        Whether the samples appear to have converged
    """
    diagnostics = ConvergenceDiagnostics()

    if isinstance(samples, list):
        result = diagnostics.diagnose_multiple_chains(samples)
    else:
        result = diagnostics.diagnose_single_chain(samples)

    if verbose:
        print("=== Convergence Diagnostics ===")
        if result.r_hat is not None:
            print(f"Gelman-Rubin R̂: {result.r_hat:.3f}")
        if result.ess is not None:
            print(f"Effective Sample Size: {result.ess:.1f}")
        if result.mcse is not None:
            print(f"Monte Carlo Std Error: {result.mcse:.4f}")
        if result.geweke_z is not None:
            print(f"Geweke z-score: {result.geweke_z:.2f}")

        print(f"\nConverged: {result.converged}")
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

    return result.converged