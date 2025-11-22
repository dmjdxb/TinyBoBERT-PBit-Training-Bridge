"""
Thermalization dynamics and mixing time analysis for TSU systems.

This module provides rigorous analysis of:
1. Time to reach thermal equilibrium
2. Autocorrelation in MCMC samples
3. Effective sample size calculations
4. Detailed balance verification

These are critical for understanding the actual time and energy costs
of thermodynamic computing, which are often ignored in idealized models.

References:
    [1] Levin & Peres (2017). "Markov Chains and Mixing Times"
    [2] Sokal (1997). "Monte Carlo Methods in Statistical Mechanics"
    [3] Newman & Barkema (1999). "Monte Carlo Methods in Statistical Physics"
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import warnings
from scipy import stats


@dataclass
class SystemParameters:
    """Parameters affecting thermalization dynamics."""

    system_size: int  # Number of spins/bits
    coupling_strength: float  # Typical J value
    temperature: float  # In units where k_B = 1
    dimension: int = 2  # System dimension (1D, 2D, 3D)

    @property
    def beta(self) -> float:
        """Inverse temperature."""
        return 1.0 / self.temperature

    @property
    def energy_scale(self) -> float:
        """Characteristic energy scale."""
        return self.coupling_strength * self.system_size

    @property
    def critical_slowing_down(self) -> bool:
        """Check if near critical temperature (2D Ising: T_c ≈ 2.269)."""
        if self.dimension == 2:
            T_c = 2.269185  # Onsager's exact solution
            return abs(self.temperature - T_c) / T_c < 0.1
        return False


class ThermalizationModel:
    """
    Model finite thermalization times and sampling efficiency.

    Real TSU devices don't instantly reach equilibrium - they require
    time to explore the energy landscape through thermal fluctuations.
    """

    def __init__(self, params: SystemParameters):
        self.params = params

        # Physical time scales (in seconds)
        self.single_flip_time = 1e-9  # 1 nanosecond per spin flip
        self.measurement_time = 1e-8  # 10 ns per measurement

        # Warn about critical slowing down
        if self.params.critical_slowing_down:
            warnings.warn(
                f"System near critical temperature! Mixing time will be very long. "
                f"T = {self.params.temperature}, T_c ≈ 2.269 for 2D Ising.",
                category=UserWarning
            )

    def estimate_mixing_time(self) -> Dict[str, float]:
        """
        Estimate time to reach thermal equilibrium.

        For Glauber dynamics: τ_mix ~ N * exp(ΔE/T)
        where ΔE is the typical energy barrier.
        """
        N = self.params.system_size
        J = self.params.coupling_strength
        T = self.params.temperature

        # Energy barriers depend on dimension
        if self.params.dimension == 1:
            # 1D: No phase transition, fast mixing
            energy_barrier = 2 * J  # Single bond breaking
            mixing_steps = N * np.log(N)  # O(N log N) mixing

        elif self.params.dimension == 2:
            # 2D: Phase transition at T_c
            if T > 2.269:  # Above T_c: fast mixing
                energy_barrier = 4 * J  # Small droplet
                mixing_steps = N * np.log(N)
            else:  # Below T_c: slow mixing due to large droplets
                energy_barrier = 2 * J * np.sqrt(N)  # Domain wall
                mixing_steps = N * np.exp(energy_barrier / T)

        else:  # 3D or higher
            # 3D: Stronger ordering, slower mixing
            energy_barrier = 6 * J  # Face of cube
            mixing_steps = N * np.exp(energy_barrier / T)

        # Account for critical slowing down
        if self.params.critical_slowing_down:
            # Near T_c: τ ~ |T - T_c|^(-zν) with zν ≈ 2
            T_c = 2.269185
            mixing_steps *= (abs(T - T_c) / T_c) ** (-2)

        # Convert to real time
        mixing_time = mixing_steps * self.single_flip_time

        return {
            'mixing_steps': mixing_steps,
            'mixing_time_seconds': mixing_time,
            'energy_barrier': energy_barrier,
            'critical_slowing': self.params.critical_slowing_down
        }

    def autocorrelation_function(
        self,
        trajectory: np.ndarray,
        max_lag: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute autocorrelation function for a trajectory.

        C(t) = <x(0)x(t)> - <x>²

        This tells us how correlated samples are over time.
        """
        if max_lag is None:
            max_lag = min(len(trajectory) // 4, 1000)

        mean = np.mean(trajectory)
        var = np.var(trajectory)

        if var == 0:
            return np.zeros(max_lag)

        # Compute autocorrelation using FFT (fast)
        n = len(trajectory)
        trajectory_centered = trajectory - mean

        # Pad to power of 2 for FFT efficiency
        n_padded = 2 ** int(np.ceil(np.log2(2 * n)))
        trajectory_padded = np.zeros(n_padded)
        trajectory_padded[:n] = trajectory_centered

        # Autocorrelation via FFT
        fft = np.fft.fft(trajectory_padded)
        power = np.abs(fft) ** 2
        acf = np.fft.ifft(power).real[:n]

        # Normalize
        acf = acf / acf[0]

        return acf[:max_lag]

    def integrated_autocorrelation_time(
        self,
        trajectory: np.ndarray,
        method: str = 'sokal'
    ) -> float:
        """
        Compute integrated autocorrelation time τ_int.

        τ_int = 1 + 2 * Σ_{t=1}^∞ C(t)

        This is the key quantity for effective sample size:
        N_eff = N / τ_int
        """
        acf = self.autocorrelation_function(trajectory)

        if method == 'sokal':
            # Sokal's adaptive truncation method
            # Stop when C(t) becomes noisy
            tau_int = 1.0
            for t in range(1, len(acf)):
                if acf[t] < 0:
                    break
                tau_int += 2 * acf[t]
                # Sokal's criterion: stop when t > c * tau_int
                if t > 6 * tau_int:
                    break

        elif method == 'simple':
            # Simple cutoff at first negative value
            first_negative = np.where(acf < 0)[0]
            if len(first_negative) > 0:
                cutoff = first_negative[0]
            else:
                cutoff = len(acf)

            tau_int = 1 + 2 * np.sum(acf[1:cutoff])

        else:
            raise ValueError(f"Unknown method: {method}")

        return max(tau_int, 1.0)  # Ensure at least 1

    def effective_sample_size(
        self,
        trajectory: np.ndarray,
        tau_int: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute effective sample size accounting for correlations.

        Real MCMC samples are correlated, so N samples don't provide
        N independent measurements. The effective sample size is:
        N_eff = N / τ_int
        """
        n_samples = len(trajectory)

        if tau_int is None:
            tau_int = self.integrated_autocorrelation_time(trajectory)

        n_eff = n_samples / tau_int

        # Statistical efficiency
        efficiency = n_eff / n_samples

        # Time to get one effective sample
        time_per_effective_sample = tau_int * self.single_flip_time

        return {
            'n_samples': n_samples,
            'tau_int': tau_int,
            'n_eff': n_eff,
            'efficiency': efficiency,
            'time_per_eff_sample': time_per_effective_sample
        }

    def sampling_requirements(
        self,
        target_error: float = 0.01,
        observable: str = 'magnetization'
    ) -> Dict[str, float]:
        """
        Calculate sampling requirements for target accuracy.

        Monte Carlo error: σ_MC = σ / sqrt(N_eff)
        where σ is the standard deviation of the observable.
        """
        # Typical variances for different observables
        if observable == 'magnetization':
            # Near T_c, susceptibility diverges: χ ~ |T - T_c|^(-γ)
            if self.params.critical_slowing_down:
                variance = 1.0  # Order 1 near critical point
            else:
                variance = 1.0 / self.params.system_size  # Central limit theorem

        elif observable == 'energy':
            # Energy fluctuations: <E²> - <E>² = T² * C_v
            variance = self.params.temperature ** 2 * self.params.system_size

        else:
            variance = 1.0  # Default

        # Number of effective samples needed
        n_eff_required = variance / (target_error ** 2)

        # Account for autocorrelation
        mixing_info = self.estimate_mixing_time()
        tau_int_estimate = mixing_info['mixing_steps'] / self.params.system_size

        # Total samples needed
        n_samples_required = n_eff_required * tau_int_estimate

        # Total time required
        total_time = n_samples_required * self.single_flip_time

        # Energy cost
        energy_per_flip = 1e-14  # 10 fJ per spin flip
        total_energy = n_samples_required * energy_per_flip

        return {
            'target_error': target_error,
            'n_eff_required': n_eff_required,
            'tau_int_estimate': tau_int_estimate,
            'n_samples_required': n_samples_required,
            'total_time_seconds': total_time,
            'total_energy_joules': total_energy,
            'efficiency_warning': self.params.critical_slowing_down
        }


class DetailedBalanceChecker:
    """
    Verify detailed balance condition for MCMC samplers.

    Detailed balance is essential for correct equilibrium distribution:
    π(x) * P(x→y) = π(y) * P(y→x)
    """

    def __init__(self, energy_function, beta: float = 1.0):
        self.energy_fn = energy_function
        self.beta = beta

    def verify_transition_probabilities(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
        transition_prob_fn
    ) -> Dict[str, float]:
        """
        Check if transition probabilities satisfy detailed balance.

        Args:
            state1: First state
            state2: Second state
            transition_prob_fn: Function computing P(state1 → state2)

        Returns:
            Dictionary with balance ratio and error
        """
        # Energies
        E1 = self.energy_fn(state1)
        E2 = self.energy_fn(state2)

        # Boltzmann weights
        pi1 = np.exp(-self.beta * E1)
        pi2 = np.exp(-self.beta * E2)

        # Transition probabilities
        P_12 = transition_prob_fn(state1, state2)
        P_21 = transition_prob_fn(state2, state1)

        # Detailed balance check
        # π(1) * P(1→2) should equal π(2) * P(2→1)
        forward_flow = pi1 * P_12
        backward_flow = pi2 * P_21

        if backward_flow > 0:
            balance_ratio = forward_flow / backward_flow
            relative_error = abs(balance_ratio - 1.0)
        else:
            balance_ratio = np.inf
            relative_error = np.inf

        # Alternative check: P_12/P_21 should equal π2/π1 = exp(-β(E2-E1))
        if P_21 > 0:
            ratio_measured = P_12 / P_21
            ratio_expected = np.exp(-self.beta * (E2 - E1))
            ratio_error = abs(ratio_measured - ratio_expected) / ratio_expected
        else:
            ratio_error = np.inf

        return {
            'forward_flow': forward_flow,
            'backward_flow': backward_flow,
            'balance_ratio': balance_ratio,
            'relative_error': relative_error,
            'ratio_error': ratio_error,
            'satisfied': relative_error < 1e-10
        }

    def check_ergodicity(
        self,
        trajectory: np.ndarray,
        state_space_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check if Markov chain is ergodic (all states reachable).

        For finite state space, we can check if all states are visited.
        """
        unique_states = np.unique(trajectory, axis=0)
        n_visited = len(unique_states)

        result = {
            'n_states_visited': n_visited,
            'trajectory_length': len(trajectory)
        }

        if state_space_size is not None:
            coverage = n_visited / state_space_size
            result['state_space_size'] = state_space_size
            result['coverage'] = coverage
            result['fully_ergodic'] = coverage > 0.99

        return result


class MixingTimeEstimator:
    """
    Advanced mixing time estimation using multiple methods.
    """

    @staticmethod
    def conductance_bound(
        transition_matrix: np.ndarray,
        stationary_dist: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate mixing time using conductance (Cheeger constant).

        τ_mix ≤ 2/Φ² * log(1/π_min)
        where Φ is the conductance.
        """
        n = len(transition_matrix)

        if stationary_dist is None:
            # Find stationary distribution (left eigenvector with eigenvalue 1)
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
            idx = np.argmax(np.abs(eigenvalues - 1.0) < 1e-10)
            stationary_dist = np.abs(eigenvectors[:, idx])
            stationary_dist /= stationary_dist.sum()

        # Compute conductance
        conductance = np.inf
        for subset_size in range(1, n // 2 + 1):
            # Try random subsets
            for _ in range(min(100, 2**subset_size)):
                S = np.random.choice(n, subset_size, replace=False)
                S_comp = np.setdiff1d(np.arange(n), S)

                # Probability mass in S
                pi_S = stationary_dist[S].sum()

                if pi_S > 0 and pi_S <= 0.5:
                    # Flow from S to S^c
                    flow = 0
                    for i in S:
                        for j in S_comp:
                            flow += stationary_dist[i] * transition_matrix[i, j]

                    # Conductance for this subset
                    phi_S = flow / pi_S
                    conductance = min(conductance, phi_S)

        # Mixing time bound
        if conductance > 0:
            pi_min = stationary_dist.min()
            mixing_time = (2 / conductance**2) * np.log(1 / pi_min)
        else:
            mixing_time = np.inf

        return mixing_time

    @staticmethod
    def spectral_gap(transition_matrix: np.ndarray) -> float:
        """
        Estimate mixing time from spectral gap.

        τ_mix ~ 1/(1 - λ₂) * log(1/ε)
        where λ₂ is the second largest eigenvalue.
        """
        eigenvalues = np.linalg.eigvals(transition_matrix)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        # Second largest eigenvalue
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0

        # Spectral gap
        gap = 1 - lambda_2

        if gap > 0:
            # Mixing time (to accuracy ε = 0.01)
            mixing_time = (1 / gap) * np.log(100)
        else:
            mixing_time = np.inf

        return mixing_time


class EffectiveSampleSize:
    """
    Multiple methods for computing effective sample size.
    """

    @staticmethod
    def batch_means(trajectory: np.ndarray, batch_size: Optional[int] = None) -> float:
        """
        Estimate effective sample size using batch means method.
        """
        n = len(trajectory)

        if batch_size is None:
            # Rule of thumb: batch size ~ sqrt(n)
            batch_size = int(np.sqrt(n))

        n_batches = n // batch_size
        if n_batches < 2:
            return 1.0

        # Compute batch means
        batch_means = []
        for i in range(n_batches):
            batch = trajectory[i * batch_size:(i + 1) * batch_size]
            batch_means.append(np.mean(batch))

        batch_means = np.array(batch_means)

        # Variance of batch means
        var_batch_means = np.var(batch_means)

        # Variance of original data
        var_data = np.var(trajectory)

        if var_batch_means > 0:
            # ESS = n * var(data) / (batch_size * var(batch_means))
            n_eff = n * var_data / (batch_size * var_batch_means)
        else:
            n_eff = n

        return min(n_eff, n)  # Can't be more than n

    @staticmethod
    def spectral_density(trajectory: np.ndarray) -> float:
        """
        Estimate effective sample size from spectral density at frequency 0.
        """
        n = len(trajectory)

        # Compute periodogram
        trajectory_centered = trajectory - np.mean(trajectory)
        fft = np.fft.fft(trajectory_centered)
        periodogram = np.abs(fft) ** 2 / n

        # Spectral density at frequency 0
        s_0 = periodogram[0]

        # Variance
        var = np.var(trajectory)

        if s_0 > 0:
            # ESS = n * var / s_0
            n_eff = n * var / s_0
        else:
            n_eff = n

        return min(n_eff, n)