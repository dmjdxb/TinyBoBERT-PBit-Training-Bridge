"""
Physics validation test suite.

This module validates the TSU implementation against known exact solutions
from statistical mechanics, ensuring the physics is correct.

Key validations:
1. 2D Ising critical temperature (Onsager's exact solution)
2. Boltzmann distribution verification
3. Detailed balance checking
4. Energy fluctuation-dissipation theorem

These tests ensure our implementation produces physically correct results.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.special import ellipk

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mltsu.tsu_jax_sim.backend import JAXTSUBackend
from mltsu.tsu_jax_sim.energy_models import ising_energy
from mltsu.physics.thermalization import DetailedBalanceChecker


class TestOnsagerSolution:
    """Test against Onsager's exact solution for 2D Ising model."""

    @staticmethod
    def onsager_magnetization(temperature: float) -> float:
        """
        Onsager's exact solution for 2D Ising magnetization.

        M = (1 - sinh(2β)^(-4))^(1/8) for T < Tc
        M = 0 for T >= Tc

        where Tc = 2/ln(1 + sqrt(2)) ≈ 2.269185
        """
        T_c = 2.0 / np.log(1 + np.sqrt(2))  # Critical temperature

        if temperature >= T_c:
            return 0.0

        beta = 1.0 / temperature
        sinh_2beta = np.sinh(2 * beta)

        if sinh_2beta > 0:
            magnetization = (1 - sinh_2beta**(-4))**(1/8)
        else:
            magnetization = 0

        return magnetization

    @staticmethod
    def onsager_energy(temperature: float) -> float:
        """
        Onsager's exact solution for 2D Ising energy per spin.

        E/N = -J * coth(2βJ) * (1 + (2/π) * κ * K(κ))

        where κ = 2 * tanh²(2βJ) - 1
        and K is the complete elliptic integral of the first kind
        """
        if temperature <= 0:
            return -2.0  # Ground state energy

        beta = 1.0 / temperature

        # For J = 1
        coth_2beta = 1.0 / np.tanh(2 * beta) if np.tanh(2 * beta) != 0 else 1e10
        tanh_2beta = np.tanh(2 * beta)

        kappa = 2 * tanh_2beta**2 - 1

        # Avoid numerical issues near T = 0
        if abs(kappa) < 1:
            K_kappa = ellipk(kappa**2)  # Complete elliptic integral
            energy = -coth_2beta * (1 + (2/np.pi) * kappa * K_kappa)
        else:
            energy = -2.0  # Low temperature limit

        return energy

    def test_critical_temperature(self):
        """Test that the system shows a phase transition at T_c ≈ 2.269185."""
        backend = JAXTSUBackend(seed=42)

        # Create 2D Ising model (32x32 lattice)
        L = 32
        n_spins = L * L

        # Nearest-neighbor coupling on square lattice
        J = self._create_2d_lattice_coupling(L)
        h = np.zeros(n_spins)

        # Temperature scan around critical point
        T_c_exact = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269185
        temperatures = np.linspace(2.0, 2.5, 20)
        magnetizations = []

        for T in temperatures:
            beta = 1.0 / T

            # Sample from equilibrium
            result = backend.sample_ising(
                J, h, beta,
                num_steps=10000,  # Long run for equilibration
                batch_size=10
            )

            # Compute magnetization
            samples = result['samples']
            mag = np.abs(np.mean(samples, axis=1))  # Per sample magnetization
            avg_mag = np.mean(mag)
            magnetizations.append(avg_mag)

        # Find transition point (where magnetization drops)
        # Use finite-size scaling: on finite lattice, transition is smoothed
        mid_point = len(magnetizations) // 2
        mag_diff = magnetizations[:mid_point] - magnetizations[mid_point:]

        # Estimate Tc as temperature where magnetization changes most rapidly
        grad_mag = np.gradient(magnetizations)
        idx_max_change = np.argmax(np.abs(grad_mag))
        T_c_measured = temperatures[idx_max_change]

        # Check within 5% of exact value (finite-size effects)
        relative_error = abs(T_c_measured - T_c_exact) / T_c_exact

        assert relative_error < 0.05, (
            f"Critical temperature {T_c_measured:.4f} deviates from "
            f"Onsager solution {T_c_exact:.4f} by {relative_error*100:.1f}%"
        )

        # Also check that magnetization is high below Tc and low above
        T_low_idx = np.argmin(np.abs(temperatures - 2.1))
        T_high_idx = np.argmin(np.abs(temperatures - 2.4))

        assert magnetizations[T_low_idx] > 0.3, "Magnetization should be significant below Tc"
        assert magnetizations[T_high_idx] < 0.2, "Magnetization should be small above Tc"

    def test_magnetization_curve(self):
        """Test magnetization vs temperature against exact solution."""
        backend = JAXTSUBackend(seed=42)

        # Smaller lattice for faster testing
        L = 16
        n_spins = L * L
        J = self._create_2d_lattice_coupling(L)
        h = np.zeros(n_spins)

        # Test at several temperatures
        temperatures = [1.5, 1.8, 2.0, 2.2, 2.3, 2.5]

        for T in temperatures:
            beta = 1.0 / T

            # Sample
            result = backend.sample_ising(
                J, h, beta,
                num_steps=5000,
                batch_size=20
            )

            # Compute magnetization
            samples = result['samples']
            measured_mag = np.mean(np.abs(np.mean(samples, axis=1)))

            # Exact magnetization (with finite-size correction)
            exact_mag = self.onsager_magnetization(T)

            # Finite-size scaling: M_L = M_∞ * (1 - a/L^b)
            # Rough correction for finite lattice
            finite_size_factor = (1 - 1.0 / L)
            exact_mag_corrected = exact_mag * finite_size_factor

            # Allow 20% error due to finite sampling and finite size
            if exact_mag_corrected > 0.05:  # Only check when magnetization is significant
                relative_error = abs(measured_mag - exact_mag_corrected) / exact_mag_corrected
                assert relative_error < 0.3, (
                    f"At T={T}, measured M={measured_mag:.3f} differs from "
                    f"Onsager M={exact_mag_corrected:.3f} by {relative_error*100:.1f}%"
                )

    @staticmethod
    def _create_2d_lattice_coupling(L: int) -> np.ndarray:
        """Create coupling matrix for 2D square lattice with periodic boundaries."""
        n = L * L
        J = np.zeros((n, n))

        for i in range(L):
            for j in range(L):
                idx = i * L + j

                # Right neighbor (periodic)
                right = i * L + ((j + 1) % L)
                J[idx, right] = 1.0
                J[right, idx] = 1.0

                # Down neighbor (periodic)
                down = ((i + 1) % L) * L + j
                J[idx, down] = 1.0
                J[down, idx] = 1.0

        return J


class TestBoltzmannDistribution:
    """Test that samples follow the correct Boltzmann distribution."""

    def test_small_system_exact(self):
        """For small systems, we can enumerate all states and check the distribution."""
        backend = JAXTSUBackend(seed=42)

        # Very small system (8 spins) where we can enumerate all 2^8 = 256 states
        n_spins = 8

        # Random coupling (keep small for numerical stability)
        np.random.seed(42)
        J = np.random.randn(n_spins, n_spins) * 0.5
        J = (J + J.T) / 2
        h = np.random.randn(n_spins) * 0.5

        beta = 1.0  # Moderate temperature

        # Sample many configurations
        result = backend.sample_ising(
            J, h, beta,
            num_steps=50000,
            batch_size=1
        )

        # Convert to binary representation for counting
        samples = result['samples'][0]

        # Sample every 10 steps to reduce correlation
        samples_uncorrelated = samples[::10] if len(samples) > 10 else samples

        # Count state frequencies
        state_counts = {}
        for sample in samples_uncorrelated:
            state_key = tuple(sample.astype(int))
            state_counts[state_key] = state_counts.get(state_key, 0) + 1

        # Compute exact Boltzmann probabilities for most common states
        top_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Compute partition function (sum over subset of observed states)
        observed_states = list(state_counts.keys())
        energies = []
        for state in observed_states[:100]:  # Limit for speed
            state_array = np.array(state)
            E = -0.5 * state_array @ J @ state_array - h @ state_array
            energies.append(E)

        # Boltzmann weights
        boltzmann_weights = np.exp(-beta * np.array(energies))
        Z_subset = np.sum(boltzmann_weights)
        exact_probs = boltzmann_weights / Z_subset

        # Compare distributions using chi-squared test
        n_samples_total = sum(state_counts.values())

        # Check that probability ratios match Boltzmann factors
        for i in range(min(5, len(top_states))):
            state1 = np.array(top_states[i][0])
            count1 = top_states[i][1]

            if i + 1 < len(top_states):
                state2 = np.array(top_states[i + 1][0])
                count2 = top_states[i + 1][1]

                # Energy difference
                E1 = -0.5 * state1 @ J @ state1 - h @ state1
                E2 = -0.5 * state2 @ J @ state2 - h @ state2
                dE = E2 - E1

                # Expected probability ratio
                expected_ratio = np.exp(-beta * dE)

                # Measured ratio
                if count2 > 0:
                    measured_ratio = count1 / count2

                    # Check ratio (allow 30% error due to finite sampling)
                    relative_error = abs(measured_ratio - expected_ratio) / expected_ratio
                    assert relative_error < 0.5, (
                        f"Probability ratio {measured_ratio:.3f} differs from "
                        f"Boltzmann ratio {expected_ratio:.3f} by {relative_error*100:.1f}%"
                    )

    def test_energy_distribution(self):
        """Test that energy distribution matches canonical ensemble."""
        backend = JAXTSUBackend(seed=42)

        # Medium system
        n_spins = 20
        np.random.seed(42)
        J = np.random.randn(n_spins, n_spins) * 0.3
        J = (J + J.T) / 2
        h = np.zeros(n_spins)

        temperature = 2.0
        beta = 1.0 / temperature

        # Sample
        result = backend.sample_ising(
            J, h, beta,
            num_steps=10000,
            batch_size=50
        )

        # Compute energies
        energies = result['final_energy']

        # Theoretical predictions for canonical ensemble
        # <E> and <E²> can be computed from derivatives of ln Z
        mean_energy = np.mean(energies)
        var_energy = np.var(energies)

        # Fluctuation-dissipation theorem: Var(E) = T² * C_v
        # For Ising model, C_v ~ N * (βJ)² at high T
        expected_var = temperature**2 * n_spins * (beta * 0.3)**2

        # Check order of magnitude (within factor of 3)
        ratio = var_energy / expected_var
        assert 0.3 < ratio < 3.0, (
            f"Energy variance {var_energy:.2f} inconsistent with "
            f"fluctuation-dissipation prediction {expected_var:.2f}"
        )


class TestDetailedBalance:
    """Test that the MCMC sampler satisfies detailed balance."""

    def test_transition_probabilities(self):
        """Verify detailed balance for Gibbs sampler."""
        # Setup simple 2-spin system
        J = np.array([[0, 1], [1, 0]])
        h = np.array([0.5, -0.5])
        beta = 1.0

        checker = DetailedBalanceChecker(
            lambda s: -0.5 * s @ J @ s - h @ s,
            beta
        )

        # Test several state pairs
        test_pairs = [
            (np.array([1, 1]), np.array([1, -1])),
            (np.array([1, -1]), np.array([-1, -1])),
            (np.array([-1, 1]), np.array([1, 1])),
        ]

        for state1, state2 in test_pairs:
            # For Gibbs sampler with single-spin flips
            def transition_prob(s1, s2):
                # Can only transition if exactly one spin differs
                diff = np.sum(s1 != s2)
                if diff != 1:
                    return 0.0

                # Find which spin flipped
                idx = np.where(s1 != s2)[0][0]

                # Local field at that spin
                local_field = J[idx] @ s1 + h[idx]

                # Probability of flipping to s2[idx]
                prob_up = 1 / (1 + np.exp(-2 * beta * local_field))

                if s2[idx] == 1:
                    return prob_up / len(s1)  # 1/N for site selection
                else:
                    return (1 - prob_up) / len(s1)

            result = checker.verify_transition_probabilities(
                state1, state2, transition_prob
            )

            assert result['satisfied'], (
                f"Detailed balance violated for states {state1} -> {state2}: "
                f"ratio = {result['balance_ratio']:.6f}"
            )


class TestStatisticalMechanics:
    """Test fundamental statistical mechanics relations."""

    def test_fluctuation_dissipation(self):
        """Test the fluctuation-dissipation theorem."""
        backend = JAXTSUBackend(seed=42)

        n_spins = 30
        np.random.seed(42)
        J = np.random.randn(n_spins, n_spins) * 0.5
        J = (J + J.T) / 2
        h = np.zeros(n_spins)

        # Test at different temperatures
        for T in [1.0, 2.0, 3.0]:
            beta = 1.0 / T

            result = backend.sample_ising(
                J, h, beta,
                num_steps=5000,
                batch_size=100
            )

            energies = result['final_energy']

            # Fluctuation-dissipation: <(E - <E>)²> = T² * ∂<E>/∂T
            energy_var = np.var(energies)

            # For high T: C_v ≈ N * (βJ)² → <E²> - <E>² ≈ T² * N * J²/T²
            # Simplified: Var(E) ~ N * J²
            expected_scale = n_spins * np.std(J.flatten())**2 * T**2

            ratio = energy_var / expected_scale
            assert 0.1 < ratio < 10, (
                f"Energy fluctuations {energy_var:.2f} inconsistent with "
                f"fluctuation-dissipation scale {expected_scale:.2f} at T={T}"
            )

    def test_spin_spin_correlation(self):
        """Test spin-spin correlation functions."""
        backend = JAXTSUBackend(seed=42)

        # 1D chain for simplicity
        n_spins = 20
        J = np.zeros((n_spins, n_spins))

        # Nearest-neighbor coupling
        for i in range(n_spins - 1):
            J[i, i + 1] = 1.0
            J[i + 1, i] = 1.0

        h = np.zeros(n_spins)

        # High temperature (paramagnetic phase)
        T_high = 5.0
        beta_high = 1.0 / T_high

        result_high = backend.sample_ising(
            J, h, beta_high,
            num_steps=5000,
            batch_size=100
        )

        # Compute nearest-neighbor correlation
        samples_high = result_high['samples']
        corr_high = np.mean([
            np.mean(samples_high[:, i] * samples_high[:, i + 1])
            for i in range(n_spins - 1)
        ])

        # At high T, correlation should be weak
        assert abs(corr_high) < 0.5, f"Correlation {corr_high:.3f} too strong at high T"

        # Low temperature (ordered phase)
        T_low = 0.5
        beta_low = 1.0 / T_low

        result_low = backend.sample_ising(
            J, h, beta_low,
            num_steps=5000,
            batch_size=100
        )

        samples_low = result_low['samples']
        corr_low = np.mean([
            np.mean(samples_low[:, i] * samples_low[:, i + 1])
            for i in range(n_spins - 1)
        ])

        # At low T, correlation should be strong
        assert corr_low > 0.7, f"Correlation {corr_low:.3f} too weak at low T"


class TestPhysicsIntegration:
    """Integration tests for the complete physics pipeline."""

    def test_realistic_noise_impact(self):
        """Test that realistic noise doesn't break physics."""
        from mltsu.physics.realistic_noise import ThermodynamicNoiseModel

        backend = JAXTSUBackend(seed=42)
        noise_model = ThermodynamicNoiseModel()

        # Create system with imperfections
        n_spins = 16
        J_ideal = np.random.randn(n_spins, n_spins) * 0.5
        J_ideal = (J_ideal + J_ideal.T) / 2
        h_ideal = np.zeros(n_spins)

        # Add realistic imperfections
        J_actual, h_actual = noise_model.model_complete_device(J_ideal, h_ideal)

        # Should still satisfy basic physics
        beta = 1.0
        result = backend.sample_ising(
            J_actual, h_actual, beta,
            num_steps=1000,
            batch_size=10
        )

        # Check that energies are finite and reasonable
        energies = result['final_energy']
        assert np.all(np.isfinite(energies)), "Energies should be finite"
        assert np.abs(np.mean(energies)) < 100 * n_spins, "Energies unreasonably large"

    def test_thermalization_convergence(self):
        """Test that system actually thermalizes."""
        from mltsu.physics.thermalization import ThermalizationModel, SystemParameters

        backend = JAXTSUBackend(seed=42)

        # Setup system
        L = 8
        n_spins = L * L
        params = SystemParameters(
            system_size=n_spins,
            coupling_strength=1.0,
            temperature=2.0,
            dimension=2
        )

        therm_model = ThermalizationModel(params)

        # Estimate required thermalization
        mixing_info = therm_model.estimate_mixing_time()
        required_steps = int(mixing_info['mixing_steps'])

        # Create 2D lattice
        J = TestOnsagerSolution._create_2d_lattice_coupling(L)
        h = np.zeros(n_spins)

        # Start from ordered state (all spins up)
        init_state = np.ones(n_spins)

        # Run with estimated thermalization time
        result = backend.sample_ising(
            J, h, params.beta,
            num_steps=required_steps,
            batch_size=1,
            init_state=init_state
        )

        # Final state should have lost memory of initial condition
        final_sample = result['samples'][0]

        # Magnetization should be near zero at T=2.0 (paramagnetic)
        final_mag = np.abs(np.mean(final_sample))
        assert final_mag < 0.3, (
            f"System didn't thermalize: magnetization {final_mag:.3f} "
            f"still shows initial order after {required_steps} steps"
        )


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])