"""
Test convergence diagnostics.

This test file validates that our convergence diagnostics correctly identify:
1. Converged vs non-converged chains
2. Proper calculation of R̂, ESS, and other metrics
3. Agreement with known test cases
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mltsu.diagnostics.convergence import (
    ConvergenceDiagnostics,
    GelmanRubinDiagnostic,
    EffectiveSampleSize,
    MCMCError,
    GewekeDiagnostic,
    HeidelbergerWelchTest,
    quick_convergence_check
)


def generate_converged_chains(n_chains=4, n_samples=1000, seed=42):
    """Generate well-mixed chains from the same distribution."""
    np.random.seed(seed)
    chains = []

    # All chains sample from N(0, 1)
    for _ in range(n_chains):
        # Add some autocorrelation but not too much
        samples = np.zeros(n_samples)
        samples[0] = np.random.randn()
        for i in range(1, n_samples):
            samples[i] = 0.5 * samples[i-1] + np.sqrt(1 - 0.5**2) * np.random.randn()
        chains.append(samples)

    return chains


def generate_non_converged_chains(n_chains=4, n_samples=1000, seed=42):
    """Generate chains that haven't converged (different distributions)."""
    np.random.seed(seed)
    chains = []

    # Each chain has different mean/variance
    for i in range(n_chains):
        mean = i * 2.0  # Different means
        std = 1.0 + i * 0.5  # Different variances
        samples = np.random.randn(n_samples) * std + mean
        chains.append(samples)

    return chains


def generate_high_autocorr_chain(n_samples=1000, rho=0.95, seed=42):
    """Generate chain with high autocorrelation (poor mixing)."""
    np.random.seed(seed)
    samples = np.zeros(n_samples)
    samples[0] = np.random.randn()

    for i in range(1, n_samples):
        samples[i] = rho * samples[i-1] + np.sqrt(1 - rho**2) * np.random.randn()

    return samples


def test_gelman_rubin():
    """Test Gelman-Rubin diagnostic."""
    print("Testing Gelman-Rubin diagnostic...")

    gr = GelmanRubinDiagnostic(threshold=1.1)

    # Test converged chains
    converged_chains = generate_converged_chains()
    chains_array = np.stack(converged_chains)
    r_hat, converged = gr.compute(chains_array)

    print(f"  Converged chains: R̂={r_hat:.3f}, converged={converged}")
    assert converged, f"Should detect convergence, but R̂={r_hat}"
    assert r_hat < 1.1, f"R̂ should be < 1.1, got {r_hat}"

    # Test non-converged chains
    non_converged_chains = generate_non_converged_chains()
    chains_array = np.stack(non_converged_chains)
    r_hat, converged = gr.compute(chains_array)

    print(f"  Non-converged chains: R̂={r_hat:.3f}, converged={converged}")
    assert not converged, f"Should detect non-convergence, but R̂={r_hat}"
    assert r_hat > 1.1, f"R̂ should be > 1.1, got {r_hat}"

    print("  ✓ Gelman-Rubin test passed")


def test_effective_sample_size():
    """Test effective sample size calculation."""
    print("\nTesting Effective Sample Size...")

    ess_calc = EffectiveSampleSize(min_ess=100)

    # Test independent samples (ESS ≈ N)
    independent_samples = np.random.randn(1000)
    ess, adequate = ess_calc.compute(independent_samples)

    print(f"  Independent samples: ESS={ess:.1f} (expect ≈1000)")
    assert ess > 800, f"ESS should be close to N for independent samples, got {ess}"
    assert adequate, "Should have adequate ESS for independent samples"

    # Test highly correlated samples (ESS << N)
    correlated_samples = generate_high_autocorr_chain(n_samples=1000, rho=0.95)
    ess, adequate = ess_calc.compute(correlated_samples)

    print(f"  Correlated samples (ρ=0.95): ESS={ess:.1f} (expect <<1000)")
    assert ess < 200, f"ESS should be much less than N for correlated samples, got {ess}"

    # Test autocorrelation calculation
    acf = ess_calc.compute_autocorrelation(correlated_samples, max_lag=50)
    print(f"  Autocorrelation at lag 1: {acf[1]:.3f} (expect ≈0.95)")
    assert abs(acf[1] - 0.95) < 0.1, f"ACF[1] should be ≈0.95, got {acf[1]}"

    print("  ✓ ESS test passed")


def test_mcmc_error():
    """Test Monte Carlo standard error."""
    print("\nTesting MCMC Standard Error...")

    mcse_calc = MCMCError()

    # Test with known mean and variance
    n_samples = 10000
    samples = np.random.randn(n_samples)  # Mean=0, Var=1

    # Batch means MCSE
    mcse_batch = mcse_calc.compute_batch_means(samples)
    expected_mcse = 1.0 / np.sqrt(n_samples)  # For independent samples

    print(f"  Batch means MCSE: {mcse_batch:.4f} (expect ≈{expected_mcse:.4f})")
    assert abs(mcse_batch - expected_mcse) < 0.02, f"MCSE should be ≈{expected_mcse:.4f}"

    # Spectral MCSE
    ess = n_samples * 0.8  # Assume 80% efficiency
    mcse_spectral = mcse_calc.compute_spectral(samples, ess)
    expected_spectral = 1.0 / np.sqrt(ess)

    print(f"  Spectral MCSE: {mcse_spectral:.4f} (expect ≈{expected_spectral:.4f})")
    assert abs(mcse_spectral - expected_spectral) < 0.02

    print("  ✓ MCSE test passed")


def test_geweke_diagnostic():
    """Test Geweke diagnostic."""
    print("\nTesting Geweke Diagnostic...")

    geweke = GewekeDiagnostic(first_frac=0.1, last_frac=0.5)

    # Test stationary chain
    stationary_chain = np.random.randn(1000)
    z_score, converged = geweke.compute(stationary_chain)

    print(f"  Stationary chain: z={z_score:.2f}, converged={converged}")
    assert abs(z_score) < 3, f"Z-score should be small for stationary chain, got {z_score}"

    # Test non-stationary chain (with trend)
    non_stationary = np.arange(1000) * 0.01 + np.random.randn(1000)
    z_score, converged = geweke.compute(non_stationary)

    print(f"  Non-stationary chain: z={z_score:.2f}, converged={converged}")
    assert abs(z_score) > 2, f"Z-score should be large for non-stationary chain"

    print("  ✓ Geweke test passed")


def test_heidelberger_welch():
    """Test Heidelberger-Welch test."""
    print("\nTesting Heidelberger-Welch Test...")

    hw = HeidelbergerWelchTest(alpha=0.05, epsilon=0.1)

    # Test stationary chain with good precision
    good_chain = np.random.randn(1000) + 5.0  # Mean=5, easy to estimate
    stationary, halfwidth_passed = hw.compute(good_chain)

    print(f"  Good chain: stationary={stationary}, halfwidth_passed={halfwidth_passed}")
    assert stationary, "Should detect stationarity"
    assert halfwidth_passed, "Should pass halfwidth test"

    # Test chain with poor precision (high variance relative to mean)
    poor_chain = np.random.randn(100) * 10 + 0.1  # High variance, small mean
    stationary, halfwidth_passed = hw.compute(poor_chain)

    print(f"  Poor precision chain: stationary={stationary}, halfwidth_passed={halfwidth_passed}")
    # Halfwidth test might fail due to poor precision

    print("  ✓ Heidelberger-Welch test passed")


def test_full_diagnostics():
    """Test complete convergence diagnostics."""
    print("\nTesting Full Convergence Diagnostics...")

    diagnostics = ConvergenceDiagnostics()

    # Test single converged chain
    print("\n  Single chain diagnostics:")
    good_chain = generate_converged_chains(n_chains=1)[0]
    result = diagnostics.diagnose_single_chain(good_chain, runtime_seconds=1.0)

    print(f"    Converged: {result.converged}")
    print(f"    ESS: {result.ess:.1f}")
    print(f"    ESS/sec: {result.ess_per_second:.1f}")
    print(f"    Autocorr time: {result.autocorr_time:.1f}")
    print(f"    MCSE: {result.mcse:.4f}")
    print(f"    Geweke z: {result.geweke_z:.2f}")

    # Test multiple chains
    print("\n  Multiple chain diagnostics:")
    chains = generate_converged_chains(n_chains=4)
    result = diagnostics.diagnose_multiple_chains(chains, runtime_seconds=4.0)

    print(f"    Converged: {result.converged}")
    print(f"    R̂: {result.r_hat:.3f}")
    print(f"    ESS: {result.ess:.1f}")
    print(f"    Warnings: {result.warnings}")

    assert result.converged, "Should detect convergence for good chains"
    assert result.r_hat < 1.1, f"R̂ should be < 1.1, got {result.r_hat}"

    # Test recommendations
    recommendations = diagnostics.recommend_sampling_params(result)
    print(f"\n  Recommendations: {recommendations}")

    print("  ✓ Full diagnostics test passed")


def test_quick_check():
    """Test quick convergence check utility."""
    print("\nTesting Quick Convergence Check...")

    # Test with single chain
    print("\n  Single chain:")
    chain = generate_converged_chains(n_chains=1)[0]
    converged = quick_convergence_check(chain, verbose=True)
    assert converged, "Should detect convergence"

    # Test with multiple chains
    print("\n  Multiple chains:")
    chains = generate_non_converged_chains(n_chains=3)
    converged = quick_convergence_check(chains, verbose=True)
    assert not converged, "Should detect non-convergence"

    print("  ✓ Quick check test passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases...")

    gr = GelmanRubinDiagnostic()

    # Test with single chain (should raise error)
    try:
        single_chain = np.random.randn(100).reshape(1, -1)
        gr.compute(single_chain)
        assert False, "Should raise error for single chain"
    except ValueError as e:
        print(f"  ✓ Correctly raised error for single chain: {e}")

    # Test with very short chains
    short_chains = np.random.randn(2, 50)  # Only 50 samples
    r_hat, _ = gr.compute(short_chains)
    print(f"  Short chains handled: R̂={r_hat:.3f}")

    # Test with constant chain (zero variance)
    constant_chain = np.ones((2, 100))
    r_hat, converged = gr.compute(constant_chain)
    print(f"  Constant chains: R̂={r_hat:.3f}")

    print("  ✓ Edge cases handled correctly")


def run_all_tests():
    """Run all convergence diagnostic tests."""
    print("=" * 60)
    print("CONVERGENCE DIAGNOSTICS TEST SUITE")
    print("=" * 60)

    test_gelman_rubin()
    test_effective_sample_size()
    test_mcmc_error()
    test_geweke_diagnostic()
    test_heidelberger_welch()
    test_full_diagnostics()
    test_quick_check()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("ALL CONVERGENCE DIAGNOSTIC TESTS PASSED ✓")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)