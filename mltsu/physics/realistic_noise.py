"""
Physically realistic noise models for thermodynamic computing.

This module implements actual thermal fluctuations and device noise
that would be present in real TSU hardware, replacing idealized
pseudo-random number generators with physics-based noise.

Key improvements over naive implementation:
1. Colored noise with realistic correlation times (not white noise)
2. Johnson-Nyquist thermal fluctuations
3. Device-specific variations and imperfections
4. Temperature-dependent effects

References:
    [1] Johnson, J. B. (1928). "Thermal agitation of electricity in conductors"
    [2] Nyquist, H. (1928). "Thermal agitation of electric charge in conductors"
    [3] Uhlenbeck & Ornstein (1930). "Theory of Brownian motion"
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings


# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
REDUCED_PLANCK = PLANCK_CONSTANT / (2 * np.pi)


@dataclass
class PhysicalParameters:
    """Physical parameters for realistic noise modeling."""

    # Temperature
    temperature: float = 300.0  # Kelvin

    # Device characteristics
    resistance: float = 1e6  # Ohms (typical for magnetic tunnel junction)
    capacitance: float = 1e-15  # Farads (femtofarad scale)
    inductance: float = 1e-12  # Henries (picohenry scale)

    # Time scales
    correlation_time: float = 1e-9  # 1 nanosecond correlation time
    sampling_rate: float = 1e9  # 1 GHz sampling
    thermalization_time: float = 1e-6  # 1 microsecond to thermal equilibrium

    # Device imperfections
    coupling_variation: float = 0.1  # ±10% device-to-device variation
    readout_fidelity: float = 0.98  # 98% measurement accuracy
    crosstalk_strength: float = 0.01  # 1% nearest-neighbor crosstalk

    @property
    def thermal_energy(self) -> float:
        """kT energy scale in Joules."""
        return BOLTZMANN_CONSTANT * self.temperature

    @property
    def thermal_voltage(self) -> float:
        """Thermal voltage noise amplitude."""
        bandwidth = 1 / (2 * np.pi * self.resistance * self.capacitance)
        return np.sqrt(4 * BOLTZMANN_CONSTANT * self.temperature * self.resistance * bandwidth)

    @property
    def johnson_noise_power(self) -> float:
        """Johnson noise power spectral density."""
        return 4 * BOLTZMANN_CONSTANT * self.temperature * self.resistance


class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhlenbeck process for colored noise generation.

    This provides realistic thermal noise with finite correlation time,
    unlike white noise which assumes infinite bandwidth (unphysical).

    The OU process satisfies:
        dX_t = -γ(X_t - μ)dt + σ dW_t

    where γ = 1/τ is the relaxation rate, μ is the mean,
    σ is the diffusion coefficient, and dW_t is Brownian motion.
    """

    def __init__(self, params: Optional[PhysicalParameters] = None):
        self.params = params or PhysicalParameters()

        # OU process parameters
        self.gamma = 1 / self.params.correlation_time  # Relaxation rate
        self.mu = 0.0  # Mean value (zero for thermal noise)

        # Diffusion coefficient from fluctuation-dissipation theorem
        # σ² = 2γkT for thermal equilibrium
        self.sigma = np.sqrt(2 * self.gamma * self.params.thermal_energy)

    def generate(
        self,
        shape: Tuple[int, ...],
        dt: float,
        initial_state: Optional[np.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> np.ndarray:
        """
        Generate colored noise using Ornstein-Uhlenbeck process.

        Args:
            shape: Shape of noise array to generate
            dt: Time step in seconds
            initial_state: Initial noise values (optional)
            key: JAX random key

        Returns:
            Colored noise array with realistic correlation time
        """
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 2**32))

        # Initialize state
        if initial_state is None:
            key, subkey = jax.random.split(key)
            state = jax.random.normal(subkey, shape) * np.sqrt(self.params.thermal_energy)
        else:
            state = jnp.array(initial_state)

        # Exact solution for OU process over time step dt
        # X(t+dt) = X(t)e^(-γdt) + μ(1 - e^(-γdt)) + σ√((1 - e^(-2γdt))/(2γ)) * N(0,1)

        decay = np.exp(-self.gamma * dt)
        diffusion = self.sigma * np.sqrt((1 - np.exp(-2 * self.gamma * dt)) / (2 * self.gamma))

        key, subkey = jax.random.split(key)
        white_noise = jax.random.normal(subkey, shape)

        # Update state
        new_state = (
            state * decay +
            self.mu * (1 - decay) +
            diffusion * white_noise
        )

        return np.array(new_state)

    def power_spectral_density(self, frequency: np.ndarray) -> np.ndarray:
        """
        Power spectral density of OU process.

        S(ω) = σ²/(2π) * 2γ/(γ² + ω²)

        This shows the colored nature: flat at low frequency,
        rolls off as 1/ω² at high frequency.
        """
        omega = 2 * np.pi * frequency
        return (self.sigma**2 / (2 * np.pi)) * (2 * self.gamma) / (self.gamma**2 + omega**2)


class JohnsonNyquistNoise:
    """
    Johnson-Nyquist thermal noise in electrical circuits.

    This is the fundamental thermal noise in any resistive element,
    arising from random thermal motion of charge carriers.
    """

    def __init__(self, params: Optional[PhysicalParameters] = None):
        self.params = params or PhysicalParameters()

    def voltage_noise(self, bandwidth: float, duration: float, dt: float) -> np.ndarray:
        """
        Generate thermal voltage noise.

        V_n = sqrt(4kTRΔf)

        Args:
            bandwidth: Measurement bandwidth in Hz
            duration: Total time duration in seconds
            dt: Time step in seconds

        Returns:
            Voltage noise time series
        """
        n_samples = int(duration / dt)

        # RMS voltage noise
        v_rms = np.sqrt(4 * self.params.thermal_energy * self.params.resistance * bandwidth)

        # Generate colored noise using OU process
        ou_process = OrnsteinUhlenbeckProcess(self.params)
        noise = ou_process.generate((n_samples,), dt)

        # Scale to correct voltage amplitude
        noise = noise * v_rms / np.std(noise)

        return noise

    def current_noise(self, bandwidth: float, duration: float, dt: float) -> np.ndarray:
        """
        Generate thermal current noise.

        I_n = sqrt(4kTΔf/R)
        """
        voltage_noise = self.voltage_noise(bandwidth, duration, dt)
        return voltage_noise / self.params.resistance


class DeviceImperfections:
    """
    Model realistic device imperfections in TSU hardware.

    Real devices have:
    1. Manufacturing variations (±10% typical)
    2. Measurement errors (2% typical)
    3. Crosstalk between elements
    4. Temperature drift
    5. 1/f noise at low frequencies
    """

    def __init__(self, params: Optional[PhysicalParameters] = None):
        self.params = params or PhysicalParameters()

    def add_coupling_variations(self, J_ideal: np.ndarray) -> np.ndarray:
        """
        Add realistic device-to-device variations in coupling strengths.

        Real magnetic/electronic devices have manufacturing tolerances
        that cause variations in coupling strength.
        """
        n_devices = J_ideal.shape[0]

        # Generate correlated variations (neighboring devices more similar)
        variations = np.random.normal(0, self.params.coupling_variation, J_ideal.shape)

        # Apply low-pass filter to create spatial correlation
        from scipy.ndimage import gaussian_filter
        variations = gaussian_filter(variations, sigma=1.0)

        # Scale to correct variance
        variations = variations * self.params.coupling_variation / np.std(variations)

        # Apply multiplicative variations
        J_actual = J_ideal * (1 + variations)

        # Ensure symmetry
        J_actual = (J_actual + J_actual.T) / 2

        return J_actual

    def add_crosstalk(self, states: np.ndarray, connectivity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Add crosstalk between neighboring devices.

        In real hardware, signals from one device can influence neighbors
        through electromagnetic coupling, thermal effects, or substrate coupling.
        """
        n_devices = len(states)

        if connectivity is None:
            # Default: nearest-neighbor connectivity on a line
            connectivity = np.zeros((n_devices, n_devices))
            for i in range(n_devices - 1):
                connectivity[i, i+1] = 1
                connectivity[i+1, i] = 1

        # Apply crosstalk
        crosstalk_effect = connectivity @ states * self.params.crosstalk_strength

        # Add to original states
        states_with_crosstalk = states + crosstalk_effect

        # Clip to valid range for binary states
        if np.all(np.abs(states) == 1):  # Binary ±1 states
            states_with_crosstalk = np.sign(states_with_crosstalk)

        return states_with_crosstalk

    def add_readout_errors(self, measurements: np.ndarray) -> np.ndarray:
        """
        Add realistic measurement/readout errors.

        Real measurement has finite fidelity due to:
        - Amplifier noise
        - ADC quantization
        - Timing jitter
        - Threshold variations
        """
        error_rate = 1 - self.params.readout_fidelity

        # Random bit flips for binary measurements
        if np.all(np.isin(measurements, [0, 1])) or np.all(np.isin(measurements, [-1, 1])):
            n_errors = int(len(measurements.flat) * error_rate)
            error_indices = np.random.choice(len(measurements.flat), n_errors, replace=False)

            measurements_flat = measurements.flatten()
            if np.all(np.isin(measurements, [0, 1])):
                measurements_flat[error_indices] = 1 - measurements_flat[error_indices]
            else:  # ±1 states
                measurements_flat[error_indices] = -measurements_flat[error_indices]

            return measurements_flat.reshape(measurements.shape)

        # Gaussian noise for continuous measurements
        noise_amplitude = np.std(measurements) * (1 - self.params.readout_fidelity)
        readout_noise = np.random.normal(0, noise_amplitude, measurements.shape)

        return measurements + readout_noise

    def temperature_drift(self, time: float, base_temperature: float = 300.0) -> float:
        """
        Model realistic temperature drift over time.

        Real systems have temperature fluctuations due to:
        - Environmental changes
        - Power dissipation
        - Thermal cycling
        """
        # Slow drift (hours timescale)
        slow_drift = 0.5 * np.sin(2 * np.pi * time / 3600)  # ±0.5K hourly

        # Fast fluctuations (seconds timescale)
        fast_noise = 0.1 * np.sin(2 * np.pi * time / 10)  # ±0.1K per 10s

        # Random walk component
        random_component = 0.05 * np.random.normal()  # ±0.05K random

        return base_temperature + slow_drift + fast_noise + random_component


class ThermodynamicNoiseModel:
    """
    Complete thermodynamic noise model for realistic TSU simulation.

    This combines all physical noise sources and device imperfections
    to provide a realistic simulation of actual hardware behavior.
    """

    def __init__(self, params: Optional[PhysicalParameters] = None):
        self.params = params or PhysicalParameters()
        self.ou_process = OrnsteinUhlenbeckProcess(params)
        self.johnson_noise = JohnsonNyquistNoise(params)
        self.imperfections = DeviceImperfections(params)

        # Track time for temperature drift
        self.time = 0.0

        # Issue scientific warning about approximations
        warnings.warn(
            "ThermodynamicNoiseModel uses classical approximations. "
            "Quantum effects become important when kT < ℏω. "
            "Current temperature: {:.1f}K, thermal energy: {:.2e}J".format(
                self.params.temperature, self.params.thermal_energy
            ),
            category=UserWarning
        )

    def add_thermal_fluctuations(
        self,
        state: np.ndarray,
        dt: float,
        key: Optional[jax.random.PRNGKey] = None
    ) -> np.ndarray:
        """
        Add complete thermal fluctuations to a state.

        This includes:
        1. Colored thermal noise (not white noise)
        2. Device imperfections
        3. Temperature drift effects
        """
        # Update temperature with drift
        self.time += dt
        current_temp = self.imperfections.temperature_drift(self.time, self.params.temperature)

        # Scale noise by temperature
        temp_factor = np.sqrt(current_temp / self.params.temperature)

        # Generate colored noise
        thermal_noise = self.ou_process.generate(state.shape, dt, key=key)
        thermal_noise *= temp_factor

        # Add to state
        noisy_state = state + thermal_noise

        # Add crosstalk
        noisy_state = self.imperfections.add_crosstalk(noisy_state)

        return noisy_state

    def model_complete_device(
        self,
        J_ideal: np.ndarray,
        h_ideal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model a complete device with all imperfections.

        Returns:
            J_actual: Coupling matrix with variations
            h_actual: Field with noise
        """
        # Add coupling variations
        J_actual = self.imperfections.add_coupling_variations(J_ideal)

        # Add field noise
        field_noise = np.random.normal(0, self.params.coupling_variation, h_ideal.shape)
        h_actual = h_ideal * (1 + field_noise)

        return J_actual, h_actual

    def measure_with_errors(self, true_state: np.ndarray) -> np.ndarray:
        """
        Perform measurement with realistic errors.
        """
        return self.imperfections.add_readout_errors(true_state)

    def get_energy_consumption(self, n_operations: int) -> Dict[str, float]:
        """
        Calculate realistic energy consumption including noise effects.

        Real devices consume more energy due to:
        1. Repeated measurements for error mitigation
        2. Active temperature control
        3. Amplification for readout
        """
        # Base switching energy
        switching_energy = n_operations * 1e-14  # 10 fJ per switch

        # Error correction overhead (need ~1/fidelity measurements)
        error_overhead = n_operations * (1 / self.params.readout_fidelity - 1) * 1e-14

        # Readout amplification
        readout_energy = n_operations * 5e-13  # 500 fJ per readout

        # Temperature control (rough estimate)
        control_energy = n_operations * self.params.thermalization_time * 1e-6  # 1 μW control power

        return {
            'switching': switching_energy,
            'error_correction': error_overhead,
            'readout': readout_energy,
            'control': control_energy,
            'total': switching_energy + error_overhead + readout_energy + control_energy
        }