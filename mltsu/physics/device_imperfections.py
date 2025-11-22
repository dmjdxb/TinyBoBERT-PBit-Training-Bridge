"""
Realistic device imperfections and manufacturing variations.

Real TSU/P-bit hardware has numerous imperfections that affect performance:
1. Manufacturing tolerances (±10% typical)
2. Temperature drift and thermal gradients
3. Crosstalk between neighboring elements
4. Aging and wear-out effects
5. Power supply noise

These effects are critical for accurate energy and performance estimates.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import warnings


@dataclass
class DeviceSpecifications:
    """Specifications for a realistic TSU device."""

    # Manufacturing specs
    nominal_resistance: float = 1e6  # Ohms
    resistance_tolerance: float = 0.1  # ±10%
    coupling_tolerance: float = 0.1  # ±10%

    # Thermal specs
    thermal_resistance: float = 100  # K/W
    heat_capacity: float = 1e-12  # J/K
    max_temperature: float = 400  # K
    min_temperature: float = 200  # K

    # Electrical specs
    supply_voltage: float = 1.0  # V
    voltage_noise: float = 0.01  # 1% noise
    current_noise: float = 0.01  # 1% noise

    # Reliability
    mean_time_to_failure: float = 1e9  # seconds (~30 years)
    activation_energy: float = 1.0  # eV for Arrhenius model

    # Layout
    element_spacing: float = 100e-9  # 100 nm pitch
    chip_size: float = 10e-3  # 10 mm


class ManufacturingVariations:
    """
    Model device-to-device variations from manufacturing.

    In real fabrication, no two devices are identical due to:
    - Lithography variations
    - Material property variations
    - Process temperature fluctuations
    - Chemical concentration gradients
    """

    def __init__(self, specs: Optional[DeviceSpecifications] = None):
        self.specs = specs or DeviceSpecifications()

    def generate_coupling_matrix(
        self,
        nominal_J: np.ndarray,
        correlation_length: float = 5.0
    ) -> np.ndarray:
        """
        Generate realistic coupling matrix with spatial correlations.

        Nearby devices tend to have similar variations due to
        gradients in manufacturing processes.

        Args:
            nominal_J: Ideal coupling matrix
            correlation_length: Spatial correlation in units of element spacing

        Returns:
            J with realistic variations
        """
        n = nominal_J.shape[0]

        # Generate spatially correlated variations
        # Use Gaussian random field with correlation
        variations = self._generate_correlated_noise(n, correlation_length)

        # Scale to correct variance
        variations = variations * self.specs.coupling_tolerance / np.std(variations)

        # Apply multiplicative variations
        J_actual = nominal_J * (1 + variations)

        # Additional asymmetry from fabrication
        # Real devices don't have perfect J_ij = J_ji
        asymmetry = np.random.normal(0, self.specs.coupling_tolerance / 10, nominal_J.shape)
        J_actual = J_actual + asymmetry

        # Make nominally symmetric (but with small asymmetry)
        J_actual = (J_actual + J_actual.T) / 2

        # Some couplings may fail completely (manufacturing defects)
        defect_rate = 0.001  # 0.1% defect rate
        defects = np.random.random(nominal_J.shape) < defect_rate
        J_actual[defects] = 0

        return J_actual

    def _generate_correlated_noise(
        self,
        n: int,
        correlation_length: float
    ) -> np.ndarray:
        """Generate spatially correlated noise using convolution."""

        # White noise
        white = np.random.normal(0, 1, (n, n))

        # Correlation kernel (Gaussian)
        x = np.arange(n) - n // 2
        y = np.arange(n) - n // 2
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2 * correlation_length**2))
        kernel = kernel / kernel.sum()

        # Convolve to create correlations
        from scipy.signal import convolve2d
        correlated = convolve2d(white, kernel, mode='same', boundary='wrap')

        return correlated

    def device_mismatch(self, n_devices: int) -> Dict[str, np.ndarray]:
        """
        Generate device-to-device parameter mismatches.

        Returns dictionary with variations for each parameter.
        """
        # Resistance variations
        resistance = self.specs.nominal_resistance * (
            1 + np.random.normal(0, self.specs.resistance_tolerance, n_devices)
        )

        # Threshold variations (affects switching behavior)
        threshold = np.random.normal(0, 0.05, n_devices)  # 5% threshold variation

        # Gain variations (affects output amplitude)
        gain = 1 + np.random.normal(0, 0.02, n_devices)  # 2% gain variation

        # Offset variations (DC offset in output)
        offset = np.random.normal(0, 0.01, n_devices)  # 1% of full scale

        return {
            'resistance': resistance,
            'threshold': threshold,
            'gain': gain,
            'offset': offset
        }


class CrosstalkModel:
    """
    Model electrical and magnetic crosstalk between TSU elements.

    Crosstalk mechanisms:
    1. Capacitive coupling (electric field)
    2. Inductive coupling (magnetic field)
    3. Substrate coupling (common ground)
    4. Thermal coupling (heat diffusion)
    """

    def __init__(self, specs: Optional[DeviceSpecifications] = None):
        self.specs = specs or DeviceSpecifications()

    def compute_crosstalk_matrix(
        self,
        n_devices: int,
        layout: str = 'grid'
    ) -> np.ndarray:
        """
        Compute crosstalk coupling between devices.

        Args:
            n_devices: Number of devices
            layout: Physical layout ('grid', 'line', 'random')

        Returns:
            Crosstalk matrix C where C[i,j] is coupling from j to i
        """
        if layout == 'grid':
            # 2D grid layout
            grid_size = int(np.sqrt(n_devices))
            if grid_size**2 != n_devices:
                grid_size += 1

            crosstalk = np.zeros((n_devices, n_devices))

            for i in range(n_devices):
                row_i, col_i = i // grid_size, i % grid_size

                for j in range(n_devices):
                    if i == j:
                        continue

                    row_j, col_j = j // grid_size, j % grid_size

                    # Distance in grid
                    distance = np.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)

                    # Crosstalk falls off with distance
                    # Nearest neighbors: ~1% crosstalk
                    # Next-nearest: ~0.1% crosstalk
                    if distance <= 1.01:  # Nearest neighbors
                        crosstalk[i, j] = 0.01
                    elif distance <= 1.5:  # Diagonal neighbors
                        crosstalk[i, j] = 0.005
                    elif distance <= 2.01:  # Next-nearest
                        crosstalk[i, j] = 0.001
                    else:
                        # Long-range crosstalk (1/r² falloff)
                        crosstalk[i, j] = 0.001 / distance**2

        elif layout == 'line':
            # 1D chain layout
            crosstalk = np.zeros((n_devices, n_devices))

            for i in range(n_devices):
                for j in range(n_devices):
                    if i != j:
                        distance = abs(i - j)
                        # Exponential decay along chain
                        crosstalk[i, j] = 0.01 * np.exp(-distance / 2)

        else:  # random layout
            # Random placement - use random crosstalk
            crosstalk = np.random.exponential(0.001, (n_devices, n_devices))
            np.fill_diagonal(crosstalk, 0)
            crosstalk = (crosstalk + crosstalk.T) / 2

        return crosstalk

    def apply_crosstalk(
        self,
        signals: np.ndarray,
        crosstalk_matrix: np.ndarray,
        nonlinear: bool = False
    ) -> np.ndarray:
        """
        Apply crosstalk to signals.

        Args:
            signals: Input signals for each device
            crosstalk_matrix: Coupling between devices
            nonlinear: Include nonlinear crosstalk effects

        Returns:
            Signals with crosstalk applied
        """
        # Linear crosstalk
        crosstalk_contribution = crosstalk_matrix @ signals

        if nonlinear:
            # Nonlinear crosstalk (e.g., from saturation effects)
            # Crosstalk is stronger when signals are large
            nonlinear_factor = 1 + 0.1 * np.abs(signals)
            crosstalk_contribution *= nonlinear_factor

        # Add crosstalk to original signals
        output = signals + crosstalk_contribution

        # Saturation effects (devices have limited dynamic range)
        max_output = 1.0
        output = np.clip(output, -max_output, max_output)

        return output


class ReadoutErrors:
    """
    Model measurement/readout errors in TSU devices.

    Error sources:
    1. Amplifier noise
    2. ADC quantization
    3. Timing jitter
    4. Threshold variations
    5. Metastability
    """

    def __init__(
        self,
        specs: Optional[DeviceSpecifications] = None,
        adc_bits: int = 12
    ):
        self.specs = specs or DeviceSpecifications()
        self.adc_bits = adc_bits
        self.quantization_levels = 2**adc_bits

    def add_amplifier_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add realistic amplifier noise.

        Includes:
        - Thermal noise
        - 1/f noise (flicker noise)
        - Shot noise
        """
        # Thermal noise (white)
        thermal_noise = np.random.normal(0, 0.001, signal.shape)

        # 1/f noise (pink noise) - stronger at low frequencies
        # Simple approximation using filtered white noise
        pink_noise = np.random.normal(0, 0.002, signal.shape)
        # Apply low-pass filter to make it "pinker"
        for i in range(1, len(pink_noise)):
            pink_noise[i] = 0.9 * pink_noise[i-1] + 0.1 * pink_noise[i]

        # Shot noise (Poisson, proportional to sqrt(signal))
        shot_noise = np.random.normal(0, np.sqrt(np.abs(signal) + 1e-10) * 0.001)

        return signal + thermal_noise + pink_noise + shot_noise

    def add_adc_quantization(self, signal: np.ndarray) -> np.ndarray:
        """Add ADC quantization error."""
        # Normalize to ADC range
        signal_normalized = (signal + 1) / 2  # Map [-1, 1] to [0, 1]

        # Quantize
        quantized = np.round(signal_normalized * (self.quantization_levels - 1))
        quantized = quantized / (self.quantization_levels - 1)

        # Map back to [-1, 1]
        return quantized * 2 - 1

    def add_timing_jitter(
        self,
        signal: np.ndarray,
        jitter_std: float = 1e-11  # 10 ps jitter
    ) -> np.ndarray:
        """
        Add timing jitter effects.

        Real measurements have timing uncertainty that can cause
        errors when signals are changing rapidly.
        """
        if len(signal.shape) == 1 and len(signal) > 1:
            # Compute signal derivative (rate of change)
            derivative = np.gradient(signal)

            # Jitter causes error proportional to derivative
            jitter_error = derivative * np.random.normal(0, jitter_std, signal.shape)

            return signal + jitter_error

        return signal  # No jitter for static measurements

    def add_metastability(
        self,
        binary_signal: np.ndarray,
        metastability_rate: float = 1e-6
    ) -> np.ndarray:
        """
        Model metastability in bistable devices.

        Sometimes devices get stuck between states, leading to
        random outcomes or delayed switching.
        """
        # Random metastable events
        metastable = np.random.random(binary_signal.shape) < metastability_rate

        # Metastable readouts are random
        binary_signal[metastable] = np.random.choice([-1, 1], size=np.sum(metastable))

        return binary_signal

    def complete_readout(
        self,
        true_signal: np.ndarray,
        binary: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Complete readout with all error sources.

        Returns:
            Measured signal and error statistics
        """
        # Start with true signal
        measured = true_signal.copy()

        # Add amplifier noise
        measured = self.add_amplifier_noise(measured)

        # Add ADC quantization
        measured = self.add_adc_quantization(measured)

        # Add timing jitter
        measured = self.add_timing_jitter(measured)

        if binary:
            # Threshold to binary
            measured = np.sign(measured)

            # Add metastability
            measured = self.add_metastability(measured)

            # Compute error rate
            errors = (measured != np.sign(true_signal))
            error_rate = np.mean(errors)
        else:
            # Compute RMS error
            error_rate = np.sqrt(np.mean((measured - true_signal)**2))

        stats = {
            'error_rate': error_rate,
            'snr_db': 20 * np.log10(np.std(true_signal) / (error_rate + 1e-10)),
            'measured_mean': np.mean(measured),
            'measured_std': np.std(measured)
        }

        return measured, stats


class TemperatureDrift:
    """
    Model temperature variations and thermal effects.

    Temperature affects:
    1. Thermal noise amplitude
    2. Device resistance (TCR)
    3. Switching thresholds
    4. Coupling strengths
    """

    def __init__(self, specs: Optional[DeviceSpecifications] = None):
        self.specs = specs or DeviceSpecifications()
        self.time = 0.0
        self.base_temperature = 300.0  # Kelvin

        # Temperature coefficient of resistance (TCR)
        self.tcr = 0.002  # 0.2% per Kelvin

    def temperature_profile(
        self,
        time: float,
        include_gradients: bool = True
    ) -> Dict[str, Any]:
        """
        Compute temperature at given time including all effects.

        Includes:
        - Ambient temperature variations
        - Self-heating from operation
        - Thermal gradients across chip
        """
        self.time = time

        # Ambient variations (slow, ~hours)
        ambient = self.base_temperature + 2 * np.sin(2 * np.pi * time / 3600)

        # Self-heating (faster, ~seconds)
        # Depends on power dissipation
        power = 1e-3  # 1 mW typical
        self_heating = power * self.specs.thermal_resistance

        # Fast fluctuations (~milliseconds)
        fast_noise = 0.1 * np.sin(2 * np.pi * time / 0.001)

        # Total temperature
        temperature = ambient + self_heating + fast_noise

        result = {
            'temperature': temperature,
            'ambient': ambient,
            'self_heating': self_heating,
            'fluctuations': fast_noise
        }

        if include_gradients:
            # Thermal gradients across chip (center is hotter)
            n_devices = 100  # Example
            grid_size = int(np.sqrt(n_devices))

            gradient = np.zeros((grid_size, grid_size))
            center = grid_size // 2

            for i in range(grid_size):
                for j in range(grid_size):
                    distance = np.sqrt((i - center)**2 + (j - center)**2)
                    # Temperature falls off from center
                    gradient[i, j] = temperature - 0.1 * distance

            result['gradient'] = gradient.flatten()

        return result

    def apply_temperature_effects(
        self,
        nominal_params: Dict[str, np.ndarray],
        temperature: float
    ) -> Dict[str, np.ndarray]:
        """
        Apply temperature-dependent changes to device parameters.
        """
        delta_T = temperature - self.base_temperature

        adjusted_params = {}

        for key, value in nominal_params.items():
            if key == 'resistance':
                # Resistance increases with temperature
                adjusted_params[key] = value * (1 + self.tcr * delta_T)

            elif key == 'coupling':
                # Coupling strength varies with temperature
                # Magnetic couplings weaken at higher T
                adjusted_params[key] = value * (1 - 0.001 * delta_T)

            elif key == 'noise':
                # Thermal noise increases with temperature
                adjusted_params[key] = value * np.sqrt(temperature / self.base_temperature)

            else:
                adjusted_params[key] = value

        return adjusted_params

    def thermal_cycling_stress(
        self,
        n_cycles: int,
        temp_range: Tuple[float, float] = (250, 350)
    ) -> float:
        """
        Model device degradation from thermal cycling.

        Uses Coffin-Manson model for thermal fatigue:
        N_f = A * (ΔT)^(-n)
        where N_f is cycles to failure
        """
        delta_T = temp_range[1] - temp_range[0]

        # Coffin-Manson parameters (typical for electronics)
        A = 1e10  # Material constant
        n = 2.0  # Exponent (typically 1.5-3)

        # Cycles to failure
        cycles_to_failure = A * (delta_T ** (-n))

        # Fraction of lifetime consumed
        damage_fraction = n_cycles / cycles_to_failure

        # Degradation factor (0 = new, 1 = failed)
        degradation = min(damage_fraction, 1.0)

        if degradation > 0.5:
            warnings.warn(
                f"Device has consumed {degradation*100:.1f}% of thermal cycling lifetime!",
                category=UserWarning
            )

        return degradation