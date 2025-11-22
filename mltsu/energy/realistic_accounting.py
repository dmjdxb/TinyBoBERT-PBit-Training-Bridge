"""
Realistic energy accounting that includes ALL system costs.

This replaces the oversimplified energy claims in the original code
with honest accounting that includes:
1. Core operation energy
2. Control logic overhead (CMOS circuits)
3. Data movement costs (DRAM, interconnect)
4. Thermal management (cooling)
5. Power delivery losses

The original claims of 1 femtojoule per operation ignored 99.9% of the actual energy costs!

References:
    [1] Landauer (1961). "Irreversibility and heat generation" - kT ln(2) limit
    [2] ITRS (2020). "International Technology Roadmap for Semiconductors"
    [3] Pedram & Nazarian (2006). "Thermal Modeling, Analysis, and Management"
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ROOM_TEMPERATURE = 300  # Kelvin


class OperationType(Enum):
    """Types of operations in TSU system."""
    SPIN_FLIP = "spin_flip"
    READOUT = "readout"
    COUPLING_UPDATE = "coupling_update"
    DATA_TRANSFER = "data_transfer"
    ERROR_CORRECTION = "error_correction"


@dataclass
class EnergyBreakdown:
    """Complete energy breakdown for an operation."""

    # Core operation
    switching_energy: float = 0.0  # Fundamental bit switching

    # Readout chain
    sensing_energy: float = 0.0  # Detecting state
    amplification_energy: float = 0.0  # Signal amplification
    adc_energy: float = 0.0  # Analog-to-digital conversion

    # Control overhead
    clock_energy: float = 0.0  # Clock distribution
    control_logic_energy: float = 0.0  # Digital control (CMOS)
    timing_energy: float = 0.0  # Timing generation

    # Data movement
    on_chip_interconnect: float = 0.0  # Moving data on chip
    off_chip_io: float = 0.0  # Chip I/O
    memory_access: float = 0.0  # DRAM/SRAM access

    # System overhead
    cooling_energy: float = 0.0  # Active cooling
    power_conversion_loss: float = 0.0  # AC-DC, voltage regulation

    @property
    def total(self) -> float:
        """Total energy including ALL costs."""
        return sum([
            self.switching_energy,
            self.sensing_energy,
            self.amplification_energy,
            self.adc_energy,
            self.clock_energy,
            self.control_logic_energy,
            self.timing_energy,
            self.on_chip_interconnect,
            self.off_chip_io,
            self.memory_access,
            self.cooling_energy,
            self.power_conversion_loss
        ])

    @property
    def core_only(self) -> float:
        """Just the switching energy (what naive models report)."""
        return self.switching_energy

    @property
    def overhead_ratio(self) -> float:
        """Ratio of overhead to core energy."""
        if self.core_only > 0:
            return (self.total - self.core_only) / self.core_only
        return float('inf')

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for analysis."""
        return {
            'switching': self.switching_energy,
            'sensing': self.sensing_energy,
            'amplification': self.amplification_energy,
            'adc': self.adc_energy,
            'clock': self.clock_energy,
            'control_logic': self.control_logic_energy,
            'timing': self.timing_energy,
            'interconnect': self.on_chip_interconnect,
            'io': self.off_chip_io,
            'memory': self.memory_access,
            'cooling': self.cooling_energy,
            'power_loss': self.power_conversion_loss,
            'total': self.total,
            'overhead_ratio': self.overhead_ratio
        }


@dataclass
class RealisticEnergyModel:
    """
    Physically realistic energy model with complete accounting.

    This replaces the fantasy numbers in the original code with
    values based on actual physics and engineering constraints.
    """

    # Temperature (affects all thermal processes)
    temperature: float = ROOM_TEMPERATURE  # Kelvin

    # ===== FUNDAMENTAL LIMITS =====
    # These are theoretical minimums that can never be beaten

    @property
    def landauer_limit(self) -> float:
        """Fundamental thermodynamic limit for bit erasure."""
        return BOLTZMANN_CONSTANT * self.temperature * np.log(2)  # ~3e-21 J at 300K

    @property
    def thermal_energy(self) -> float:
        """kT energy scale."""
        return BOLTZMANN_CONSTANT * self.temperature  # ~4e-21 J at 300K

    # ===== P-BIT/TSU OPERATION ENERGY =====
    # Realistic values including device physics

    # Core switching (optimistic but possible)
    pbit_switching_energy: float = 1e-14  # 10 fJ - 1000x above Landauer limit

    # Readout chain (cannot be avoided)
    sensing_energy: float = 1e-13  # 100 fJ - Detecting magnetic/resistive state
    amplification_energy: float = 5e-13  # 500 fJ - Amplifying to logic levels
    adc_energy: float = 1e-12  # 1 pJ - 12-bit ADC at 1 Gsps

    # Control logic (CMOS overhead)
    cmos_gate_energy: float = 1e-15  # 1 fJ per gate at 7nm
    gates_per_operation: int = 1000  # Realistic control complexity

    # ===== MEMORY AND DATA MOVEMENT =====
    # Often dominates total energy!

    # SRAM (on-chip cache)
    sram_read_energy: float = 5e-15  # 5 fJ/bit at 7nm
    sram_write_energy: float = 1e-14  # 10 fJ/bit at 7nm

    # DRAM (main memory)
    dram_read_energy: float = 2e-12  # 2 pJ/bit
    dram_write_energy: float = 2e-12  # 2 pJ/bit

    # Interconnect
    wire_energy_per_mm: float = 1e-13  # 100 fJ/bit/mm on-chip
    typical_distance_mm: float = 5.0  # 5mm average on-chip distance

    # ===== SYSTEM INTEGRATION =====

    # Clock distribution (significant at GHz frequencies)
    clock_power_per_element: float = 1e-6  # 1 μW per clocked element
    clock_frequency: float = 1e9  # 1 GHz

    # Thermal management
    cooling_efficiency: float = 0.3  # COP for active cooling

    # Power delivery
    ac_dc_efficiency: float = 0.85  # AC to DC conversion
    voltage_regulation_efficiency: float = 0.90  # On-chip regulation
    distribution_efficiency: float = 0.95  # Power distribution network

    @property
    def total_power_efficiency(self) -> float:
        """Overall wall-plug to chip efficiency."""
        return (self.ac_dc_efficiency *
                self.voltage_regulation_efficiency *
                self.distribution_efficiency)  # ~72%

    def compute_operation_energy(
        self,
        operation: OperationType,
        n_bits: int = 1,
        include_overhead: bool = True
    ) -> EnergyBreakdown:
        """
        Compute complete energy for an operation.

        Args:
            operation: Type of operation
            n_bits: Number of bits involved
            include_overhead: Include all system overhead (default True)

        Returns:
            Complete energy breakdown
        """
        breakdown = EnergyBreakdown()

        # Core switching energy
        if operation in [OperationType.SPIN_FLIP, OperationType.COUPLING_UPDATE]:
            breakdown.switching_energy = self.pbit_switching_energy * n_bits

        # Readout energy (always needed to observe result)
        if operation != OperationType.DATA_TRANSFER:
            breakdown.sensing_energy = self.sensing_energy * n_bits
            breakdown.amplification_energy = self.amplification_energy * n_bits
            breakdown.adc_energy = self.adc_energy * n_bits

        if include_overhead:
            # Control logic energy
            breakdown.control_logic_energy = (
                self.cmos_gate_energy * self.gates_per_operation
            )

            # Clock energy
            breakdown.clock_energy = (
                self.clock_power_per_element / self.clock_frequency
            )

            # Data movement (assuming some SRAM access)
            breakdown.memory_access = self.sram_read_energy * 32  # 32-bit word

            # On-chip communication
            breakdown.on_chip_interconnect = (
                self.wire_energy_per_mm * self.typical_distance_mm * n_bits
            )

            # Power delivery losses (applied to all active energy)
            active_energy = (
                breakdown.switching_energy +
                breakdown.sensing_energy +
                breakdown.amplification_energy +
                breakdown.adc_energy +
                breakdown.control_logic_energy +
                breakdown.clock_energy +
                breakdown.memory_access +
                breakdown.on_chip_interconnect
            )

            breakdown.power_conversion_loss = (
                active_energy * (1 / self.total_power_efficiency - 1)
            )

            # Cooling energy (remove the heat generated)
            breakdown.cooling_energy = active_energy / self.cooling_efficiency

        return breakdown

    def compare_to_claims(self, claimed_energy: float) -> Dict[str, float]:
        """
        Compare realistic energy to claimed values.

        Args:
            claimed_energy: What the original code claims (usually 1e-15 J)

        Returns:
            Comparison metrics
        """
        realistic = self.compute_operation_energy(OperationType.SPIN_FLIP)

        return {
            'claimed_energy_J': claimed_energy,
            'realistic_energy_J': realistic.total,
            'core_only_energy_J': realistic.core_only,
            'overhead_energy_J': realistic.total - realistic.core_only,
            'claim_error_factor': realistic.total / claimed_energy,
            'overhead_percentage': realistic.overhead_ratio * 100,
            'landauer_limit_J': self.landauer_limit,
            'above_landauer_factor': realistic.total / self.landauer_limit
        }

    def energy_per_flop_equivalent(self) -> float:
        """
        Energy per floating-point-operation equivalent.

        TSU operations are not FLOPs, but we can estimate equivalent
        computational work for comparison.
        """
        # One Ising update ~= 10 FLOPs (energy calculation + probability)
        tsu_energy = self.compute_operation_energy(OperationType.SPIN_FLIP).total
        return tsu_energy / 10  # Energy per "equivalent FLOP"


class SystemEnergyCalculator:
    """Calculate energy for complete system operations."""

    def __init__(self, model: Optional[RealisticEnergyModel] = None):
        self.model = model or RealisticEnergyModel()

    def ising_iteration(
        self,
        n_spins: int,
        n_sweeps: int = 1
    ) -> Dict[str, float]:
        """
        Energy for complete Ising model iteration.

        Args:
            n_spins: Number of spins in system
            n_sweeps: Number of complete sweeps

        Returns:
            Energy breakdown
        """
        total_energy = 0.0
        breakdowns = []

        for _ in range(n_sweeps):
            for _ in range(n_spins):
                # Each spin update
                breakdown = self.model.compute_operation_energy(
                    OperationType.SPIN_FLIP,
                    n_bits=1
                )
                breakdowns.append(breakdown)
                total_energy += breakdown.total

        # Add data transfer energy (reading final state)
        data_energy = n_spins * 32 * self.model.dram_read_energy  # 32 bits per spin
        total_energy += data_energy

        return {
            'total_energy_J': total_energy,
            'energy_per_spin_J': total_energy / (n_spins * n_sweeps),
            'core_energy_J': sum(b.core_only for b in breakdowns),
            'overhead_energy_J': sum(b.total - b.core_only for b in breakdowns),
            'data_transfer_J': data_energy,
            'time_seconds': n_spins * n_sweeps / 1e9,  # Assuming 1 GHz
            'average_power_W': total_energy / (n_spins * n_sweeps / 1e9)
        }

    def attention_layer(
        self,
        seq_length: int,
        d_model: int,
        n_heads: int,
        n_samples: int = 32
    ) -> Dict[str, float]:
        """
        Energy for thermodynamic attention layer.

        Args:
            seq_length: Sequence length
            d_model: Model dimension
            n_heads: Number of attention heads
            n_samples: Monte Carlo samples for TSU attention

        Returns:
            Energy comparison
        """
        # TSU attention: sample binary patterns
        tsu_operations = seq_length * seq_length * n_heads * n_samples
        tsu_energy = self.model.compute_operation_energy(
            OperationType.SPIN_FLIP,
            n_bits=tsu_operations
        ).total

        # Standard attention: matrix multiply
        # Energy ~ 2 * seq_length^2 * d_model * energy_per_MAC
        gpu_energy_per_mac = 1e-12  # 1 pJ per MAC on modern GPU
        gpu_operations = 2 * seq_length * seq_length * d_model
        gpu_energy = gpu_operations * gpu_energy_per_mac

        return {
            'tsu_energy_J': tsu_energy,
            'gpu_energy_J': gpu_energy,
            'energy_ratio': tsu_energy / gpu_energy,
            'tsu_operations': tsu_operations,
            'gpu_operations': gpu_operations,
            'samples_overhead': n_samples  # Need many samples for accuracy
        }


class EnergyComparison:
    """Compare TSU energy to conventional computing."""

    def __init__(self):
        self.tsu_model = RealisticEnergyModel()

        # GPU specs (e.g., NVIDIA A100)
        self.gpu_tdp = 400  # Watts
        self.gpu_peak_tflops = 19.5  # FP32 TFLOPS
        self.gpu_energy_per_flop = self.gpu_tdp / (self.gpu_peak_tflops * 1e12)

        # CPU specs (e.g., Intel Xeon)
        self.cpu_tdp = 250  # Watts
        self.cpu_peak_gflops = 1000  # GFLOPS
        self.cpu_energy_per_flop = self.cpu_tdp / (self.cpu_peak_gflops * 1e9)

    def generate_comparison_table(self) -> str:
        """Generate comparison table of energy costs."""

        tsu_energy = self.tsu_model.compute_operation_energy(
            OperationType.SPIN_FLIP,
            include_overhead=True
        )

        lines = [
            "=" * 70,
            "HONEST ENERGY COMPARISON (Including ALL Overhead)",
            "=" * 70,
            "",
            "Fundamental Limits:",
            f"  Landauer limit (kT ln 2):        {self.tsu_model.landauer_limit:.2e} J",
            f"  Thermal energy (kT):             {self.tsu_model.thermal_energy:.2e} J",
            "",
            "TSU/P-bit Operation (REALISTIC):",
            f"  Core switching only:             {tsu_energy.core_only:.2e} J",
            f"  + Readout & amplification:       {tsu_energy.sensing_energy + tsu_energy.amplification_energy:.2e} J",
            f"  + Control logic (CMOS):          {tsu_energy.control_logic_energy:.2e} J",
            f"  + Data movement:                 {tsu_energy.memory_access + tsu_energy.on_chip_interconnect:.2e} J",
            f"  + Power delivery losses:         {tsu_energy.power_conversion_loss:.2e} J",
            f"  + Cooling:                       {tsu_energy.cooling_energy:.2e} J",
            f"  = TOTAL REALISTIC:               {tsu_energy.total:.2e} J",
            f"  Overhead ratio:                  {tsu_energy.overhead_ratio:.1f}x",
            "",
            "Conventional Computing:",
            f"  GPU (A100) per FLOP:             {self.gpu_energy_per_flop:.2e} J",
            f"  CPU (Xeon) per FLOP:             {self.cpu_energy_per_flop:.2e} J",
            "",
            "REALISTIC Advantage:",
            f"  TSU vs GPU:                      {self.gpu_energy_per_flop / (tsu_energy.total / 10):.1f}x",
            f"  TSU vs CPU:                      {self.cpu_energy_per_flop / (tsu_energy.total / 10):.1f}x",
            "",
            "Original Claims vs Reality:",
            f"  Original claim:                  1e-15 J (1 fJ)",
            f"  Reality:                         {tsu_energy.total:.2e} J",
            f"  Claim error:                     {tsu_energy.total / 1e-15:.0f}x too optimistic",
            "",
            "=" * 70,
            "CONCLUSION: Realistic advantage is 10-100x, not 1000x",
            "The original claims ignored 99.9% of the actual energy costs!",
            "=" * 70
        ]

        return "\n".join(lines)


class PowerDeliveryModel:
    """Model realistic power delivery and thermal constraints."""

    def __init__(self):
        # Chip specifications
        self.chip_area_mm2 = 100  # 10mm x 10mm chip
        self.max_power_density_w_per_mm2 = 1.0  # Thermal limit

        # Package specifications
        self.package_thermal_resistance = 0.5  # K/W
        self.max_junction_temp = 125  # Celsius
        self.ambient_temp = 25  # Celsius

    def max_sustainable_power(self) -> float:
        """Maximum power before thermal throttling."""
        # Thermal constraint
        thermal_limit = (
            (self.max_junction_temp - self.ambient_temp) /
            self.package_thermal_resistance
        )

        # Power density constraint
        density_limit = self.chip_area_mm2 * self.max_power_density_w_per_mm2

        return min(thermal_limit, density_limit)

    def operations_per_second(self, energy_per_op: float) -> float:
        """Maximum operations per second under thermal constraints."""
        max_power = self.max_sustainable_power()
        return max_power / energy_per_op

    def efficiency_vs_frequency(self, frequency: float) -> float:
        """Power efficiency as function of operating frequency."""
        # Efficiency drops at high frequency due to:
        # - Higher leakage at higher voltage
        # - Worse voltage regulator efficiency
        # - Increased switching losses

        f_nominal = 1e9  # 1 GHz nominal

        if frequency <= f_nominal:
            return 0.9  # 90% efficiency at or below nominal
        else:
            # Efficiency drops with frequency
            return 0.9 * (f_nominal / frequency) ** 0.5


def demonstrate_reality_check():
    """Demonstrate the huge gap between claims and reality."""

    print("\n" + "=" * 80)
    print("REALITY CHECK: TSU/P-bit Energy Claims vs. Physical Reality")
    print("=" * 80)

    # Create realistic model
    model = RealisticEnergyModel()
    comparison = EnergyComparison()

    # Show comparison table
    print(comparison.generate_comparison_table())

    # Show specific examples
    calculator = SystemEnergyCalculator(model)

    print("\n" + "=" * 80)
    print("EXAMPLE: 100-spin Ising Model Iteration")
    print("=" * 80)

    ising_result = calculator.ising_iteration(n_spins=100, n_sweeps=100)

    print(f"Total energy:           {ising_result['total_energy_J']:.2e} J")
    print(f"Core energy only:       {ising_result['core_energy_J']:.2e} J")
    print(f"Overhead energy:        {ising_result['overhead_energy_J']:.2e} J")
    print(f"Average power:          {ising_result['average_power_W']:.3f} W")
    print(f"Time:                   {ising_result['time_seconds']*1e6:.1f} μs")

    # Issue warning
    warnings.warn(
        "\n" + "=" * 80 + "\n"
        "SCIENTIFIC INTEGRITY WARNING:\n"
        "The energy numbers in this codebase have been updated to reflect\n"
        "realistic physical constraints. The original claims of 1 fJ per\n"
        "operation ignored control logic, data movement, power delivery,\n"
        "and cooling - which dominate the actual energy consumption.\n"
        "\n"
        "Realistic advantage: 10-100x over GPUs (still valuable!)\n"
        "Original claim: 1000x (physically impossible with current technology)\n"
        + "=" * 80,
        category=UserWarning
    )


if __name__ == "__main__":
    demonstrate_reality_check()