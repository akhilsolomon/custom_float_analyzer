import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any

class CustomFloatSystem:
    """
    Models an arbitrary floating-point number system with 1 sign bit,
    E exponent bits, and F fraction bits (Total bits = 1 + E + F).
    Uses standard IEEE 754 conventions (Biased exponent, implicit leading 1 for normals).
    """
    def __init__(self, e_bits: int, f_bits: int):
        self.e_bits = e_bits
        self.f_bits = f_bits
        self.total_bits = 1 + e_bits + f_bits
        self.bias = (2 ** (e_bits - 1)) - 1 if e_bits > 0 else 0
        
        # Calculate Range Metrics
        if e_bits > 0:
            # IEEE-like: Max exponent code is reserved for Inf/NaN, so max normal exponent code is (2^E - 2)
            max_exp_code = (2 ** e_bits) - 2
            self.max_normal_exp = max_exp_code - self.bias
            self.min_normal_exp = 1 - self.bias
            
            self.max_value = (2 - (2 ** -f_bits)) * (2 ** self.max_normal_exp)
            self.min_normal = 1.0 * (2 ** self.min_normal_exp)
            # Subnormal minimum value
            self.min_subnormal = (2 ** -f_bits) * (2 ** self.min_normal_exp)
        else:
            # No exponent means purely fixed-point fraction / scaled representation
            self.max_value = 1.0 - (2 ** -f_bits)
            self.min_normal = 2 ** -f_bits
            self.min_subnormal = 2 ** -f_bits

        # Machine Epsilon (Precision metric)
        self.epsilon = 2 ** -f_bits

    def get_specs(self) -> Dict[str, Any]:
        return {
            "Config": f"1-{self.e_bits}-{self.f_bits}",
            "Total Bits": self.total_bits,
            "Exponent Bits (E)": self.e_bits,
            "Fraction Bits (F)": self.f_bits,
            "Bias": self.bias,
            "Max Value": self.max_value,
            "Min Normal": self.min_normal,
            "Min Subnormal": self.min_subnormal,
            "Machine Epsilon (Precision)": self.epsilon
        }

    def quantize(self, val: float) -> Tuple[float, float]:
        """
        Simulates the quantization of a real python float into this custom format.
        Returns: (Quantized Value, Absolute Error)
        """
        sign = -1.0 if val < 0 else 1.0
        abs_val = abs(val)

        if abs_val == 0:
            return 0.0, 0.0
        
        # Overflow handling
        if abs_val > self.max_value:
            return sign * self.max_value, abs(abs_val - self.max_value)
        
        # Underflow handling (below smallest subnormal)
        if abs_val < self.min_subnormal:
            return 0.0, abs_val

        # Subnormal range
        if abs_val < self.min_normal:
            # Scale by the fixed subnormal exponent shift
            scaled = abs_val / (2 ** self.min_normal_exp)
            # Quantize the fraction directly without the implicit 1
            quantized_fraction = round(scaled * (2 ** self.f_bits)) / (2 ** self.f_bits)
            quantized_val = sign * quantized_fraction * (2 ** self.min_normal_exp)
            return quantized_val, abs(val - quantized_val)

        # Normal range
        # Find exponent using base-2 log
        exp = int(np.floor(np.log2(abs_val)))
        # Bound exponent to maximum capacity
        exp = min(exp, self.max_normal_exp)
        exp = max(exp, self.min_normal_exp)
        
        mantissa = abs_val / (2 ** exp)  # Will be in range [1.0, 2.0)
        
        # Quantize the fractional part (subtract implicit 1)
        fraction = mantissa - 1.0
        quantized_fraction = round(fraction * (2 ** self.f_bits)) / (2 ** self.f_bits)
        
        quantized_val = sign * (1.0 + quantized_fraction) * (2 ** exp)
        return quantized_val, abs(val - quantized_val)


class FloatRepresentationAnalyzer:
    """
    Analyzes, compares, and profiles datasets across multiple CustomFloatSystems.
    """
    @staticmethod
    def evaluate_dataset(data: np.ndarray, systems: List[CustomFloatSystem]) -> pd.DataFrame:
        """Evaluates a dataset against multiple configurations to find metrics."""
        results = []
        for sys in systems:
            errors = []
            overflows = 0
            underflows = 0
            
            for val in data:
                q_val, err = sys.quantize(val)
                errors.append(err)
                if abs(val) > sys.max_value:
                    overflows += 1
                elif abs(val) < sys.min_subnormal and val != 0:
                    underflows += 1
                    
            errors = np.array(errors)
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(errors**2))
            
            specs = sys.get_specs()
            specs.update({
                "Dataset MAE": mae,
                "Dataset RMSE": rmse,
                "Overflow Count": overflows,
                "Underflow Count": underflows
            })
            results.append(specs)
            
        return pd.DataFrame(results)

    @staticmethod
    def plot_tradeoffs(total_bits: int):
        """Plots Precision vs Range tradeoff by shifting bit resource allocations safely."""
        e_variants = list(range(2, total_bits - 2))
        max_values_log10 = []
        epsilons = []
        configs = []

        for e in e_variants:
            f = total_bits - 1 - e
            sys = CustomFloatSystem(e, f)
            configs.append(f"E={e}, F={f}")
            epsilons.append(sys.epsilon)
            
            # Safely calculate log10(max_value) to avoid OverflowError
            # log10((2 - 2^-F) * 2^max_normal_exp) = log10(2 - 2^-F) + max_normal_exp * log10(2)
            try:
                if sys.max_normal_exp > 1000: # Beyond standard float limits
                    log10_val = np.log10(2 - (2 ** -f)) + (sys.max_normal_exp * np.log10(2))
                else:
                    log10_val = np.log10(sys.max_value)
                max_values_log10.append(log10_val)
            except (OverflowError, RuntimeWarning):
                log10_val = np.log10(2 - (2 ** -f)) + (sys.max_normal_exp * np.log10(2))
                max_values_log10.append(log10_val)

        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:red'
        ax1.set_xlabel(f'Bit Allocations (Total={total_bits} bits, 1 Sign Bit)')
        ax1.set_ylabel('Dynamic Range (Max Value Log10)', color=color)
        ax1.plot(configs, max_values_log10, color=color, marker='o', linewidth=2, label='Max Value')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Machine Epsilon / Precision (Log10)', color=color)
        ax2.plot(configs, np.log10(epsilons), color=color, marker='s', linestyle='--', linewidth=2, label='Precision (Epsilon)')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Precision vs. Range Tradeoff Window ({total_bits}-bit budget)')
        fig.tight_layout()
        plt.show()