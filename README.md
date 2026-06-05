# Custom Floating-Point Architecture Analyzer

This repository contains a modular Python library designed to simulate, analyze, and profile custom floating-point number representation systems. It allows users to vary the Exponent ($E$) and Fraction ($F$) bit allocations under arbitrary bit budgets to determine the optimal configuration for specific deep learning models or datasets.

---

## Project Overview

Modern hardware architectures (like TPUs and custom AI accelerators) increasingly rely on alternative precision formats like `bfloat16` or `FP16` to optimize compute efficiency and memory bandwidth. This framework provides an analytical tool to explore those architectural trade-offs quantitatively.

### Key Capabilities
* **Arbitrary Bit Formatting:** Simulates custom floating-point representations using standard IEEE 754 conventions (sign bit, biased exponent, and implicit leading mantissa bit).
* **Quantization & Error Analysis:** Maps real numbers to custom bit configurations and calculates absolute representation errors to evaluate the best fit.
* **Dataset Profiling:** Evaluates complete datasets to measure Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and tracks numerical underflow and overflow conditions.
* **Trade-off Visualization:** Programmatically charts the global design window for a fixed bit budget, illustrating the inverse relationship between dynamic range and precision resolution.

---

## Repository Structure

* `float_analyzer.py`: Contains the core simulator class (`CustomFloatSystem`) and the evaluation engine class (`FloatRepresentationAnalyzer`).
* `main.py`: A demonstration script executing functional verification workflows across structural comparisons, single-value quantization checks, data distribution analysis, and tradeoff plotting.

---

## Getting Started

### Prerequisites

The project requires Python 3 and basic scientific computing libraries. Install the necessary dependencies via pip:

```bash
pip install numpy pandas matplotlib
