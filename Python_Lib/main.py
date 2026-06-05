from float_analyzer import CustomFloatSystem, FloatRepresentationAnalyzer
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # Initialize alternative setups
    # Let's compare standard bfloat16 representation vs Float16 (IEEE Half)
    bfloat16 = CustomFloatSystem(e_bits=8, f_bits=7)
    fp16 = CustomFloatSystem(e_bits=5, f_bits=10)
    
    print("==================================================")
    print("TASK 1: Structural Specifications Comparison")
    print("==================================================")
    df_specs = pd.DataFrame([bfloat16.get_specs(), fp16.get_specs()])
    print(df_specs.to_string(index=False))
    print("\n")

    print("==================================================")
    print("TASK 2: Quantization Analysis for a Target Input")
    print("==================================================")
    target_num = 345.678
    
    bf16_val, bf16_err = bfloat16.quantize(target_num)
    fp16_val, fp16_err = fp16.quantize(target_num)
    
    print(f"Target Real Number: {target_num}")
    print(f"bfloat16 (E=8, F=7)  -> Quantized: {bf16_val:<10} | Abs Error: {bf16_err:.5f}")
    print(f"fp16     (E=5, F=10) -> Quantized: {fp16_val:<10} | Abs Error: {fp16_err:.5f}")
    best_fit = "fp16 (More Fraction bits)" if fp16_err < bf16_err else "bfloat16 (More Exponent bits)"
    print(f"🏆 Best Fit Representation for {target_num}: {best_fit}")
    print("\n")

    print("==================================================")
    print("TASK 3: Evaluation over a Model/Dataset Distribution")
    print("==================================================")
    # Generating a mock neural network weights sample dataset (Normal distribution + standard outliers)
    np.random.seed(42)
    dataset = np.random.normal(loc=0.0, scale=15.0, size=1000)
    
    analysis_df = FloatRepresentationAnalyzer.evaluate_dataset(dataset, [bfloat16, fp16])
    print(analysis_df[['Config', 'Dataset MAE', 'Dataset RMSE', 'Overflow Count', 'Underflow Count']].to_string(index=False))
    print("\n")

    print("==================================================")
    print("TASK 4: Plotting Global Allocations For Fixed Budget")
    print("==================================================")
    print("Generating trade-off mapping window for a total budget of 16 bits...")
    FloatRepresentationAnalyzer.plot_tradeoffs(total_bits=16)