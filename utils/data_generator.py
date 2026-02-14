"""
Synthetic Data Generator for Cement Manufacturing
Uses Latin Hypercube Sampling to create realistic training data
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chemistry_calc import CementChemistry


class SyntheticDataGenerator:
    """
    Generates realistic cement manufacturing data using:
    - Latin Hypercube Sampling for input space coverage
    - Chemistry models for outputs
    - Noise injection for measurement uncertainty
    """
    
    def __init__(self, seed=42):
        """Initialize the generator with a random seed for reproducibility"""
        np.random.seed(seed)
        self.chem = CementChemistry()
        
        # Define realistic ranges for input variables (16 features)
        self.feature_ranges = {
            # Blend ratios (%) - sum must be 100
            'clinker_pct': (70, 95),
            'fly_ash_pct': (0, 25),
            'slag_pct': (0, 35),
            'limestone_pct': (0, 15),
            
            # Chemistry (oxide percentages in clinker)
            'CaO': (60, 67),
            'SiO2': (18, 24),
            'Al2O3': (4, 8),
            'Fe2O3': (2, 5),
            
            # Operations
            'kiln_temp': (1400, 1500),  # °C
            'kiln_rpm': (0.5, 4.0),
            'fuel_input': (80, 120),  # kg coal per ton clinker
            'residence_time': (20, 40),  # minutes
            
            # Economic
            'fuel_price': (50, 150),  # $/ton
            'carbon_tax': (20, 100),  # $/ton CO2
            'transport_cost': (0.1, 0.5),  # $/km
            'demand': (500, 3000)  # tons/day
        }
        
        self.feature_names = list(self.feature_ranges.keys())
    
    def generate_lhs_samples(self, n_samples=2000):
        """
        Generate samples using Latin Hypercube Sampling
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        
        Returns:
        --------
        pd.DataFrame : Sampled input features
        """
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=len(self.feature_names), seed=42)
        
        # Generate samples in [0, 1]^d
        samples = sampler.random(n=n_samples)
        
        # Scale to actual ranges
        scaled_samples = np.zeros_like(samples)
        for i, feature in enumerate(self.feature_names):
            lower, upper = self.feature_ranges[feature]
            scaled_samples[:, i] = samples[:, i] * (upper - lower) + lower
        
        # Create DataFrame
        df = pd.DataFrame(scaled_samples, columns=self.feature_names)
        
        # Fix blend ratios to sum to 100%
        blend_cols = ['clinker_pct', 'fly_ash_pct', 'slag_pct', 'limestone_pct']
        blend_sum = df[blend_cols].sum(axis=1)
        for col in blend_cols:
            df[col] = (df[col] / blend_sum) * 100
        
        return df
    
    def calculate_outputs(self, inputs_df):
        """
        Calculate output targets from input features
        
        Parameters:
        -----------
        inputs_df : pd.DataFrame
            Input features
        
        Returns:
        --------
        pd.DataFrame : Complete dataset with inputs and outputs
        """
        outputs = []
        
        for idx, row in inputs_df.iterrows():
            # Extract features
            clinker_pct = row['clinker_pct'] / 100
            fa_pct = row['fly_ash_pct'] / 100
            slag_pct = row['slag_pct'] / 100
            limestone_pct = row['limestone_pct'] / 100
            
            kiln_temp = row['kiln_temp']
            fuel_input = row['fuel_input']
            fuel_price = row['fuel_price']
            carbon_tax = row['carbon_tax']
            
            # --- Calculate Strength (MPa) ---
            # Use modified Bolomey equation
            # Assume 400 kg/m³ total cementitious, W/C = 0.45
            cement_content = 400
            water_content = 180
            
            scm_dict = {
                'fly_ash': row['fly_ash_pct'],
                'slag': row['slag_pct']
            }
            
            strength = self.chem.strength_prediction_bolomey(
                cement_content=cement_content,
                water_content=water_content,
                scm_percentages=scm_dict,
                age_days=28
            )
            
            # Temperature adjustment (higher temp -> slightly higher strength)
            temp_factor = 1 + (kiln_temp - 1450) / 1450 * 0.05
            strength = strength * temp_factor
            
            # --- Calculate Emissions (kg CO2 per kg cement) ---
            # Assume 1 ton of cement production
            clinker_mass = 1000 * clinker_pct  # kg clinker in 1 ton cement
            
            emissions_dict = self.chem.calculate_co2_emissions(
                clinker_mass=clinker_mass,
                fuel_mass=fuel_input,
                fuel_type='coal'
            )
            
            # Total emissions per kg cement
            emissions = emissions_dict['total'] / 1000  # kg CO2 per kg cement
            
            # --- Calculate Cost ($/ton cement) ---
            # Material costs (simplified)
            clinker_cost = 50  # $/ton
            fly_ash_cost = 20
            slag_cost = 30
            limestone_cost = 10
            
            material_cost = (
                clinker_pct * clinker_cost +
                fa_pct * fly_ash_cost * 100 +
                slag_pct * slag_cost * 100 +
                limestone_pct * limestone_cost * 100
            ) / 100
            
            # Energy cost
            energy_cost = fuel_input / 1000 * fuel_price
            
            # Carbon tax cost
            carbon_cost = emissions * carbon_tax
            
            # Transport cost (simplified)
            transport_cost = row['transport_cost'] * 50  # Assume 50 km average
            
            total_cost = material_cost + energy_cost + carbon_cost + transport_cost
            
            # --- Calculate Circularity (%) ---
            # Higher SCM usage = higher circularity
            circularity = (fa_pct + slag_pct) * 100
            
            # --- Calculate Risk (normalized variance) ---
            # Risk from temperature variance and blend complexity
            temp_risk = abs(kiln_temp - 1450) / 100  # Normalized
            blend_risk = (fa_pct + slag_pct) * 0.2  # More SCMs = slightly higher risk
            risk = (temp_risk + blend_risk) / 2
            
            outputs.append({
                'strength': strength,
                'emissions': emissions,
                'cost': total_cost,
                'circularity': circularity,
                'risk': risk
            })
        
        # Combine inputs and outputs
        outputs_df = pd.DataFrame(outputs)
        result_df = pd.concat([inputs_df.reset_index(drop=True), 
                               outputs_df.reset_index(drop=True)], axis=1)
        
        return result_df
    
    def add_noise(self, df, noise_level=0.05):
        """
        Add Gaussian noise to simulate measurement uncertainty
        
        Parameters:
        -----------
        df : pd.DataFrame
            Clean data
        noise_level : float
            Standard deviation as fraction of value (e.g., 0.05 = 5%)
        
        Returns:
        --------
        pd.DataFrame : Noisy data
        """
        noisy_df = df.copy()
        
        # Add noise to outputs only
        output_cols = ['strength', 'emissions', 'cost', 'circularity', 'risk']
        
        for col in output_cols:
            if col in noisy_df.columns:
                noise = np.random.normal(0, noise_level * noisy_df[col].abs(), 
                                        size=len(noisy_df))
                noisy_df[col] = noisy_df[col] + noise
                
                # Ensure positive values
                if col != 'risk':  # risk can be close to zero
                    noisy_df[col] = noisy_df[col].clip(lower=0.1)
        
        return noisy_df
    
    def generate_dataset(self, n_samples=2000, noise_level=0.05, 
                        train_ratio=0.7, val_ratio=0.15):
        """
        Generate complete dataset with train/val/test splits
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples
        noise_level : float
            Noise level for outputs
        train_ratio : float
            Fraction for training set
        val_ratio : float
            Fraction for validation set
        
        Returns:
        --------
        dict : Dictionary with 'train', 'val', 'test' DataFrames
        """
        print(f"Generating {n_samples} samples using Latin Hypercube Sampling...")
        
        # Step 1: Generate inputs
        inputs_df = self.generate_lhs_samples(n_samples)
        print(f"✓ Generated {len(inputs_df)} input samples")
        
        # Step 2: Calculate outputs
        print("Calculating outputs using chemistry models...")
        full_df = self.calculate_outputs(inputs_df)
        print("✓ Calculated all outputs (strength, emissions, cost, circularity, risk)")
        
        # Step 3: Add noise
        print(f"Adding {noise_level*100:.0f}% Gaussian noise...")
        full_df = self.add_noise(full_df, noise_level)
        print("✓ Added measurement noise")
        
        # Step 4: Split data
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Shuffle
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = full_df.iloc[:n_train]
        val_df = full_df.iloc[n_train:n_train+n_val]
        test_df = full_df.iloc[n_train+n_val:]
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_df)} samples ({train_ratio*100:.0f}%)")
        print(f"  Validation: {len(val_df)} samples ({val_ratio*100:.0f}%)")
        print(f"  Test: {len(test_df)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'full': full_df
        }
    
    def save_datasets(self, datasets, output_dir='data/synthetic'):
        """Save generated datasets to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, df in datasets.items():
            if split_name == 'full':
                continue
            filepath = os.path.join(output_dir, f'{split_name}_data.csv')
            df.to_csv(filepath, index=False)
            print(f"✓ Saved {filepath}")
        
        # Also save full dataset
        full_path = os.path.join(output_dir, 'full_dataset.csv')
        datasets['full'].to_csv(full_path, index=False)
        print(f"✓ Saved {full_path}")
    
    def print_statistics(self, datasets):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        df = datasets['full']
        
        print("\nInput Features:")
        for col in self.feature_names:
            print(f"  {col:20s}: [{df[col].min():6.2f}, {df[col].max():6.2f}] "
                  f"(mean: {df[col].mean():6.2f})")
        
        print("\nOutput Targets:")
        output_cols = ['strength', 'emissions', 'cost', 'circularity', 'risk']
        for col in output_cols:
            print(f"  {col:20s}: [{df[col].min():6.2f}, {df[col].max():6.2f}] "
                  f"(mean: {df[col].mean():6.2f})")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("SYNTHETIC DATA GENERATION FOR CEMENT OPTIMIZATION")
    print("="*60)
    
    # Create generator
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate datasets
    datasets = generator.generate_dataset(
        n_samples=2000,
        noise_level=0.07,  # 7% noise
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Print statistics
    generator.print_statistics(datasets)
    
    # Save to files
    print("\n" + "="*60)
    print("SAVING DATASETS")
    print("="*60)
    generator.save_datasets(datasets, output_dir='data/synthetic')
    
    print("\n" + "="*60)
    print("✓ DATA GENERATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Build MT-PINN model (models/pinn.py)")
    print("  2. Implement physics loss (models/physics_loss.py)")
    print("  3. Train the model (experiments/train_pinn.py)")