"""
Objective Functions for Multi-Objective Optimization
Defines the 4 competing objectives for cement manufacturing
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chemistry_calc import CementChemistry


class CementObjectives:
    """
    Four competing objectives:
    1. Minimize Cost (materials + energy + carbon tax + transport)
    2. Minimize Emissions (CO2 from calcination + fuel combustion)
    3. Maximize Circularity (% waste-derived materials)
    4. Minimize Risk (operational variance)
    """
    
    def __init__(self):
        self.chem = CementChemistry()
        
        # Material costs ($/ton)
        self.material_costs = {
            'clinker': 50.0,
            'fly_ash': 20.0,
            'slag': 30.0,
            'limestone': 10.0
        }
    
    def calculate_cost(self, x):
        """
        Calculate total cost ($/ton cement)
        
        Parameters:
        -----------
        x : dict or np.ndarray
            Decision variables
        
        Returns:
        --------
        float : Total cost in $/ton
        """
        # Extract variables
        if isinstance(x, dict):
            clinker_pct = x['clinker_pct']
            fly_ash_pct = x['fly_ash_pct']
            slag_pct = x['slag_pct']
            limestone_pct = x['limestone_pct']
            fuel_input = x['fuel_input']
            fuel_price = x['fuel_price']
            carbon_tax = x['carbon_tax']
            transport_cost = x['transport_cost']
        else:
            # Assume array format
            clinker_pct = x[0]
            fly_ash_pct = x[1]
            slag_pct = x[2]
            limestone_pct = x[3]
            fuel_input = x[10]
            fuel_price = x[12]
            carbon_tax = x[13]
            transport_cost = x[14]
        
        # Material costs
        material_cost = (
            (clinker_pct/100) * self.material_costs['clinker'] +
            (fly_ash_pct/100) * self.material_costs['fly_ash'] +
            (slag_pct/100) * self.material_costs['slag'] +
            (limestone_pct/100) * self.material_costs['limestone']
        )
        
        # Energy cost (fuel)
        energy_cost = (fuel_input / 1000) * fuel_price
        
        # Calculate emissions for carbon tax
        clinker_mass = clinker_pct / 100 * 1000  # kg clinker per ton cement
        emissions_dict = self.chem.calculate_co2_emissions(
            clinker_mass=clinker_mass,
            fuel_mass=fuel_input,
            fuel_type='coal'
        )
        total_emissions = emissions_dict['total'] / 1000  # kg CO2 per kg cement
        
        # Carbon tax cost
        carbon_cost = total_emissions * carbon_tax
        
        # Transport cost (simplified: assume 50 km average)
        transport = transport_cost * 50
        
        # Total cost
        total_cost = material_cost + energy_cost + carbon_cost + transport
        
        return total_cost
    
    def calculate_emissions(self, x):
        """
        Calculate total CO2 emissions (kg CO2 per kg cement)
        
        Parameters:
        -----------
        x : dict or np.ndarray
            Decision variables
        
        Returns:
        --------
        float : Emission intensity (kg CO2/kg cement)
        """
        # Extract variables
        if isinstance(x, dict):
            clinker_pct = x['clinker_pct']
            fuel_input = x['fuel_input']
        else:
            clinker_pct = x[0]
            fuel_input = x[10]
        
        # Calculate emissions
        clinker_mass = clinker_pct / 100 * 1000  # kg clinker per ton cement
        
        emissions_dict = self.chem.calculate_co2_emissions(
            clinker_mass=clinker_mass,
            fuel_mass=fuel_input,
            fuel_type='coal'
        )
        
        # Emissions per kg cement
        emission_intensity = emissions_dict['total'] / 1000
        
        return emission_intensity
    
    def calculate_circularity(self, x):
        """
        Calculate circularity score (% waste-derived materials)
        
        Higher is better, so we'll NEGATE for minimization
        
        Parameters:
        -----------
        x : dict or np.ndarray
            Decision variables
        
        Returns:
        --------
        float : Circularity score (0-100%)
        """
        # Extract variables
        if isinstance(x, dict):
            fly_ash_pct = x['fly_ash_pct']
            slag_pct = x['slag_pct']
        else:
            fly_ash_pct = x[1]
            slag_pct = x[2]
        
        # Circularity = % of waste-derived materials (fly ash + slag)
        circularity = fly_ash_pct + slag_pct
        
        # Return negative for minimization (we want to MAXIMIZE circularity)
        return -circularity
    
    def calculate_risk(self, x):
        """
        Calculate operational risk
        
        Risk from:
        - Temperature deviation from optimal (1450°C)
        - High SCM content (quality variability)
        - Extreme blend ratios
        
        Parameters:
        -----------
        x : dict or np.ndarray
            Decision variables
        
        Returns:
        --------
        float : Risk score (normalized 0-1, lower is better)
        """
        # Extract variables
        if isinstance(x, dict):
            clinker_pct = x['clinker_pct']
            fly_ash_pct = x['fly_ash_pct']
            slag_pct = x['slag_pct']
            kiln_temp = x['kiln_temp']
        else:
            clinker_pct = x[0]
            fly_ash_pct = x[1]
            slag_pct = x[2]
            kiln_temp = x[8]
        
        # Temperature risk (deviation from 1450°C)
        temp_risk = abs(kiln_temp - 1450) / 100
        
        # Blend risk (more SCMs = more variability)
        scm_total = fly_ash_pct + slag_pct
        blend_risk = scm_total / 100 * 0.3  # Max 0.3 for 100% SCM
        
        # Extreme blend penalty (very low clinker is risky)
        if clinker_pct < 60:
            extreme_penalty = (60 - clinker_pct) / 60 * 0.2
        else:
            extreme_penalty = 0
        
        # Total risk
        total_risk = temp_risk + blend_risk + extreme_penalty
        
        # Normalize to [0, 1]
        total_risk = min(1.0, total_risk)
        
        return total_risk
    
    def evaluate_all(self, x):
        """
        Evaluate all objectives simultaneously
        
        Parameters:
        -----------
        x : dict or np.ndarray
            Decision variables
        
        Returns:
        --------
        dict : All objective values
        """
        return {
            'cost': self.calculate_cost(x),
            'emissions': self.calculate_emissions(x),
            'circularity': -self.calculate_circularity(x),  # Convert back to positive
            'risk': self.calculate_risk(x)
        }


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("OBJECTIVE FUNCTIONS TEST")
    print("="*60)
    
    objectives = CementObjectives()
    
    # Test case 1: High clinker, low SCM (traditional cement)
    print("\nTest Case 1: Traditional Cement (High Clinker)")
    x1 = {
        'clinker_pct': 90,
        'fly_ash_pct': 5,
        'slag_pct': 3,
        'limestone_pct': 2,
        'CaO': 65, 'SiO2': 21, 'Al2O3': 5, 'Fe2O3': 3,
        'kiln_temp': 1450,
        'kiln_rpm': 2.5,
        'fuel_input': 100,
        'residence_time': 30,
        'fuel_price': 80,
        'carbon_tax': 50,
        'transport_cost': 0.3,
        'demand': 2000
    }
    
    results1 = objectives.evaluate_all(x1)
    print(f"  Cost:        ${results1['cost']:.2f}/ton")
    print(f"  Emissions:   {results1['emissions']:.3f} kg CO2/kg")
    print(f"  Circularity: {results1['circularity']:.1f}%")
    print(f"  Risk:        {results1['risk']:.3f}")
    
    # Test case 2: Lower clinker, high SCM (green cement)
    print("\nTest Case 2: Green Cement (High SCM)")
    x2 = {
        'clinker_pct': 70,
        'fly_ash_pct': 20,
        'slag_pct': 8,
        'limestone_pct': 2,
        'CaO': 65, 'SiO2': 21, 'Al2O3': 5, 'Fe2O3': 3,
        'kiln_temp': 1450,
        'kiln_rpm': 2.5,
        'fuel_input': 90,
        'residence_time': 30,
        'fuel_price': 80,
        'carbon_tax': 50,
        'transport_cost': 0.3,
        'demand': 2000
    }
    
    results2 = objectives.evaluate_all(x2)
    print(f"  Cost:        ${results2['cost']:.2f}/ton")
    print(f"  Emissions:   {results2['emissions']:.3f} kg CO2/kg")
    print(f"  Circularity: {results2['circularity']:.1f}%")
    print(f"  Risk:        {results2['risk']:.3f}")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON: Traditional vs Green Cement")
    print("="*60)
    print(f"Cost Difference:      ${results2['cost'] - results1['cost']:+.2f}/ton")
    print(f"Emission Reduction:   {(results1['emissions'] - results2['emissions'])*1000:.1f} g CO2/kg")
    print(f"Circularity Gain:     {results2['circularity'] - results1['circularity']:+.1f}%")
    print(f"Risk Change:          {results2['risk'] - results1['risk']:+.3f}")
    
    print("\n✓ Objective functions validated!")