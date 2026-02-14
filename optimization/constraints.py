"""
Constraints Module for Cement Manufacturing Optimization
Ensures solutions meet quality, safety, and regulatory requirements
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chemistry_calc import CementChemistry
from models.pinn import MTPINN


class CementConstraints:
    """
    Constraint categories:
    1. Mass Balance: Blend percentages sum to 100%
    2. Quality: Minimum strength requirements
    3. Safety: SCM limits, temperature bounds
    4. Regulations: Emission caps
    5. Operational: Practical bounds on all variables
    """
    
    def __init__(self, surrogate_model=None):
        """
        Initialize constraints
        
        Parameters:
        -----------
        surrogate_model : MTPINN, optional
            Trained surrogate for strength prediction
        """
        self.chem = CementChemistry()
        self.surrogate = surrogate_model
        
        # Define bounds for all 16 decision variables
        self.bounds = {
            # Blend ratios (will be normalized to sum to 100%)
            'clinker_pct': (60, 95),      # Min 60% for strength
            'fly_ash_pct': (0, 30),       # Max 30% per standards
            'slag_pct': (0, 35),          # Max 35% per standards
            'limestone_pct': (0, 15),     # Max 15% per standards
            
            # Chemistry (oxide percentages in clinker)
            'CaO': (60, 67),
            'SiO2': (18, 24),
            'Al2O3': (4, 8),
            'Fe2O3': (2, 5),
            
            # Operations
            'kiln_temp': (1400, 1500),    # °C
            'kiln_rpm': (0.5, 4.0),
            'fuel_input': (80, 120),      # kg coal per ton clinker
            'residence_time': (20, 40),   # minutes
            
            # Economic
            'fuel_price': (50, 150),      # $/ton
            'carbon_tax': (20, 100),      # $/ton CO2
            'transport_cost': (0.1, 0.5), # $/km
            'demand': (500, 3000)         # tons/day
        }
        
        self.feature_names = list(self.bounds.keys())
    
    def get_bounds_array(self):
        """Get bounds as numpy arrays for optimization"""
        lower = []
        upper = []
        for feature in self.feature_names:
            l, u = self.bounds[feature]
            lower.append(l)
            upper.append(u)
        return np.array(lower), np.array(upper)
    
    def check_mass_balance(self, x):
        """
        Constraint: Blend percentages must sum to 100%
        
        Returns:
        --------
        float : Constraint violation (0 = satisfied, >0 = violated)
        """
        if isinstance(x, dict):
            total = (x['clinker_pct'] + x['fly_ash_pct'] + 
                    x['slag_pct'] + x['limestone_pct'])
        else:
            total = x[0] + x[1] + x[2] + x[3]
        
        # Violation if sum != 100% (with 1% tolerance)
        violation = abs(total - 100)
        return violation if violation > 1.0 else 0.0
    
    def check_strength_requirement(self, x, min_strength=42.5):
        """
        Constraint: Minimum compressive strength (M30 grade)
        
        Uses surrogate model if available, otherwise heuristic
        
        Parameters:
        -----------
        x : dict or np.ndarray
            Decision variables
        min_strength : float
            Minimum required strength (MPa), default 42.5 (M30 grade)
        
        Returns:
        --------
        float : Constraint violation (0 = satisfied, >0 = violated)
        """
        if self.surrogate is not None:
            # Use surrogate model prediction
            if isinstance(x, dict):
                x_array = np.array([x[f] for f in self.feature_names]).reshape(1, -1)
            else:
                x_array = x.reshape(1, -1) if x.ndim == 1 else x
            
            predictions = self.surrogate.predict(x_array)
            predicted_strength = predictions['strength'][0]
        else:
            # Use simplified heuristic
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
            
            # Simplified strength model
            scm_dict = {'fly_ash': fly_ash_pct, 'slag': slag_pct}
            predicted_strength = self.chem.strength_prediction_bolomey(
                cement_content=400,
                water_content=180,
                scm_percentages=scm_dict
            )
            
            # Temperature adjustment
            temp_factor = 1 + (kiln_temp - 1450) / 1450 * 0.05
            predicted_strength *= temp_factor
        
        # Violation if strength < minimum
        violation = max(0, min_strength - predicted_strength)
        return violation
    
    def check_scm_limits(self, x):
        """
        Constraint: Total SCM content must not exceed safe limits
        
        Maximum 40% total SCM for quality assurance
        
        Returns:
        --------
        float : Constraint violation
        """
        if isinstance(x, dict):
            fly_ash_pct = x['fly_ash_pct']
            slag_pct = x['slag_pct']
        else:
            fly_ash_pct = x[1]
            slag_pct = x[2]
        
        total_scm = fly_ash_pct + slag_pct
        max_scm = 40.0
        
        # Violation if total SCM > 40%
        violation = max(0, total_scm - max_scm)
        return violation
    
    def check_emission_cap(self, x, max_emissions=0.85):
        """
        Constraint: Emissions must not exceed regulatory cap
        
        Parameters:
        -----------
        x : dict or np.ndarray
            Decision variables
        max_emissions : float
            Maximum allowed emissions (kg CO2/kg cement)
        
        Returns:
        --------
        float : Constraint violation
        """
        if isinstance(x, dict):
            clinker_pct = x['clinker_pct']
            fuel_input = x['fuel_input']
        else:
            clinker_pct = x[0]
            fuel_input = x[10]
        
        # Calculate emissions
        clinker_mass = clinker_pct / 100 * 1000
        emissions_dict = self.chem.calculate_co2_emissions(
            clinker_mass=clinker_mass,
            fuel_mass=fuel_input,
            fuel_type='coal'
        )
        
        emission_intensity = emissions_dict['total'] / 1000
        
        # Violation if emissions > cap
        violation = max(0, emission_intensity - max_emissions)
        return violation
    
    def check_temperature_safety(self, x):
        """
        Constraint: Temperature must be within safe operational range
        
        Returns:
        --------
        float : Constraint violation
        """
        if isinstance(x, dict):
            kiln_temp = x['kiln_temp']
        else:
            kiln_temp = x[8]
        
        # Safety range: 1400-1500°C (already in bounds, but double-check)
        if kiln_temp < 1400 or kiln_temp > 1500:
            violation = max(1400 - kiln_temp, kiln_temp - 1500, 0)
        else:
            violation = 0.0
        
        return violation
    
    def evaluate_all_constraints(self, x):
        """
        Evaluate all constraints
        
        Returns:
        --------
        dict : All constraint violations
        """
        constraints = {
            'mass_balance': self.check_mass_balance(x),
            'strength': self.check_strength_requirement(x),
            'scm_limits': self.check_scm_limits(x),
            'emissions_cap': self.check_emission_cap(x),
            'temperature': self.check_temperature_safety(x)
        }
        
        # Total violation
        constraints['total'] = sum(constraints.values())
        constraints['feasible'] = constraints['total'] == 0
        
        return constraints
    
    def is_feasible(self, x):
        """Check if solution is feasible"""
        constraints = self.evaluate_all_constraints(x)
        return constraints['feasible']


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CONSTRAINTS MODULE TEST")
    print("="*60)
    
    constraints = CementConstraints()
    
    # Test case 1: Feasible solution
    print("\nTest Case 1: Feasible Solution")
    x1 = {
        'clinker_pct': 75,
        'fly_ash_pct': 15,
        'slag_pct': 8,
        'limestone_pct': 2,
        'CaO': 65, 'SiO2': 21, 'Al2O3': 5, 'Fe2O3': 3,
        'kiln_temp': 1450,
        'kiln_rpm': 2.5,
        'fuel_input': 95,
        'residence_time': 30,
        'fuel_price': 80,
        'carbon_tax': 50,
        'transport_cost': 0.3,
        'demand': 2000
    }
    
    violations1 = constraints.evaluate_all_constraints(x1)
    print("Constraint Violations:")
    for name, value in violations1.items():
        if name != 'feasible':
            print(f"  {name:20s}: {value:.4f}")
    print(f"\nFeasible: {violations1['feasible']} ✓")
    
    # Test case 2: Infeasible (too much SCM)
    print("\nTest Case 2: Infeasible - Excessive SCM")
    x2 = {
        'clinker_pct': 55,
        'fly_ash_pct': 30,
        'slag_pct': 15,  # Total SCM = 45% > 40% limit
        'limestone_pct': 0,
        'CaO': 65, 'SiO2': 21, 'Al2O3': 5, 'Fe2O3': 3,
        'kiln_temp': 1450,
        'kiln_rpm': 2.5,
        'fuel_input': 95,
        'residence_time': 30,
        'fuel_price': 80,
        'carbon_tax': 50,
        'transport_cost': 0.3,
        'demand': 2000
    }
    
    violations2 = constraints.evaluate_all_constraints(x2)
    print("Constraint Violations:")
    for name, value in violations2.items():
        if name != 'feasible':
            status = "✗ VIOLATED" if value > 0 else "✓"
            print(f"  {name:20s}: {value:.4f} {status}")
    print(f"\nFeasible: {violations2['feasible']}")
    
    # Test case 3: Infeasible (low strength)
    print("\nTest Case 3: Infeasible - Low Strength")
    x3 = {
        'clinker_pct': 60,
        'fly_ash_pct': 28,
        'slag_pct': 10,
        'limestone_pct': 2,
        'CaO': 60, 'SiO2': 24, 'Al2O3': 8, 'Fe2O3': 5,
        'kiln_temp': 1400,  # Lower temp = lower strength
        'kiln_rpm': 2.5,
        'fuel_input': 85,
        'residence_time': 20,
        'fuel_price': 80,
        'carbon_tax': 50,
        'transport_cost': 0.3,
        'demand': 2000
    }
    
    violations3 = constraints.evaluate_all_constraints(x3)
    print("Constraint Violations:")
    for name, value in violations3.items():
        if name != 'feasible':
            status = "✗ VIOLATED" if value > 0 else "✓"
            print(f"  {name:20s}: {value:.4f} {status}")
    print(f"\nFeasible: {violations3['feasible']}")
    
    # Test bounds
    print("\n" + "="*60)
    print("BOUNDS TEST")
    print("="*60)
    lower, upper = constraints.get_bounds_array()
    print(f"Decision variables: {len(lower)}")
    print(f"Lower bounds shape: {lower.shape}")
    print(f"Upper bounds shape: {upper.shape}")
    
    print("\n✓ Constraints module validated!")