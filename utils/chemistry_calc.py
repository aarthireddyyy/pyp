"""
Chemistry Calculator Module for Cement Manufacturing
Implements Bogue equations, stoichiometry, and mass balance calculations
"""

import numpy as np


class CementChemistry:
    """
    Handles all cement chemistry calculations including:
    - Bogue equations for clinker phase composition
    - Stoichiometric CO2 emission calculations
    - Mass and element balance validation
    - Energy calculations
    """
    
    def __init__(self):
        # Molecular weights (g/mol)
        self.MW = {
            'CaCO3': 100.09,
            'CaO': 56.08,
            'CO2': 44.01,
            'SiO2': 60.08,
            'Al2O3': 101.96,
            'Fe2O3': 159.69,
            'C3S': 228.32,  # Ca3SiO5
            'C2S': 172.24,  # Ca2SiO4
            'C3A': 270.19,  # Ca3Al2O6
            'C4AF': 485.96  # Ca4Al2Fe2O10
        }
        
        # Emission factors (kg CO2 per kg material)
        self.emission_factors = {
            'calcination': 0.525,  # CaCO3 -> CaO + CO2
            'coal': 2.86,          # kg CO2 per kg coal
            'natural_gas': 2.75,   # kg CO2 per kg natural gas
            'electricity': 0.82    # kg CO2 per kWh (grid average)
        }
        
        # Energy content (GJ/ton)
        self.energy_content = {
            'coal': 29.3,
            'natural_gas': 50.0,
            'electricity': 3.6  # per MWh
        }
    
    def bogue_equations(self, oxide_composition):
        """
        Calculate clinker phase composition using modified Bogue equations
        
        Parameters:
        -----------
        oxide_composition : dict
            Oxide percentages: {'CaO': 65, 'SiO2': 21, 'Al2O3': 5, 'Fe2O3': 3}
        
        Returns:
        --------
        dict : Phase composition (C3S, C2S, C3A, C4AF) in percentages
        """
        C = oxide_composition.get('CaO', 0)
        S = oxide_composition.get('SiO2', 0)
        A = oxide_composition.get('Al2O3', 0)
        F = oxide_composition.get('Fe2O3', 0)
        
        # Modified Bogue equations (simplified)
        # C3S (Alite) - main strength component
        C3S = 4.071 * C - 7.600 * S - 6.718 * A - 1.430 * F - 2.852
        C3S = max(0, min(100, C3S))  # Bound between 0-100%
        
        # C2S (Belite)
        C2S = 2.867 * S - 0.7544 * C3S
        C2S = max(0, min(100, C2S))
        
        # C3A (Aluminate)
        C3A = 2.650 * A - 1.692 * F
        C3A = max(0, min(100, C3A))
        
        # C4AF (Ferrite)
        C4AF = 3.043 * F
        C4AF = max(0, min(100, C4AF))
        
        # Normalize to 100% if needed
        total = C3S + C2S + C3A + C4AF
        if total > 0:
            C3S = C3S / total * 100
            C2S = C2S / total * 100
            C3A = C3A / total * 100
            C4AF = C4AF / total * 100
        
        return {
            'C3S': C3S,
            'C2S': C2S,
            'C3A': C3A,
            'C4AF': C4AF
        }
    
    def calculate_co2_emissions(self, clinker_mass, fuel_mass, fuel_type='coal'):
        """
        Calculate total CO2 emissions from cement production
        
        Parameters:
        -----------
        clinker_mass : float
            Mass of clinker produced (kg)
        fuel_mass : float
            Mass of fuel consumed (kg)
        fuel_type : str
            Type of fuel ('coal', 'natural_gas')
        
        Returns:
        --------
        dict : Breakdown of CO2 emissions
        """
        # Process emissions (calcination of CaCO3 -> CaO + CO2)
        # Stoichiometry: 1 mol CaCO3 produces 1 mol CO2
        # CO2/CaCO3 = 44.01/100.09 = 0.44
        calcination_co2 = clinker_mass * self.emission_factors['calcination']
        
        # Fuel combustion emissions
        fuel_co2 = fuel_mass * self.emission_factors.get(fuel_type, 2.86)
        
        # Total emissions
        total_co2 = calcination_co2 + fuel_co2
        
        return {
            'calcination': calcination_co2,
            'fuel': fuel_co2,
            'total': total_co2,
            'intensity': total_co2 / clinker_mass  # kg CO2 per kg clinker
        }
    
    def mass_balance_check(self, inputs, outputs):
        """
        Validate mass balance: inputs = outputs + emissions
        
        Parameters:
        -----------
        inputs : dict
            Input materials {'limestone': x, 'clay': y, ...}
        outputs : dict
            Output materials {'clinker': z, ...}
        
        Returns:
        --------
        dict : Balance status and error
        """
        total_input = sum(inputs.values())
        total_output = sum(outputs.values())
        
        # Account for CO2 loss during calcination (~35% of input mass)
        expected_loss = inputs.get('limestone', 0) * 0.44
        
        balance_error = abs(total_input - total_output - expected_loss)
        relative_error = balance_error / total_input if total_input > 0 else 0
        
        return {
            'balanced': relative_error < 0.05,  # Within 5%
            'error': balance_error,
            'relative_error': relative_error,
            'input_mass': total_input,
            'output_mass': total_output,
            'co2_loss': expected_loss
        }
    
    def element_conservation(self, input_oxides, output_oxides):
        """
        Check conservation of elements (Ca, Si, Al, Fe)
        
        Parameters:
        -----------
        input_oxides : dict
            Input oxide composition
        output_oxides : dict
            Output oxide composition
        
        Returns:
        --------
        dict : Conservation errors for each element
        """
        errors = {}
        
        # Check each major element
        for oxide in ['CaO', 'SiO2', 'Al2O3', 'Fe2O3']:
            input_val = input_oxides.get(oxide, 0)
            output_val = output_oxides.get(oxide, 0)
            error = abs(input_val - output_val)
            errors[oxide] = error
        
        return errors
    
    def energy_balance(self, fuel_mass, fuel_type, clinker_mass, kiln_temp):
        """
        Calculate energy balance for clinker production
        
        Parameters:
        -----------
        fuel_mass : float
            Fuel consumed (kg)
        fuel_type : str
            Fuel type
        clinker_mass : float
            Clinker produced (kg)
        kiln_temp : float
            Kiln temperature (°C)
        
        Returns:
        --------
        dict : Energy metrics
        """
        # Energy input from fuel (GJ)
        energy_in = fuel_mass / 1000 * self.energy_content.get(fuel_type, 29.3)
        
        # Theoretical energy needed for clinkerization (~1.7 GJ/ton at 1450°C)
        theoretical_energy = clinker_mass / 1000 * 1.7
        
        # Temperature adjustment (higher temp = more energy)
        temp_factor = 1 + (kiln_temp - 1450) / 1450 * 0.1
        adjusted_energy = theoretical_energy * temp_factor
        
        # Efficiency
        efficiency = adjusted_energy / energy_in if energy_in > 0 else 0
        
        return {
            'energy_input': energy_in,
            'energy_required': adjusted_energy,
            'efficiency': efficiency,
            'specific_energy': energy_in / clinker_mass * 1000  # GJ/ton
        }
    
    def strength_prediction_bolomey(self, cement_content, water_content, 
                                   scm_percentages=None, age_days=28):
        """
        Predict concrete strength using modified Bolomey-Féret equation
        
        Parameters:
        -----------
        cement_content : float
            Cement content (kg/m³)
        water_content : float
            Water content (kg/m³)
        scm_percentages : dict, optional
            SCM replacement percentages {'fly_ash': 20, 'slag': 0}
        age_days : int
            Concrete age in days
        
        Returns:
        --------
        float : Compressive strength (MPa)
        """
        # Base strength coefficient for OPC
        K = 25.0
        
        # Adjust K for SCMs
        if scm_percentages:
            fa_pct = scm_percentages.get('fly_ash', 0) / 100
            slag_pct = scm_percentages.get('slag', 0) / 100
            
            # SCMs reduce early strength but can increase later strength
            # Simplified model
            K = K * (1 - 0.2 * fa_pct - 0.15 * slag_pct)
        
        # Water-cement ratio
        wc_ratio = water_content / cement_content if cement_content > 0 else 1.0
        
        # Bolomey-Féret equation: f'c = K * (C/W - 0.5)
        if wc_ratio < 0.3:
            wc_ratio = 0.3  # Minimum practical W/C ratio
        
        strength = K * (1/wc_ratio - 0.5)
        
        # Age factor (simplified)
        if age_days < 28:
            age_factor = age_days / 28  # Linear approximation
            strength = strength * age_factor
        
        # Ensure realistic bounds
        strength = max(0, min(100, strength))
        
        return strength


# Example usage and testing
if __name__ == "__main__":
    chem = CementChemistry()
    
    # Test 1: Bogue equations
    print("=" * 50)
    print("TEST 1: Bogue Equations")
    print("=" * 50)
    oxide_comp = {'CaO': 65, 'SiO2': 21, 'Al2O3': 5, 'Fe2O3': 3}
    phases = chem.bogue_equations(oxide_comp)
    print(f"Input oxides: {oxide_comp}")
    print(f"Clinker phases: {phases}")
    print(f"Total: {sum(phases.values()):.1f}%\n")
    
    # Test 2: CO2 emissions
    print("=" * 50)
    print("TEST 2: CO2 Emissions")
    print("=" * 50)
    emissions = chem.calculate_co2_emissions(
        clinker_mass=1000,  # 1 ton
        fuel_mass=100,      # 100 kg coal
        fuel_type='coal'
    )
    print(f"Clinker: 1000 kg, Fuel: 100 kg coal")
    print(f"Calcination CO2: {emissions['calcination']:.1f} kg")
    print(f"Fuel CO2: {emissions['fuel']:.1f} kg")
    print(f"Total CO2: {emissions['total']:.1f} kg")
    print(f"Emission intensity: {emissions['intensity']:.3f} kg CO2/kg clinker\n")
    
    # Test 3: Mass balance
    print("=" * 50)
    print("TEST 3: Mass Balance")
    print("=" * 50)
    inputs = {'limestone': 1400, 'clay': 250, 'sand': 50}
    outputs = {'clinker': 1000}
    balance = chem.mass_balance_check(inputs, outputs)
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")
    print(f"Balanced: {balance['balanced']}")
    print(f"Relative error: {balance['relative_error']*100:.2f}%\n")
    
    # Test 4: Strength prediction
    print("=" * 50)
    print("TEST 4: Strength Prediction")
    print("=" * 50)
    strength = chem.strength_prediction_bolomey(
        cement_content=400,
        water_content=180,
        scm_percentages={'fly_ash': 20, 'slag': 0},
        age_days=28
    )
    print(f"Cement: 400 kg/m³, Water: 180 kg/m³, Fly ash: 20%")
    print(f"Predicted 28-day strength: {strength:.1f} MPa")