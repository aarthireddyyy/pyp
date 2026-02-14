"""
Simplified NSGA-II Multi-Objective Optimization
For cement manufacturing optimization
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimization.objectives import CementObjectives
from optimization.constraints import CementConstraints


class SimplifiedNSGAII:
    """
    Simplified NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    
    For multi-objective optimization with constraints
    """
    
    def __init__(self, objectives, constraints, pop_size=100, n_gen=50):
        """
        Initialize NSGA-II optimizer
        
        Parameters:
        -----------
        objectives : CementObjectives
            Objective functions
        constraints : CementConstraints
            Constraint functions
        pop_size : int
            Population size
        n_gen : int
            Number of generations
        """
        self.objectives = objectives
        self.constraints = constraints
        self.pop_size = pop_size
        self.n_gen = n_gen
        
        # Get bounds
        self.lower, self.upper = constraints.get_bounds_array()
        self.n_vars = len(self.lower)
    
    def initialize_population(self):
        """Create initial random population within bounds"""
        pop = np.random.uniform(
            low=self.lower,
            high=self.upper,
            size=(self.pop_size, self.n_vars)
        )
        
        # Normalize blend ratios to sum to 100%
        for i in range(self.pop_size):
            blend_sum = pop[i, 0] + pop[i, 1] + pop[i, 2] + pop[i, 3]
            pop[i, 0:4] = (pop[i, 0:4] / blend_sum) * 100
        
        return pop
    
    def evaluate_objectives(self, individual):
        """Evaluate all 4 objectives for an individual"""
        cost = self.objectives.calculate_cost(individual)
        emissions = self.objectives.calculate_emissions(individual)
        circularity = self.objectives.calculate_circularity(individual)  # Negative
        risk = self.objectives.calculate_risk(individual)
        
        # Return as array (all minimization)
        return np.array([cost, emissions, circularity, risk])
    
    def evaluate_constraints(self, individual):
        """Check constraint violations"""
        violations = self.constraints.evaluate_all_constraints(individual)
        return violations['total']
    
    def dominates(self, obj1, obj2):
        """
        Check if obj1 dominates obj2
        
        obj1 dominates obj2 if:
        - obj1 is no worse than obj2 in all objectives
        - obj1 is strictly better than obj2 in at least one objective
        """
        no_worse = np.all(obj1 <= obj2)
        strictly_better = np.any(obj1 < obj2)
        return no_worse and strictly_better
    
    def fast_non_dominated_sort(self, objectives_pop):
        """
        Non-dominated sorting (core of NSGA-II)
        
        Returns:
        --------
        list : Fronts (list of lists of indices)
        """
        n = len(objectives_pop)
        domination_counts = np.zeros(n, dtype=int)  # How many dominate this solution
        dominated_solutions = [[] for _ in range(n)]  # Which solutions this dominates
        fronts = [[]]
        
        # Calculate domination
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objectives_pop[i], objectives_pop[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives_pop[j], objectives_pop[i]):
                        domination_counts[i] += 1
            
            # If not dominated by anyone, it's in the first front
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)
        
        # Remove empty last front
        return fronts[:-1]
    
    def crowding_distance(self, objectives_front):
        """Calculate crowding distance for diversity"""
        n = len(objectives_front)
        if n == 0:
            return np.array([])
        
        distances = np.zeros(n)
        n_obj = objectives_front.shape[1]
        
        for m in range(n_obj):
            # Sort by objective m
            sorted_indices = np.argsort(objectives_front[:, m])
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Calculate crowding distance for middle points
            obj_range = objectives_front[sorted_indices[-1], m] - objectives_front[sorted_indices[0], m]
            
            if obj_range > 0:
                for i in range(1, n-1):
                    distances[sorted_indices[i]] += (
                        (objectives_front[sorted_indices[i+1], m] - 
                         objectives_front[sorted_indices[i-1], m]) / obj_range
                    )
        
        return distances
    
    def tournament_selection(self, population, objectives_pop, fronts, crowding_distances):
        """Tournament selection for mating"""
        n = len(population)
        selected = []
        
        for _ in range(self.pop_size):
            # Random tournament
            i, j = np.random.choice(n, 2, replace=False)
            
            # Find fronts
            front_i = next(idx for idx, front in enumerate(fronts) if i in front)
            front_j = next(idx for idx, front in enumerate(fronts) if j in front)
            
            # Select based on front rank and crowding distance
            if front_i < front_j:
                selected.append(population[i])
            elif front_i > front_j:
                selected.append(population[j])
            else:
                # Same front, use crowding distance
                if crowding_distances[i] > crowding_distances[j]:
                    selected.append(population[i])
                else:
                    selected.append(population[j])
        
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        """Simulated binary crossover"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(self.n_vars):
            if np.random.rand() < 0.9:  # Crossover probability
                beta = np.random.rand()
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        
        # Ensure bounds
        child1 = np.clip(child1, self.lower, self.upper)
        child2 = np.clip(child2, self.lower, self.upper)
        
        # Normalize blend ratios
        blend_sum1 = child1[0] + child1[1] + child1[2] + child1[3]
        child1[0:4] = (child1[0:4] / blend_sum1) * 100
        
        blend_sum2 = child2[0] + child2[1] + child2[2] + child2[3]
        child2[0:4] = (child2[0:4] / blend_sum2) * 100
        
        return child1, child2
    
    def mutate(self, individual):
        """Polynomial mutation"""
        mutated = individual.copy()
        
        for i in range(self.n_vars):
            if np.random.rand() < 0.1:  # Mutation probability
                delta = np.random.normal(0, 0.1)
                mutated[i] = individual[i] + delta * (self.upper[i] - self.lower[i])
        
        # Ensure bounds
        mutated = np.clip(mutated, self.lower, self.upper)
        
        # Normalize blend ratios
        blend_sum = mutated[0] + mutated[1] + mutated[2] + mutated[3]
        mutated[0:4] = (mutated[0:4] / blend_sum) * 100
        
        return mutated
    
    def optimize(self):
        """Main optimization loop"""
        print("="*60)
        print("NSGA-II MULTI-OBJECTIVE OPTIMIZATION")
        print("="*60)
        print(f"Population size: {self.pop_size}")
        print(f"Generations: {self.n_gen}")
        print(f"Decision variables: {self.n_vars}")
        print(f"Objectives: 4 (cost, emissions, -circularity, risk)")
        print("="*60)
        
        # Initialize population
        print("\nInitializing population...")
        population = self.initialize_population()
        
        best_hypervolume = 0
        
        for gen in range(self.n_gen):
            # Evaluate objectives
            objectives_pop = np.array([self.evaluate_objectives(ind) for ind in population])
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(objectives_pop)
            
            # Calculate crowding distance for each front
            all_crowding_distances = np.zeros(len(population))
            for front in fronts:
                if len(front) > 0:
                    distances = self.crowding_distance(objectives_pop[front])
                    all_crowding_distances[front] = distances
            
            # Selection
            parents = self.tournament_selection(population, objectives_pop, fronts, all_crowding_distances)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    child1, child2 = self.crossover(parents[i], parents[i+1])
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    offspring.extend([child1, child2])
            
            offspring = np.array(offspring[:self.pop_size])
            
            # Combine population and offspring
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.array([self.evaluate_objectives(ind) for ind in combined_pop])
            
            # Survival selection
            fronts_combined = self.fast_non_dominated_sort(combined_obj)
            
            new_population = []
            new_obj = []
            for front in fronts_combined:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend([combined_pop[i] for i in front])
                    new_obj.extend([combined_obj[i] for i in front])
                else:
                    # Need crowding distance to break tie
                    remaining = self.pop_size - len(new_population)
                    if remaining > 0:
                        distances = self.crowding_distance(combined_obj[front])
                        sorted_indices = np.argsort(-distances)[:remaining]
                        new_population.extend([combined_pop[front[i]] for i in sorted_indices])
                        new_obj.extend([combined_obj[front[i]] for i in sorted_indices])
                    break
            
            population = np.array(new_population)
            objectives_pop = np.array(new_obj)
            
            # Progress report
            if (gen + 1) % 10 == 0 or gen == 0:
                pareto_front_size = len(fronts[0])
                print(f"Gen {gen+1:3d}: Pareto front size = {pareto_front_size}")
        
        # Final evaluation
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        
        # Extract Pareto front
        final_objectives = np.array([self.evaluate_objectives(ind) for ind in population])
        final_fronts = self.fast_non_dominated_sort(final_objectives)
        pareto_indices = final_fronts[0]
        
        pareto_population = population[pareto_indices]
        pareto_objectives = final_objectives[pareto_indices]
        
        print(f"\nPareto front solutions: {len(pareto_population)}")
        
        # Convert circularity back to positive
        pareto_objectives[:, 2] = -pareto_objectives[:, 2]
        
        return pareto_population, pareto_objectives


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("NSGA-II OPTIMIZER TEST")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create objectives and constraints
    objectives = CementObjectives()
    constraints = CementConstraints()
    
    # Create optimizer
    optimizer = SimplifiedNSGAII(
        objectives=objectives,
        constraints=constraints,
        pop_size=50,  # Small for testing
        n_gen=20      # Few generations for testing
    )
    
    # Run optimization
    pareto_pop, pareto_obj = optimizer.optimize()
    
    # Display results
    print("\n" + "="*60)
    print("PARETO FRONT ANALYSIS")
    print("="*60)
    
    print(f"\nObjective Ranges:")
    print(f"  Cost:        ${pareto_obj[:, 0].min():.2f} - ${pareto_obj[:, 0].max():.2f}")
    print(f"  Emissions:   {pareto_obj[:, 1].min():.3f} - {pareto_obj[:, 1].max():.3f} kg CO2/kg")
    print(f"  Circularity: {pareto_obj[:, 2].min():.1f}% - {pareto_obj[:, 2].max():.1f}%")
    print(f"  Risk:        {pareto_obj[:, 3].min():.3f} - {pareto_obj[:, 3].max():.3f}")
    
    # Save results
    print("\nSaving results...")
    os.makedirs('results/optimization', exist_ok=True)
    
    # Save Pareto front
    pareto_df = pd.DataFrame(
        pareto_obj,
        columns=['cost', 'emissions', 'circularity', 'risk']
    )
    pareto_df.to_csv('results/optimization/pareto_front.csv', index=False)
    print("✓ Saved results/optimization/pareto_front.csv")
    
    print("\n✓ Optimization test complete!")