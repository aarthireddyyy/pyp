"""
Visualization Module for Multi-Objective Optimization Results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os

class OptimizationVisualizer:
    """
    Create visualizations for optimization results
    """
    
    def __init__(self, output_dir='results/plots'):
        """Initialize visualizer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
    
    def plot_2d_projections(self, pareto_df, filename='pareto_2d_projections.png'):
        """
        Create 2D projections of Pareto front
        
        Shows all pairwise combinations of objectives
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Pareto Front: 2D Projections', fontsize=14, fontweight='bold')
        
        objectives = ['cost', 'emissions', 'circularity', 'risk']
        labels = {
            'cost': 'Cost ($/ton)',
            'emissions': 'Emissions (kg CO₂/kg)',
            'circularity': 'Circularity (%)',
            'risk': 'Risk (normalized)'
        }
        
        plot_idx = 0
        combinations = [
            ('cost', 'emissions'),
            ('cost', 'circularity'),
            ('cost', 'risk'),
            ('emissions', 'circularity'),
            ('emissions', 'risk'),
            ('circularity', 'risk')
        ]
        
        for i in range(2):
            for j in range(3):
                if plot_idx < len(combinations):
                    obj1, obj2 = combinations[plot_idx]
                    ax = axes[i, j]
                    
                    # Scatter plot
                    scatter = ax.scatter(pareto_df[obj1], pareto_df[obj2],
                                       c=pareto_df['emissions'], cmap='RdYlGn_r',
                                       s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
                    
                    ax.set_xlabel(labels[obj1])
                    ax.set_ylabel(labels[obj2])
                    ax.set_title(f'{obj1.capitalize()} vs {obj2.capitalize()}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add colorbar for first plot
                    if plot_idx == 0:
                        cbar = plt.colorbar(scatter, ax=ax)
                        cbar.set_label('Emissions\n(kg CO₂/kg)', rotation=270, labelpad=20)
                
                plot_idx += 1
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {filepath}")
        
        return filepath
    
    def plot_pareto_front_3d(self, pareto_df, filename='pareto_3d.png'):
        """
        Create 3D visualization of Pareto front (Cost vs Emissions vs Circularity)
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 3D scatter
        scatter = ax.scatter(pareto_df['cost'], 
                           pareto_df['emissions'], 
                           pareto_df['circularity'],
                           c=pareto_df['risk'], 
                           cmap='RdYlGn_r',
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
        
        ax.set_xlabel('Cost ($/ton)', fontsize=11, labelpad=10)
        ax.set_ylabel('Emissions (kg CO₂/kg)', fontsize=11, labelpad=10)
        ax.set_zlabel('Circularity (%)', fontsize=11, labelpad=10)
        ax.set_title('3D Pareto Front: Cost vs Emissions vs Circularity', 
                    fontsize=13, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Risk', rotation=270, labelpad=20)
        
        # Better viewing angle
        ax.view_init(elev=20, azim=45)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {filepath}")
        
        return filepath
    
    def plot_trade_off_analysis(self, pareto_df, filename='trade_off_analysis.png'):
        """
        Analyze trade-offs between objectives
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Trade-off Analysis', fontsize=14, fontweight='bold')
        
        # Plot 1: Cost vs Emissions trade-off
        ax = axes[0, 0]
        ax.scatter(pareto_df['emissions'], pareto_df['cost'], 
                  s=80, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Emissions (kg CO₂/kg)')
        ax.set_ylabel('Cost ($/ton)')
        ax.set_title('Trade-off: Lower Emissions ↔ Higher Cost')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(pareto_df['emissions'], pareto_df['cost'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(pareto_df['emissions'].min(), pareto_df['emissions'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, label=f'Trend: y={z[0]:.1f}x{z[1]:+.1f}')
        ax.legend()
        
        # Plot 2: Circularity vs Emissions
        ax = axes[0, 1]
        ax.scatter(pareto_df['circularity'], pareto_df['emissions'],
                  s=80, alpha=0.6, c='green', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Circularity (%)')
        ax.set_ylabel('Emissions (kg CO₂/kg)')
        ax.set_title('Benefit: Higher Circularity → Lower Emissions')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(pareto_df['circularity'], pareto_df['emissions'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(pareto_df['circularity'].min(), pareto_df['circularity'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, label=f'Trend: y={z[0]:.3f}x{z[1]:+.2f}')
        ax.legend()
        
        # Plot 3: Cost distribution
        ax = axes[1, 0]
        ax.hist(pareto_df['cost'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(pareto_df['cost'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: ${pareto_df["cost"].mean():.2f}')
        ax.set_xlabel('Cost ($/ton)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Costs in Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Multi-objective radar chart sample
        ax = axes[1, 1]
        
        # Select 3 representative solutions: min cost, min emissions, max circularity
        idx_min_cost = pareto_df['cost'].idxmin()
        idx_min_emissions = pareto_df['emissions'].idxmin()
        idx_max_circularity = pareto_df['circularity'].idxmax()
        
        # Normalize objectives for comparison (0-1 scale)
        norm_df = pareto_df.copy()
        for col in ['cost', 'emissions', 'risk']:
            norm_df[col] = (pareto_df[col] - pareto_df[col].min()) / (pareto_df[col].max() - pareto_df[col].min())
        norm_df['circularity'] = (pareto_df['circularity'] - pareto_df['circularity'].min()) / (pareto_df['circularity'].max() - pareto_df['circularity'].min())
        
        categories = ['Cost\n(normalized)', 'Emissions\n(normalized)', 'Circularity\n(normalized)', 'Risk\n(normalized)']
        values_cost = norm_df.loc[idx_min_cost, ['cost', 'emissions', 'circularity', 'risk']].values
        values_emissions = norm_df.loc[idx_min_emissions, ['cost', 'emissions', 'circularity', 'risk']].values
        values_circularity = norm_df.loc[idx_max_circularity, ['cost', 'emissions', 'circularity', 'risk']].values
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(x - width, values_cost, width, label='Min Cost Solution', alpha=0.8)
        ax.bar(x, values_emissions, width, label='Min Emissions Solution', alpha=0.8)
        ax.bar(x + width, values_circularity, width, label='Max Circularity Solution', alpha=0.8)
        
        ax.set_ylabel('Normalized Value (0-1)')
        ax.set_title('Comparison of Representative Solutions')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {filepath}")
        
        return filepath
    
    def plot_objective_statistics(self, pareto_df, filename='objective_statistics.png'):
        """
        Statistical summary of objectives
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Objective Statistics in Pareto Front', fontsize=14, fontweight='bold')
        
        objectives = ['cost', 'emissions', 'circularity', 'risk']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (obj, color) in enumerate(zip(objectives, colors)):
            ax = axes[idx // 2, idx % 2]
            
            # Box plot with scatter
            bp = ax.boxplot([pareto_df[obj]], positions=[1], widths=0.6,
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
            
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add scatter overlay
            y = pareto_df[obj]
            x = np.random.normal(1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=30, color=color)
            
            # Statistics text
            stats_text = f"Mean: {pareto_df[obj].mean():.3f}\n"
            stats_text += f"Std: {pareto_df[obj].std():.3f}\n"
            stats_text += f"Min: {pareto_df[obj].min():.3f}\n"
            stats_text += f"Max: {pareto_df[obj].max():.3f}"
            
            ax.text(1.5, pareto_df[obj].mean(), stats_text,
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
            
            # Labels
            units = {
                'cost': '$/ton',
                'emissions': 'kg CO₂/kg',
                'circularity': '%',
                'risk': 'normalized'
            }
            ax.set_ylabel(f'{obj.capitalize()} ({units[obj]})')
            ax.set_title(f'{obj.capitalize()} Distribution')
            ax.set_xlim(0.5, 2)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {filepath}")
        
        return filepath
    
    def create_summary_report(self, pareto_df, filename='optimization_summary.txt'):
        """
        Create text summary of optimization results
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("MULTI-OBJECTIVE OPTIMIZATION SUMMARY\n")
            f.write("AI-Driven Cement Manufacturing Optimization\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Pareto Front Size: {len(pareto_df)} solutions\n\n")
            
            f.write("OBJECTIVE RANGES:\n")
            f.write("-"*60 + "\n")
            for obj in ['cost', 'emissions', 'circularity', 'risk']:
                f.write(f"{obj.capitalize():15s}: ")
                f.write(f"[{pareto_df[obj].min():.3f}, {pareto_df[obj].max():.3f}] ")
                f.write(f"(mean: {pareto_df[obj].mean():.3f}, std: {pareto_df[obj].std():.3f})\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("REPRESENTATIVE SOLUTIONS\n")
            f.write("="*60 + "\n\n")
            
            # Minimum cost solution
            idx = pareto_df['cost'].idxmin()
            f.write("1. MINIMUM COST SOLUTION:\n")
            f.write(f"   Cost:        ${pareto_df.loc[idx, 'cost']:.2f}/ton\n")
            f.write(f"   Emissions:   {pareto_df.loc[idx, 'emissions']:.3f} kg CO₂/kg\n")
            f.write(f"   Circularity: {pareto_df.loc[idx, 'circularity']:.1f}%\n")
            f.write(f"   Risk:        {pareto_df.loc[idx, 'risk']:.3f}\n\n")
            
            # Minimum emissions solution
            idx = pareto_df['emissions'].idxmin()
            f.write("2. MINIMUM EMISSIONS SOLUTION:\n")
            f.write(f"   Cost:        ${pareto_df.loc[idx, 'cost']:.2f}/ton\n")
            f.write(f"   Emissions:   {pareto_df.loc[idx, 'emissions']:.3f} kg CO₂/kg\n")
            f.write(f"   Circularity: {pareto_df.loc[idx, 'circularity']:.1f}%\n")
            f.write(f"   Risk:        {pareto_df.loc[idx, 'risk']:.3f}\n\n")
            
            # Maximum circularity solution
            idx = pareto_df['circularity'].idxmax()
            f.write("3. MAXIMUM CIRCULARITY SOLUTION:\n")
            f.write(f"   Cost:        ${pareto_df.loc[idx, 'cost']:.2f}/ton\n")
            f.write(f"   Emissions:   {pareto_df.loc[idx, 'emissions']:.3f} kg CO₂/kg\n")
            f.write(f"   Circularity: {pareto_df.loc[idx, 'circularity']:.1f}%\n")
            f.write(f"   Risk:        {pareto_df.loc[idx, 'risk']:.3f}\n\n")
            
            # Balanced solution (closest to mean of all objectives)
            norm_df = pareto_df.copy()
            for col in pareto_df.columns:
                norm_df[col] = (pareto_df[col] - pareto_df[col].min()) / (pareto_df[col].max() - pareto_df[col].min())
            
            distances = np.sqrt(((norm_df - 0.5)**2).sum(axis=1))
            idx = distances.idxmin()
            
            f.write("4. BALANCED SOLUTION (closest to center):\n")
            f.write(f"   Cost:        ${pareto_df.loc[idx, 'cost']:.2f}/ton\n")
            f.write(f"   Emissions:   {pareto_df.loc[idx, 'emissions']:.3f} kg CO₂/kg\n")
            f.write(f"   Circularity: {pareto_df.loc[idx, 'circularity']:.1f}%\n")
            f.write(f"   Risk:        {pareto_df.loc[idx, 'risk']:.3f}\n\n")
            
            f.write("="*60 + "\n")
        
        print(f"✓ Saved {filepath}")
        return filepath


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("VISUALIZATION MODULE TEST")
    print("="*60)
    
    # Load Pareto front data
    pareto_df = pd.read_csv('../optimization/results/optimization/pareto_front.csv')
    print(f"\nLoaded {len(pareto_df)} Pareto-optimal solutions")
    
    # Create visualizer
    viz = OptimizationVisualizer(output_dir='results/plots')
    
    # Generate all plots
    print("\nGenerating visualizations...")
    viz.plot_2d_projections(pareto_df)
    viz.plot_pareto_front_3d(pareto_df)
    viz.plot_trade_off_analysis(pareto_df)
    viz.plot_objective_statistics(pareto_df)
    viz.create_summary_report(pareto_df)
    
    print("\n✓ All visualizations created!")