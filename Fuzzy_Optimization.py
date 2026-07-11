"""
Fuzzy Optimization Module for Dynamic IO Model
===============================================
This module performs fuzzy linear programming to optimize intermediate demand
within a ±5% flexibility band, minimizing sectoral damages caused by climate change.
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# ============================================================================
# CONSTANTS
# ============================================================================

NS = 93                     # Number of sectors
N_YEARS = 40                # Simulation horizon
FLEXIBILITY_BAND = 0.05     # ±5% adjustment range for intermediate demand

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_io_table(file_path: str) -> np.ndarray:
    """Load input-output table from Excel file."""
    io = pd.read_excel(file_path, sheet_name='Sheet1', index_col=0)
    return np.array(io)


def extract_intermediate_consumption(io: np.ndarray) -> np.ndarray:
    """Extract total intermediate consumption (column 93)."""
    return io[0:NS, 93]


def compute_damage_matrix(inter_bau: np.ndarray, inter_ssp: np.ndarray) -> np.ndarray:
    """
    Compute damage matrix as ratio of BAU to SSP intermediate consumption.
    This represents the relative vulnerability of each sector.
    """
    damage = inter_bau / inter_ssp
    return damage.reshape(-1, NS)


# ============================================================================
# OPTIMIZATION FUNCTION
# ============================================================================

def optimize_intermediate_demand(damage: np.ndarray, 
                                 z_sum: np.ndarray,
                                 total_outlay_ref: float,
                                 band: float = FLEXIBILITY_BAND) -> Dict:
    """
    Solve fuzzy linear programming problem.
    
    Minimize:     C * Z_Sum_New
    Subject to:   sum(Z_Sum_New) <= sum(Z_Sum)
                  (1 - band) * Z_Sum <= Z_Sum_New <= (1 + band) * Z_Sum
    
    Parameters:
    -----------
    damage : np.ndarray (1, NS)
        Damage coefficients for each sector
    z_sum : np.ndarray (NS,)
        Original intermediate consumption by sector
    total_outlay_ref : float
        Reference total outlay for constraint
    band : float
        Flexibility band (±%)
    
    Returns:
    --------
    dict with keys: 'Z_new', 'total_outlay_new', 'damage_reduction', 'objective'
    """
    
    # Handle invalid values
    damage = np.nan_to_num(damage, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Constraint: sum(Z_Sum_New) <= sum(Z_Sum)
    A_ub = -np.ones((1, NS))
    b_ub = -np.sum(z_sum)
    
    # Bounds: ±band flexibility
    mins = z_sum * (1 - band)
    maxs = z_sum * (1 + band)
    bounds = list(zip(mins, maxs))
    
    # Solve linear program
    result = linprog(
        damage.reshape(-1),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs'
    )
    
    if not result.success:
        print(f"Optimization warning: {result.message}")
        # Fallback to original values
        z_sum_new = z_sum.copy()
    else:
        z_sum_new = result.x
    
    # Calculate new outlays using technical coefficient ratios
    # (Assuming CC matrix handles sectoral allocation)
    total_outlay_new = np.sum(z_sum_new)
    
    damage_reduction = np.sum(z_sum) - total_outlay_new
    
    return {
        'Z_new': z_sum_new,
        'total_outlay_new': total_outlay_new,
        'damage_reduction': damage_reduction,
        'objective': result.fun if result.success else 0.0,
        'success': result.success
    }


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_optimization(n_years: int = N_YEARS, band: float = FLEXIBILITY_BAND) -> Dict:
    """
    Run fuzzy optimization for all years and store results.
    
    Parameters:
    -----------
    n_years : int
        Number of years to simulate
    band : float
        Flexibility band for intermediate demand
    
    Returns:
    --------
    dict containing all outputs and damage results
    """
    
    # Containers for data
    data_bau = {}
    data_ssp = {}
    damage_holder = {}
    
    # Step 1: Load all data and compute damage matrices
    for year in range(1, n_years + 1):
        bau_file = f'Output{year}.xlsx'
        ssp_file = f'Output{year}OPTM.xlsx'
        
        io_bau = load_io_table(bau_file)
        io_ssp = load_io_table(ssp_file)
        
        inter_bau = extract_intermediate_consumption(io_bau)
        inter_ssp = extract_intermediate_consumption(io_ssp)
        
        damage = compute_damage_matrix(inter_bau, inter_ssp)
        
        data_bau[f'Year_{year}'] = io_bau
        data_ssp[f'Year_{year}'] = io_ssp
        damage_holder[f'Year_{year}'] = damage
    
    # Step 2: Apply optimization for each year
    damage_reduction = {}
    total_output_original = {}
    total_output_optimized = {}
    total_output_bau = {}
    
    for year in range(1, n_years + 1):
        io_ssp = data_ssp[f'Year_{year}']
        io_bau = data_bau[f'Year_{year}']
        damage = damage_holder[f'Year_{year}']
        
        # Extract components
        Z = io_ssp[0:NS, 0:NS]
        Z_sum = io_ssp[0:NS, 93]
        ZPT = io_ssp[NS:NS+3, :]
        ZS = io_ssp[NS+3:NS+12, 0:NS]
        ZE = io_ssp[0:NS, NS+1:NS+8]
        
        total_output_bau_year = io_bau[0:NS, NS+9]
        total_output_ssp = io_ssp[0:NS, NS+9].reshape(NS, 1)
        total_outlays = io_ssp[NS+13, 0:NS].reshape(1, NS)
        
        # Compute technical coefficients
        Z_sum_original = Z_sum.copy()
        
        # A_plus: ratio of Z to Z_sum
        A_plus = Z / Z_sum.reshape(1, -1)
        A_plus = np.nan_to_num(A_plus, nan=0.0, posinf=0.0, neginf=0.0)
        
        # CC: ratio of total output to Z_sum
        CC = (total_output_ssp / Z_sum.reshape(NS, 1))
        CC = np.nan_to_num(CC, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- Optimization ---
        result = optimize_intermediate_demand(
            damage=damage,
            z_sum=Z_sum,
            total_outlay_ref=total_outlays.sum(),
            band=band
        )
        
        Z_sum_new = result['Z_new']
        
        # Reconstruct new intermediate consumption matrix
        Z_new = Z_sum_new.reshape(-1, 1) * A_plus
        
        # Reconstruct new outlays
        total_outlays_new = CC * Z_sum_new.reshape(NS, 1)
        total_outlays_new = np.nan_to_num(
            total_outlays_new, nan=0.0, posinf=0.0, neginf=0.0
        )
        
        # Aggregate results
        total_x_new = np.sum(total_outlays_new, axis=0)
        total_x_original = np.sum(total_output_ssp, axis=0)
        damage_reduction_year = total_x_original - total_x_new
        
        # Store results
        damage_reduction[f'Year_{year}'] = damage_reduction_year
        total_output_original[f'Year_{year}'] = total_x_original
        total_output_optimized[f'Year_{year}'] = total_x_new
        total_output_bau[f'Year_{year}'] = np.sum(total_output_bau_year, axis=0)
    
    return {
        'damage_reduction': damage_reduction,
        'total_output_original': total_output_original,
        'total_output_optimized': total_output_optimized,
        'total_output_bau': total_output_bau,
        'data_bau': data_bau,
        'data_ssp': data_ssp
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_results(results: Dict, scenario_name: str = "SSP5-DS"):
    """
    Generate plots for total output and damage reduction.
    """
    
    n_years = len(results['total_output_original'])
    
    # Extract data for plotting
    total_x_bau = []
    total_x_orig = []
    total_x_opt = []
    damage_red = []
    
    for year in range(1, n_years + 1):
        total_x_bau.append(results['total_output_bau'][f'Year_{year}'])
        total_x_orig.append(results['total_output_original'][f'Year_{year}'])
        total_x_opt.append(results['total_output_optimized'][f'Year_{year}'])
        damage_red.append(results['damage_reduction'][f'Year_{year}'])
    
    years = np.arange(1, n_years + 1)
    
    # Figure 1: Total Output Comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(years, total_x_bau, color='#000000', label='BAU', linewidth=2)
    ax1.plot(years, total_x_orig, color='#ff0000', label=f'{scenario_name}', linewidth=2)
    ax1.plot(years, total_x_opt, color='#3e6aff', label=f'{scenario_name} - Optimized', linewidth=2)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Output')
    ax1.set_title(f'Total Output Comparison: {scenario_name}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'total_output_{scenario_name}.png', dpi=300)
    plt.show()
    
    # Figure 2: Damage Reduction
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(years, damage_red, color='#ff0000', label='Damage Reduction', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Damage Reduction')
    ax2.set_title(f'Damage Reduction: {scenario_name}')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'damage_reduction_{scenario_name}.png', dpi=300)
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run optimization
    results = run_optimization(n_years=40, band=0.05)
    
    # Plot results
    plot_results(results, scenario_name="SSP5-DS")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    
    total_damage_reduction = sum(results['damage_reduction'][f'Year_{y}'] for y in range(1, 41))
    avg_damage_reduction = total_damage_reduction / 40
    
    print(f"Total Damage Reduction over 40 years: {total_damage_reduction:,.2f}")
    print(f"Average Annual Damage Reduction: {avg_damage_reduction:,.2f}")
    print("="*60)