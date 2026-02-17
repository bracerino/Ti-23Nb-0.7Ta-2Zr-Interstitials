#!/usr/bin/env python3
"""
MACE Calculation Script (Local POSCAR Files)
Generated on: 2025-09-30 15:39:05
Calculation Type: Geometry Optimization
Model: MACE-MATPES-PBE-0 (medium) - No +U
Device: cuda
Precision: float32

This script reads POSCAR files from the current directory.
Place your POSCAR files in the same directory as this script before running.
"""

import os
import time
import numpy as np
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
from copy import deepcopy
import threading
import queue
import zipfile
import io

# Set threading before other imports
os.environ['OMP_NUM_THREADS'] = '16'

import torch
torch.set_num_threads(16)

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.constraints import FixAtoms, ExpCellFilter, UnitCellFilter

# PyMatGen imports
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# MACE imports
try:
    from mace.calculators import mace_mp, mace_off
    MACE_AVAILABLE = True
except ImportError:
    try:
        from mace.calculators import MACECalculator
        MACE_AVAILABLE = True
    except ImportError:
        MACE_AVAILABLE = False

# CHGNet imports
try:
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    CHGNET_AVAILABLE = True
except ImportError:
    CHGNET_AVAILABLE = False

# SevenNet imports (requires torch 2.6 compatibility)
try:
    torch.serialization.add_safe_globals([slice])  # Required for torch 2.6
    from sevenn.calculator import SevenNetCalculator
    SEVENNET_AVAILABLE = True
except ImportError:
    SEVENNET_AVAILABLE = False

# MatterSim imports
try:
    from mattersim.forcefield import MatterSimCalculator
    MATTERSIM_AVAILABLE = True
except ImportError:
    MATTERSIM_AVAILABLE = False

# ORB imports
try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False

# Nequix imports
try:
    from nequix.calculator import NequixCalculator
    NEQUIX_AVAILABLE = True
except ImportError:
    NEQUIX_AVAILABLE = False

# Check if any calculator is available
if not (MACE_AVAILABLE or CHGNET_AVAILABLE or SEVENNET_AVAILABLE or MATTERSIM_AVAILABLE or ORB_AVAILABLE or NEQUIX_AVAILABLE):
    print("❌ No MLIP calculators available!")
    print("Please install at least one:")
    print("  - MACE: pip install mace-torch")
    print("  - CHGNet: pip install chgnet") 
    print("  - SevenNet: pip install sevenn")
    print("  - MatterSim: pip install mattersim")
    print("  - ORB: pip install orb-models")
    print("  - Nequix: pip install nequix")
    exit(1)
else:
    available_models = []
    if MACE_AVAILABLE:
        available_models.append("MACE")
    if CHGNET_AVAILABLE:
        available_models.append("CHGNet")
    if SEVENNET_AVAILABLE:
        available_models.append("SevenNet")
    if MATTERSIM_AVAILABLE:
        available_models.append("MatterSim")
    if ORB_AVAILABLE:
        available_models.append("ORB")
    if NEQUIX_AVAILABLE:
        available_models.append("Nequix")
    print(f"✅ Available MLIP models: {', '.join(available_models)}")



def wrap_positions_in_cell(atoms):
    wrapped_atoms = atoms.copy()
    fractional_coords = wrapped_atoms.get_scaled_positions()
    wrapped_fractional = fractional_coords % 1.0
    wrapped_atoms.set_scaled_positions(wrapped_fractional)
    return wrapped_atoms


def get_lattice_parameters(atoms):
    cell = atoms.get_cell()
    a, b, c = np.linalg.norm(cell, axis=1)
    
    def angle_between_vectors(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    alpha = angle_between_vectors(cell[1], cell[2])
    beta = angle_between_vectors(cell[0], cell[2])
    gamma = angle_between_vectors(cell[0], cell[1])
    
    volume = np.abs(np.linalg.det(cell))
    
    return {
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'volume': volume
    }


def get_atomic_composition(atoms):
    symbols = atoms.get_chemical_symbols()
    total_atoms = len(symbols)
    
    composition = {}
    for symbol in symbols:
        composition[symbol] = composition.get(symbol, 0) + 1
    
    concentrations = {}
    for element, count in composition.items():
        concentrations[element] = (count / total_atoms) * 100
    
    return composition, concentrations



def save_elastic_constants_to_csv(structure_name, elastic_tensor, csv_filename="results/elastic_constants_cij.csv"):

    
    elastic_data = {
        'structure_name': structure_name,
        'C11_GPa': float(elastic_tensor[0, 0]),
        'C12_GPa': float(elastic_tensor[0, 1]),
        'C13_GPa': float(elastic_tensor[0, 2]),
        'C14_GPa': float(elastic_tensor[0, 3]),
        'C15_GPa': float(elastic_tensor[0, 4]),
        'C16_GPa': float(elastic_tensor[0, 5]),
        'C21_GPa': float(elastic_tensor[1, 0]),
        'C22_GPa': float(elastic_tensor[1, 1]),
        'C23_GPa': float(elastic_tensor[1, 2]),
        'C24_GPa': float(elastic_tensor[1, 3]),
        'C25_GPa': float(elastic_tensor[1, 4]),
        'C26_GPa': float(elastic_tensor[1, 5]),
        'C31_GPa': float(elastic_tensor[2, 0]),
        'C32_GPa': float(elastic_tensor[2, 1]),
        'C33_GPa': float(elastic_tensor[2, 2]),
        'C34_GPa': float(elastic_tensor[2, 3]),
        'C35_GPa': float(elastic_tensor[2, 4]),
        'C36_GPa': float(elastic_tensor[2, 5]),
        'C41_GPa': float(elastic_tensor[3, 0]),
        'C42_GPa': float(elastic_tensor[3, 1]),
        'C43_GPa': float(elastic_tensor[3, 2]),
        'C44_GPa': float(elastic_tensor[3, 3]),
        'C45_GPa': float(elastic_tensor[3, 4]),
        'C46_GPa': float(elastic_tensor[3, 5]),
        'C51_GPa': float(elastic_tensor[4, 0]),
        'C52_GPa': float(elastic_tensor[4, 1]),
        'C53_GPa': float(elastic_tensor[4, 2]),
        'C54_GPa': float(elastic_tensor[4, 3]),
        'C55_GPa': float(elastic_tensor[4, 4]),
        'C56_GPa': float(elastic_tensor[4, 5]),
        'C61_GPa': float(elastic_tensor[5, 0]),
        'C62_GPa': float(elastic_tensor[5, 1]),
        'C63_GPa': float(elastic_tensor[5, 2]),
        'C64_GPa': float(elastic_tensor[5, 3]),
        'C65_GPa': float(elastic_tensor[5, 4]),
        'C66_GPa': float(elastic_tensor[5, 5])
    }

    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
        
        if structure_name in df_existing['structure_name'].values:
            df_existing.loc[df_existing['structure_name'] == structure_name, list(elastic_data.keys())] = list(elastic_data.values())
        else:
            df_new_row = pd.DataFrame([elastic_data])
            df_existing = pd.concat([df_existing, df_new_row], ignore_index=True)
        
        df_existing.to_csv(csv_filename, index=False)
    else:
        df_new = pd.DataFrame([elastic_data])
        df_new.to_csv(csv_filename, index=False)
    
    print(f"  💾 Elastic constants saved to {csv_filename}")


def append_optimization_summary(filename, structure_name, initial_atoms, final_atoms, 
                               initial_energy, final_energy, convergence_status, steps, selective_dynamics=None):
    
    initial_lattice = get_lattice_parameters(initial_atoms)
    final_lattice = get_lattice_parameters(final_atoms)
    composition, concentrations = get_atomic_composition(final_atoms)
    
    energy_change = final_energy - initial_energy
    volume_change = ((final_lattice['volume'] - initial_lattice['volume']) / initial_lattice['volume']) * 100
    
    a_change = ((final_lattice['a'] - initial_lattice['a']) / initial_lattice['a']) * 100
    b_change = ((final_lattice['b'] - initial_lattice['b']) / initial_lattice['b']) * 100
    c_change = ((final_lattice['c'] - initial_lattice['c']) / initial_lattice['c']) * 100
    
    alpha_change = final_lattice['alpha'] - initial_lattice['alpha']
    beta_change = final_lattice['beta'] - initial_lattice['beta']
    gamma_change = final_lattice['gamma'] - initial_lattice['gamma']
    
    comp_formula = "".join([f"{element}{composition[element]}" for element in sorted(composition.keys())])
    
    elements = sorted(composition.keys())
    conc_values = [concentrations[element] for element in elements]
    conc_string = " ".join([f"{element}:{conc:.1f}" for element, conc in zip(elements, conc_values)])
    
    constraint_info = "None"
    if selective_dynamics is not None:
        total_atoms = len(selective_dynamics)
        completely_fixed = sum(1 for flags in selective_dynamics if not any(flags))
        partially_fixed = sum(1 for flags in selective_dynamics if not all(flags) and any(flags))
        free_atoms = sum(1 for flags in selective_dynamics if all(flags))
        
        constraint_parts = []
        if completely_fixed > 0:
            constraint_parts.append(f"{completely_fixed}complete")
        if partially_fixed > 0:
            constraint_parts.append(f"{partially_fixed}partial")
        if free_atoms > 0:
            constraint_parts.append(f"{free_atoms}free")
        
        constraint_info = ",".join(constraint_parts)
    
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a') as f:
        if not file_exists:
            header = "Structure,Formula,Atoms,Composition,Steps,Convergence,E_initial_eV,E_final_eV,E_change_eV,E_per_atom_eV,a_init_A,b_init_A,c_init_A,alpha_init_deg,beta_init_deg,gamma_init_deg,V_init_A3,a_final_A,b_final_A,c_final_A,alpha_final_deg,beta_final_deg,gamma_final_deg,V_final_A3,a_change_percent,b_change_percent,c_change_percent,alpha_change_deg,beta_change_deg,gamma_change_deg,V_change_percent"
            f.write(header + "\n")
        
        line = f"{structure_name},{comp_formula},{len(final_atoms)},{conc_string},{steps},{convergence_status},{initial_energy:.6f},{final_energy:.6f},{energy_change:.6f},{final_energy/len(final_atoms):.6f},{initial_lattice['a']:.6f},{initial_lattice['b']:.6f},{initial_lattice['c']:.6f},{initial_lattice['alpha']:.3f},{initial_lattice['beta']:.3f},{initial_lattice['gamma']:.3f},{initial_lattice['volume']:.6f},{final_lattice['a']:.6f},{final_lattice['b']:.6f},{final_lattice['c']:.6f},{final_lattice['alpha']:.3f},{final_lattice['beta']:.3f},{final_lattice['gamma']:.3f},{final_lattice['volume']:.6f},{a_change:.3f},{b_change:.3f},{c_change:.3f},{alpha_change:.3f},{beta_change:.3f},{gamma_change:.3f},{volume_change:.3f}"
        f.write(line + "\n")


def read_poscar_with_selective_dynamics(filename):
    atoms = read(filename)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    selective_dynamics = None
    if len(lines) > 7:
        line_7 = lines[7].strip().upper()
        if line_7.startswith('S'):
            selective_dynamics = []
            coord_start = 9
            
            for i in range(coord_start, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        flags = [parts[j].upper() == 'T' for j in [3, 4, 5]]
                        selective_dynamics.append(flags)
                    except (IndexError, ValueError):
                        break
                elif len(parts) == 3:
                    break
    
    return atoms, selective_dynamics


def write_poscar_with_selective_dynamics(atoms, filename, selective_dynamics=None, comment="Optimized structure"):
    if selective_dynamics is not None and len(selective_dynamics) == len(atoms):
        with open(filename, 'w') as f:
            f.write(f"{comment}\n")
            f.write("1.0\n")
            
            cell = atoms.get_cell()
            for i in range(3):
                f.write(f"  {cell[i][0]:16.12f}  {cell[i][1]:16.12f}  {cell[i][2]:16.12f}\n")
            
            symbols = atoms.get_chemical_symbols()
            unique_symbols = []
            symbol_counts = []
            for symbol in symbols:
                if symbol not in unique_symbols:
                    unique_symbols.append(symbol)
                    symbol_counts.append(symbols.count(symbol))
            
            f.write("  " + "  ".join(unique_symbols) + "\n")
            f.write("  " + "  ".join(map(str, symbol_counts)) + "\n")
            
            f.write("Selective dynamics\n")
            f.write("Direct\n")
            
            scaled_positions = atoms.get_scaled_positions()
            for symbol in unique_symbols:
                for i, atom_symbol in enumerate(symbols):
                    if atom_symbol == symbol:
                        pos = scaled_positions[i]
                        flags = selective_dynamics[i]
                        flag_str = "  ".join(["T" if flag else "F" for flag in flags])
                        f.write(f"  {pos[0]:16.12f}  {pos[1]:16.12f}  {pos[2]:16.12f}   {flag_str}\n")
    else:
        write(filename, atoms, format='vasp', direct=True, sort=True)
        with open(filename, 'r') as f:
            lines = f.readlines()
        with open(filename, 'w') as f:
            f.write(f"{comment}\n")
            for line in lines[1:]:
                f.write(line)

                
def apply_selective_dynamics_constraints(atoms, selective_dynamics):
    """Apply selective dynamics as ASE constraints with support for partial fixing."""
    if selective_dynamics is None or len(selective_dynamics) != len(atoms):
        return atoms
    
    has_constraints = False
    for flags in selective_dynamics:
        if not all(flags): 
            has_constraints = True
            break
    
    if not has_constraints:
        print(f"  🔄 Selective dynamics found but all atoms are completely free")
        return atoms
    
    try:
        from ase.constraints import FixCartesian, FixAtoms
        
        constraints = []
        constraint_summary = []
        
        completely_fixed_indices = []
        partial_constraints = []
        
        for i, flags in enumerate(selective_dynamics):
            if not any(flags): 
                completely_fixed_indices.append(i)
            elif not all(flags):  # Some directions fixed (partial)
                mask = [not flag for flag in flags] 
                partial_constraints.append((i, mask))
        
        if completely_fixed_indices:
            constraints.append(FixAtoms(indices=completely_fixed_indices))
            constraint_summary.append(f"{len(completely_fixed_indices)} atoms completely fixed")
        
        if partial_constraints:
            partial_groups = {}
            for atom_idx, mask in partial_constraints:
                mask_key = tuple(mask)
                if mask_key not in partial_groups:
                    partial_groups[mask_key] = []
                partial_groups[mask_key].append(atom_idx)
            
            for mask, atom_indices in partial_groups.items():
                for atom_idx in atom_indices:
                    constraints.append(FixCartesian(atom_idx, mask))
                
                fixed_dirs = [dir_name for dir_name, is_fixed in zip(['x', 'y', 'z'], mask) if is_fixed]
                constraint_summary.append(f"{len(atom_indices)} atoms fixed in {','.join(fixed_dirs)} directions")
        
        if constraints:
            atoms.set_constraint(constraints)
            
            total_constrained = len(completely_fixed_indices) + len(partial_constraints)
            print(f"  📌 Applied selective dynamics to {total_constrained}/{len(atoms)} atoms:")
            for summary in constraint_summary:
                print(f"    - {summary}")
        
    except ImportError:
        print(f"  ⚠️ FixCartesian not available, only applying complete atom fixing")
        fixed_indices = []
        for i, flags in enumerate(selective_dynamics):
            if not any(flags):
                fixed_indices.append(i)
        
        if fixed_indices:
            from ase.constraints import FixAtoms
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)
            print(f"  📌 Applied complete fixing to {len(fixed_indices)}/{len(atoms)} atoms")
        else:
            print(f"  ⚠️ No completely fixed atoms found, partial constraints not supported")
    
    except Exception as e:
        print(f"  ⚠️ FixCartesian failed ({str(e)}), falling back to complete atom fixing only")
        fixed_indices = []
        for i, flags in enumerate(selective_dynamics):
            if not any(flags): 
                fixed_indices.append(i)
        
        if fixed_indices:
            from ase.constraints import FixAtoms
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)
            print(f"  📌 Applied complete fixing to {len(fixed_indices)}/{len(atoms)} atoms (fallback)")
        else:
            print(f"  ⚠️ No completely fixed atoms found")
    
    return atoms



def generate_concentration_combinations(substitutions):
    """Generate all possible combinations of concentrations."""
    import itertools

    has_multiple = any('concentration_list' in sub_info and len(sub_info['concentration_list']) > 1
                       for sub_info in substitutions.values())

    if not has_multiple:
        single_combo = {}
        for element, sub_info in substitutions.items():
            if 'concentration_list' in sub_info:
                concentration = sub_info['concentration_list'][0]
            else:
                concentration = sub_info.get('concentration', 0.5)

            element_count = sub_info.get('element_count', 0)
            n_substitute = int(element_count * concentration)

            single_combo[element] = {
                'new_element': sub_info['new_element'],
                'concentration': concentration,
                'n_substitute': n_substitute,
                'n_remaining': element_count - n_substitute
            }
        return [single_combo]

    elements = []
    concentration_lists = []

    for element, sub_info in substitutions.items():
        elements.append(element)
        if 'concentration_list' in sub_info:
            concentration_lists.append(sub_info['concentration_list'])
        else:
            concentration_lists.append([sub_info.get('concentration', 0.5)])

    combinations = []
    for conc_combo in itertools.product(*concentration_lists):
        combo_substitutions = {}
        for i, element in enumerate(elements):
            concentration = conc_combo[i]
            element_count = substitutions[element].get('element_count', 0)
            n_substitute = int(element_count * concentration)

            combo_substitutions[element] = {
                'new_element': substitutions[element]['new_element'],
                'concentration': concentration,
                'n_substitute': n_substitute,
                'n_remaining': element_count - n_substitute
            }

        combinations.append(combo_substitutions)

    return combinations

def create_combination_name(combo_substitutions):
    """Create a descriptive name for a concentration combination."""
    name_parts = []

    for original_element, sub_info in combo_substitutions.items():
        new_element = sub_info['new_element']
        concentration = sub_info['concentration']
        remaining_concentration = 1 - concentration

        if concentration == 0:
            name_parts.append(f"{original_element}100pct")
        elif concentration == 1:
            if new_element == 'VACANCY':
                name_parts.append(f"{original_element}0pct_100pct_vacant")
            else:
                name_parts.append(f"{new_element}100pct")
        else:
            remaining_pct = int(remaining_concentration * 100)
            substitute_pct = int(concentration * 100)

            if new_element == 'VACANCY':
                name_parts.append(f"{original_element}{remaining_pct}pct_{substitute_pct}pct_vacant")
            else:
                name_parts.append(f"{original_element}{remaining_pct}pct_{new_element}{substitute_pct}pct")

    return "_".join(name_parts)

def sort_concentration_combinations(concentration_combinations):
    """Sort concentration combinations for consistent ordering."""
    def get_sort_key(combo_substitutions):
        sort_values = []
        for element in sorted(combo_substitutions.keys()):
            concentration = combo_substitutions[element]['concentration']
            sort_values.append(concentration)
        return tuple(sort_values)

    return sorted(concentration_combinations, key=get_sort_key)

def calculate_formation_energy(structure_energy, atoms, reference_energies):
    if structure_energy is None:
        return None

    element_counts = {}
    for symbol in atoms.get_chemical_symbols():
        element_counts[symbol] = element_counts.get(symbol, 0) + 1

    total_reference_energy = 0
    for element, count in element_counts.items():
        if element not in reference_energies or reference_energies[element] is None:
            return None
        total_reference_energy += count * reference_energies[element]

    total_atoms = sum(element_counts.values())
    formation_energy_per_atom = (structure_energy - total_reference_energy) / total_atoms
    return formation_energy_per_atom


def create_cell_filter(atoms, pressure, cell_constraint, optimize_lattice, hydrostatic_strain):
    pressure_eV_A3 = pressure * 0.00624150913

    if cell_constraint == "Full cell (lattice + angles)":
        if hydrostatic_strain:
            return ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3, hydrostatic_strain=True)
        else:
            return ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3)
    else:
        if hydrostatic_strain:
            return UnitCellFilter(atoms, scalar_pressure=pressure_eV_A3, hydrostatic_strain=True)
        else:
            mask = [optimize_lattice['a'], optimize_lattice['b'], optimize_lattice['c'], False, False, False]
            return UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)


class OptimizationLogger:
    def __init__(self, filename, max_steps, output_dir="optimized_structures"):
        self.filename = filename
        self.step_count = 0
        self.max_steps = max_steps
        self.step_times = []
        self.step_start_time = time.time()
        self.output_dir = output_dir
        self.trajectory = []
        
    def __call__(self, optimizer=None):
        current_time = time.time()
        
        if self.step_count > 0:
            step_time = current_time - self.step_start_time
            self.step_times.append(step_time)
        
        self.step_count += 1
        self.step_start_time = current_time
        
        if optimizer is not None and hasattr(optimizer, 'atoms'):
            if hasattr(optimizer.atoms, 'atoms'):
                atoms = optimizer.atoms.atoms
            else:
                atoms = optimizer.atoms
                
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            energy = atoms.get_potential_energy()

            # Calculate energy per atom
            energy_per_atom = energy / len(atoms)
            
            if hasattr(self, 'previous_energy') and self.previous_energy is not None:
                energy_change = abs(energy - self.previous_energy)
                energy_change_per_atom = energy_change / len(atoms)
            else:
                energy_change = float('inf')
                energy_change_per_atom = float('inf')
            self.previous_energy = energy
            
            try:
                stress = atoms.get_stress(voigt=True)
                max_stress = np.max(np.abs(stress))
            except:
                max_stress = 0.0
            
            lattice = get_lattice_parameters(atoms)
            
            self.trajectory.append({
                'step': self.step_count,
                'energy': energy,
                'max_force': max_force,
                'positions': atoms.positions.copy(),
                'cell': atoms.cell.array.copy(),
                'lattice': lattice.copy()
            })
            
            if len(self.step_times) > 0:
                avg_time = np.mean(self.step_times)
                remaining_steps = max(0, self.max_steps - self.step_count)
                estimated_remaining_time = avg_time * remaining_steps
                
                if avg_time < 60:
                    avg_time_str = f"{avg_time:.1f}s"
                else:
                    avg_time_str = f"{avg_time/60:.1f}m"
                
                if estimated_remaining_time < 60:
                    remaining_time_str = f"{estimated_remaining_time:.1f}s"
                elif estimated_remaining_time < 3600:
                    remaining_time_str = f"{estimated_remaining_time/60:.1f}m"
                else:
                    remaining_time_str = f"{estimated_remaining_time/3600:.1f}h"
                
                print(f"    Step {self.step_count}: E={energy:.6f} eV ({energy_per_atom:.6f} eV/atom), "
                      f"F_max={max_force:.4f} eV/Å, Max_Stress={max_stress:.4f} GPa, "
                      f"ΔE={energy_change:.2e} eV ({energy_change_per_atom:.2e} eV/atom) | "
                      f"Max. time: {remaining_time_str} ({remaining_steps} steps)")
            else:
                print(f"    Step {self.step_count}: E={energy:.6f} eV ({energy_per_atom:.6f} eV/atom), "
                      f"F_max={max_force:.4f} eV/Å, Max_Stress={max_stress:.4f} GPa, "
                      f"ΔE={energy_change:.2e} eV ({energy_change_per_atom:.2e} eV/atom)")




def main():
    start_time = time.time()
    print("🚀 Starting MACE calculation script...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔬 Calculation type: Geometry Optimization")
    print(f"🤖 Model: MACE-MATPES-PBE-0 (medium) - No +U")
    print(f"💻 Device: cuda")
    print(f"🧵 CPU threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")

    Path("optimized_structures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    

    print("\n📁 Looking for POSCAR files in current directory...")
    structure_files = [f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar")]

    if not structure_files:
        print("❌ No POSCAR files found in current directory!")
        print("Please place files starting with 'POSCAR' or ending with '.vasp' in the same directory as this script.")
        return

    print(f"✅ Found {len(structure_files)} structure files:")
    for i, filename in enumerate(structure_files, 1):
        try:
            atoms = read(filename)
            composition = "".join([f"{symbol}{list(atoms.get_chemical_symbols()).count(symbol)}" 
                                 for symbol in sorted(set(atoms.get_chemical_symbols()))])
            print(f"  {i}. {filename} - {composition} ({len(atoms)} atoms)")
        except Exception as e:
            print(f"  {i}. {filename} - ❌ Error: {str(e)}")

    print("\n🔧 Setting up MLIP calculator...")
    device = "cuda"
    print(f"🔧 Initializing MACE-MP calculator on {device}...")
    try:
        from mace.calculators import mace_mp

        calculator = mace_mp(
            model="https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model", dispersion=False, default_dtype="float32", device=device)
        print(f"✅ MACE-MP calculator initialized successfully on {device}")

    except Exception as e:
        print(f"❌ MACE-MP initialization failed on {device}: {e}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = mace_mp(
                    model="https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model", dispersion=False, default_dtype="float32", device="cpu")
                print("✅ MACE-MP calculator initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {cpu_error}")
                raise cpu_error
        else:
            raise e

    # Run calculations
    print("\n⚡ Starting calculations...")
    calc_start_time = time.time()
    structure_files = [f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar")]
    results = []
    print(f"🔧 Found {len(structure_files)} structure files for optimization")

    optimizer_type = "BFGS"
    fmax = 0.005
    max_steps = 1000
    optimization_type = "Both atoms and cell"
    cell_constraint = "Lattice parameters only (fix angles)"
    pressure = 0.0
    hydrostatic_strain = False
    optimize_lattice = {'a': True, 'b': True, 'c': True}
    
    print(f"⚙️ Optimization settings:")
    print(f"  - Optimizer: {optimizer_type}")
    print(f"  - Force threshold: {fmax} eV/Å")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Type: {optimization_type}")
    if pressure > 0:
        print(f"  - Pressure: {pressure} GPa")

    reference_energies = {}
    print("🔬 Calculating atomic reference energies...")
    all_elements = set()
    for filename in structure_files:
        atoms, _ = read_poscar_with_selective_dynamics(filename)
        for symbol in atoms.get_chemical_symbols():
            all_elements.add(symbol)

    print(f"🧪 Found elements: {', '.join(sorted(all_elements))}")
    
    for i, element in enumerate(sorted(all_elements)):
        print(f"  📍 Calculating reference for {element} ({i+1}/{len(all_elements)})...")
        atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
        atom.calc = calculator
        reference_energies[element] = atom.get_potential_energy()
        print(f"  ✅ {element}: {reference_energies[element]:.6f} eV")

    for i, filename in enumerate(structure_files):
        print(f"\n🔧 Processing structure {i+1}/{len(structure_files)}: {filename}")
        structure_start_time = time.time()
        try:
            atoms, selective_dynamics = read_poscar_with_selective_dynamics(filename)
            atoms.calc = calculator
            print(f"  📊 Structure has {len(atoms)} atoms")
            initial_atoms_copy = atoms.copy()
            if selective_dynamics is not None:
                atoms = apply_selective_dynamics_constraints(atoms, selective_dynamics)
            else:
                print(f"  🔄 No selective dynamics found - all atoms free to move")

            initial_energy = atoms.get_potential_energy()
            initial_forces = atoms.get_forces()
            initial_max_force = np.max(np.linalg.norm(initial_forces, axis=1))
            print(f"  📊 Initial energy: {initial_energy:.6f} eV")
            print(f"  📊 Initial max force: {initial_max_force:.4f} eV/Å")

            if optimization_type == "Atoms only (fixed cell)":
                optimization_object = atoms
                opt_mode = "atoms_only"
                print(f"  🔒 Optimizing atoms only (fixed cell)")
            elif optimization_type == "Cell only (fixed atoms)":
                existing_constraints = atoms.constraints if hasattr(atoms, 'constraints') and atoms.constraints else []
                all_fixed_constraint = FixAtoms(mask=[True] * len(atoms))
                atoms.set_constraint([all_fixed_constraint] + existing_constraints)
                optimization_object = create_cell_filter(atoms, pressure, cell_constraint, optimize_lattice, hydrostatic_strain)
                opt_mode = "cell_only"
                print(f"  🔒 Optimizing cell only (fixed atoms)")
            else:
                optimization_object = create_cell_filter(atoms, pressure, cell_constraint, optimize_lattice, hydrostatic_strain)
                opt_mode = "both"
                print(f"  🔄 Optimizing both atoms and cell")

            logger = OptimizationLogger(filename, max_steps, "optimized_structures")
            
            if optimizer_type == "LBFGS":
                optimizer = LBFGS(optimization_object, logfile=f"results/{filename}_opt.log")
            else:
                optimizer = BFGS(optimization_object, logfile=f"results/{filename}_opt.log")

            optimizer.attach(lambda: logger(optimizer), interval=1)

            print(f"  🏃 Running {optimizer_type} optimization...")
            if opt_mode == "cell_only":
                optimizer.run(fmax=0.1, steps=max_steps)
            else:
                optimizer.run(fmax=fmax, steps=max_steps)

            if hasattr(optimization_object, 'atoms'):
                final_atoms = optimization_object.atoms
            else:
                final_atoms = optimization_object

            final_energy = final_atoms.get_potential_energy()
            final_forces = final_atoms.get_forces()
            max_final_force = np.max(np.linalg.norm(final_forces, axis=1))

            force_converged = max_final_force < fmax
            if opt_mode in ["cell_only", "both"]:
                try:
                    final_stress = final_atoms.get_stress(voigt=True)
                    max_final_stress = np.max(np.abs(final_stress))
                    stress_converged = max_final_stress < 0.1
                except:
                    stress_converged = True
                    max_final_stress = 0.0
            else:
                stress_converged = True
                max_final_stress = 0.0

            if opt_mode == "atoms_only":
                convergence_status = "CONVERGED" if force_converged else "MAX_STEPS_REACHED"
            elif opt_mode == "cell_only":
                convergence_status = "CONVERGED" if stress_converged else "MAX_STEPS_REACHED"
            else:
                convergence_status = "CONVERGED" if (force_converged and stress_converged) else "MAX_STEPS_REACHED"

            base_name = filename.replace('.vasp', '').replace('POSCAR', '')
            
            output_filename = f"optimized_structures/optimized_{base_name}.vasp"
            
            print(f"  💾 Saving optimized structure to {output_filename}")
            write_poscar_with_selective_dynamics(
                final_atoms, 
                output_filename, 
                selective_dynamics, 
                f"Optimized - {convergence_status}"
            )
            
            detailed_summary_file = "results/optimization_detailed_summary.csv"
            print(f"  📊 Appending detailed summary to {detailed_summary_file}")
            append_optimization_summary(
                detailed_summary_file, 
                filename, 
                initial_atoms_copy, 
                final_atoms,      
                initial_energy, 
                final_energy, 
                convergence_status, 
                optimizer.nsteps,
                selective_dynamics
            )
            
            trajectory_filename = f"optimized_structures/trajectory_{base_name}.xyz"
            print(f"  📈 Saving optimization trajectory to {trajectory_filename}")
            
            with open(trajectory_filename, 'w') as traj_file:
                symbols = final_atoms.get_chemical_symbols()
                for step_data in logger.trajectory:
                    num_atoms = len(step_data['positions'])
                    energy = step_data['energy']
                    max_force = step_data['max_force']
                    lattice = step_data['lattice']
                    step = step_data['step']
                    cell_matrix = step_data['cell']
                    
                    lattice_string = " ".join([f"{x:.6f}" for row in cell_matrix for x in row])
                    
                    traj_file.write(f"{num_atoms}\n")
                    
                    comment = (f'Step={step} Energy={energy:.6f} Max_Force={max_force:.6f} '
                              f'a={lattice["a"]:.6f} b={lattice["b"]:.6f} c={lattice["c"]:.6f} '
                              f'alpha={lattice["alpha"]:.3f} beta={lattice["beta"]:.3f} gamma={lattice["gamma"]:.3f} '
                              f'Volume={lattice["volume"]:.6f} '
                              f'Lattice="{lattice_string}" '
                              f'Properties=species:S:1:pos:R:3')
                    traj_file.write(f"{comment}\n")
                    
                    for j, pos in enumerate(step_data['positions']):
                        symbol = symbols[j] if j < len(symbols) else 'X'
                        traj_file.write(f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")

            result = {
                "structure": filename,
                "initial_energy_eV": initial_energy,
                "final_energy_eV": final_energy,
                "energy_change_eV": final_energy - initial_energy,
                "initial_max_force_eV_per_A": initial_max_force,
                "final_max_force_eV_per_A": max_final_force,
                "max_stress_GPa": max_final_stress,
                "convergence_status": convergence_status,
                "optimization_steps": optimizer.nsteps,
                "calculation_type": "geometry_optimization",
                "num_atoms": len(atoms),
                "opt_mode": opt_mode,
                "optimizer_type": optimizer_type,
                "fmax": fmax,
                "max_steps": max_steps,
                "optimization_type": optimization_type,
                "cell_constraint": cell_constraint,
                "pressure": pressure,
                "hydrostatic_strain": hydrostatic_strain,
                "has_selective_dynamics": selective_dynamics is not None,
                "num_fixed_atoms": len([flags for flags in (selective_dynamics or []) if not any(flags)]),
                "output_structure": output_filename,
                "trajectory_file": trajectory_filename,
                "optimize_lattice_a": optimize_lattice.get('a', True) if isinstance(optimize_lattice, dict) else True,
                "optimize_lattice_b": optimize_lattice.get('b', True) if isinstance(optimize_lattice, dict) else True,
                "optimize_lattice_c": optimize_lattice.get('c', True) if isinstance(optimize_lattice, dict) else True
            }

            formation_energy = calculate_formation_energy(final_energy, final_atoms, reference_energies)
            result["formation_energy_eV_per_atom"] = formation_energy

            structure_time = time.time() - structure_start_time
            print(f"  ✅ Optimization completed: {convergence_status}")
            print(f"  ✅ Final energy: {final_energy:.6f} eV (Δ={final_energy - initial_energy:.6f} eV)")
            print(f"  ✅ Final max force: {max_final_force:.4f} eV/Å")
            if opt_mode in ["cell_only", "both"]:
                print(f"  ✅ Final max stress: {max_final_stress:.4f} GPa")
            print(f"  ✅ Steps: {optimizer.nsteps}")
            print(f"  ⏱️ Structure time: {structure_time:.1f}s")
            print(f"  💾 Saved to: {output_filename}")
            if formation_energy is not None:
                print(f"  ✅ Formation energy: {formation_energy:.6f} eV/atom")
            results.append(result)
            
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/optimization_results.csv", index=False)
            print(f"  💾 Results updated and saved")

        except Exception as e:
            print(f"  ❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({"structure": filename, "error": str(e)})
            
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/optimization_results.csv", index=False)
            print(f"  💾 Results updated and saved (with error)")

    df_results = pd.DataFrame(results)
    df_results.to_csv("results/optimization_results.csv", index=False)

    print(f"\n💾 Saved all results to results/optimization_results.csv")
    print(f"📁 Optimized structures saved in optimized_structures/ directory")

    with open("results/optimization_summary.txt", "w") as f:
        f.write("MACE Geometry Optimization Results\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            if "error" not in result:
                f.write(f"Structure: {result['structure']}\n")
                f.write(f"Initial Energy: {result['initial_energy_eV']:.6f} eV\n")
                f.write(f"Final Energy: {result['final_energy_eV']:.6f} eV\n")
                f.write(f"Energy Change: {result['energy_change_eV']:.6f} eV\n")
                f.write(f"Final Max Force: {result['final_max_force_eV_per_A']:.4f} eV/Å\n")
                f.write(f"Max Stress: {result['max_stress_GPa']:.4f} GPa\n")
                f.write(f"Convergence: {result['convergence_status']}\n")
                f.write(f"Steps: {result['optimization_steps']}\n")
                f.write(f"Atoms: {result['num_atoms']}\n")
                f.write(f"Selective Dynamics: {result['has_selective_dynamics']}\n")
                if result['has_selective_dynamics']:
                    f.write(f"Fixed Atoms: {result['num_fixed_atoms']}/{result['num_atoms']}\n")
                f.write(f"Output File: {result['output_structure']}\n")
                if "formation_energy_eV_per_atom" in result and result["formation_energy_eV_per_atom"] is not None:
                    f.write(f"Formation Energy: {result['formation_energy_eV_per_atom']:.6f} eV/atom\n")
                f.write("\n")
            else:
                f.write(f"Structure: {result['structure']} - ERROR: {result['error']}\n\n")
    
    print(f"💾 Saved summary to results/optimization_summary.txt")
    print("\n📊 Generating optimization plots...")
    successful_results = [r for r in results if "error" not in r]
    
    if len(successful_results) > 0:
        try:
            import matplotlib.pyplot as plt
            
            plt.rcParams.update({
                'font.size': 18,
                'axes.titlesize': 24,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18,
                'figure.titlesize': 26
            })

            structure_names = [r["structure"] for r in successful_results]
            final_energies = [r["final_energy_eV"] for r in successful_results]
            
            # 1. Total Energy Plot
            plt.figure(figsize=(16, 12))
            bars = plt.bar(range(len(structure_names)), final_energies, color='steelblue', alpha=0.7)
            plt.xlabel('Structure', fontsize=22, fontweight='bold')
            plt.ylabel('Final Energy (eV)', fontsize=22, fontweight='bold')
            plt.title('Final Energy After Optimization', fontsize=26, fontweight='bold', pad=20)
            plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                      rotation=45, ha='right', fontsize=18, fontweight='bold')
            plt.yticks(fontsize=18, fontweight='bold')
            
            y_min, y_max = plt.ylim()
            y_range = y_max - y_min
            plt.ylim(y_min, y_max + y_range * 0.15)
            
            for i, (bar, energy) in enumerate(zip(bars, final_energies)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_range * 0.02, 
                        f'{energy:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold', 
                        rotation=90, color='black')
            
            plt.tight_layout()
            plt.savefig('results/optimization_final_energy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Saved final energy plot: results/optimization_final_energy_comparison.png")
            
            # 2. Formation Energy Plot
            formation_energies = [r.get("formation_energy_eV_per_atom") for r in successful_results]
            valid_formation = [(name, fe, result) for name, fe, result in zip(structure_names, formation_energies, successful_results) if fe is not None]
            
            if valid_formation:
                valid_names, valid_fe, valid_results = zip(*valid_formation)
                
                plt.figure(figsize=(16, 12))
                colors = ['green' if fe == min(valid_fe) else 'orange' for fe in valid_fe]
                bars = plt.bar(range(len(valid_names)), valid_fe, color=colors, alpha=0.7)
                plt.xlabel('Structure', fontsize=22, fontweight='bold')
                plt.ylabel('Formation Energy (eV/atom)', fontsize=22, fontweight='bold')
                plt.title('Formation Energy per Atom After Optimization', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(range(len(valid_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in valid_names], 
                          rotation=45, ha='right', fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')
                
                y_min, y_max = plt.ylim()
                y_range = y_max - y_min
                
                has_negative = any(fe < 0 for fe in valid_fe)
                has_positive = any(fe > 0 for fe in valid_fe)
                
                if has_negative and has_positive:
                    plt.ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)
                elif has_negative and not has_positive:
                    plt.ylim(y_min - y_range * 0.15, y_max + y_range * 0.05)
                else:
                    plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)

                for i, (bar, fe) in enumerate(zip(bars, valid_fe)):
                    if fe >= 0:
                        y_pos = bar.get_height() + y_range * 0.02
                        va_align = 'bottom'
                    else:
                        y_pos = bar.get_height() - y_range * 0.02
                        va_align = 'top'
                    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
                            f'{fe:.4f}', ha='center', va=va_align, fontsize=16, fontweight='bold', 
                            rotation=90, color='black')
                
                plt.tight_layout()
                plt.savefig('results/optimization_formation_energy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ Saved formation energy plot: results/optimization_formation_energy_comparison.png")
            
            # 3. Relative Energy Plot
            if len(final_energies) > 1:
                min_energy = min(final_energies)
                relative_energies = [(e - min_energy) * 1000 for e in final_energies]  # Convert to meV
                
                plt.figure(figsize=(16, 12))
                colors = ['green' if re == 0 else 'orange' for re in relative_energies]
                bars = plt.bar(range(len(structure_names)), relative_energies, color=colors, alpha=0.7)
                plt.xlabel('Structure', fontsize=22, fontweight='bold')
                plt.ylabel('Relative Energy (meV)', fontsize=22, fontweight='bold')
                plt.title('Relative Energy Comparison After Optimization (vs. Lowest Energy)', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                          rotation=45, ha='right', fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')
                
                y_min, y_max = plt.ylim()
                y_range = max(relative_energies) if max(relative_energies) > 0 else 1
                plt.ylim(-y_range * 0.1, max(relative_energies) + y_range * 0.15)
                
                for i, (bar, re) in enumerate(zip(bars, relative_energies)):
                    if re > 0:
                        y_pos = bar.get_height() + y_range * 0.02
                        va_align = 'bottom'
                    else:
                        y_pos = y_range * 0.05 
                        va_align = 'bottom'
                    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
                            f'{re:.1f}', ha='center', va=va_align, fontsize=16, fontweight='bold', 
                            rotation=90, color='black')
                
                plt.tight_layout()
                plt.savefig('results/optimization_relative_energy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ Saved relative energy plot: results/optimization_relative_energy_comparison.png")
            
            # 4. Lattice Parameter Changes Plot
            print("  📏 Reading detailed optimization summary for lattice changes...")
            try:
                detailed_summary_file = "results/optimization_detailed_summary.csv"
                if os.path.exists(detailed_summary_file):
                    df_lattice = pd.read_csv(detailed_summary_file)
                    
                    if len(df_lattice) > 0:
                        plt.figure(figsize=(18, 10))
                        
                        structures = df_lattice['Structure'].tolist()
                        a_changes = df_lattice['a_change_percent'].tolist()
                        b_changes = df_lattice['b_change_percent'].tolist()
                        c_changes = df_lattice['c_change_percent'].tolist()

                        x = np.arange(len(structures))
                        width = 0.25
                        
                        bars1 = plt.bar(x - width, a_changes, width, label='a parameter', color='red', alpha=0.7)
                        bars2 = plt.bar(x, b_changes, width, label='b parameter', color='green', alpha=0.7)
                        bars3 = plt.bar(x + width, c_changes, width, label='c parameter', color='blue', alpha=0.7)
                        
                        plt.xlabel('Structure', fontsize=22, fontweight='bold')
                        plt.ylabel('Lattice Parameter Change (%)', fontsize=22, fontweight='bold')
                        plt.title('Lattice Parameter Changes After Optimization', fontsize=26, fontweight='bold', pad=20)
                        plt.xticks(x, [name.replace('.vasp', '').replace('POSCAR_', '') for name in structures], 
                                  rotation=45, ha='right', fontsize=18, fontweight='bold')
                        plt.yticks(fontsize=18, fontweight='bold')
                        
                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
                        plt.grid(True, alpha=0.3, axis='y')
                        
                        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                                  ncol=3, fontsize=18, frameon=False)

                        plt.subplots_adjust(bottom=0.2)
                        plt.tight_layout()
                        plt.savefig('results/lattice_parameter_changes.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        print("  ✅ Saved lattice changes plot: results/lattice_parameter_changes.png")
                    else:
                        print("  ⚠️ No lattice data found in detailed summary")
                else:
                    print("  ⚠️ Detailed optimization summary file not found")
                    
            except Exception as lattice_error:
                print(f"  ⚠️ Error creating lattice changes plot: {lattice_error}")
            
            plt.rcParams.update(plt.rcParamsDefault)
            
        except ImportError:
            print("  ⚠️ Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"  ⚠️ Error generating plots: {e}")
    
    else:
        print("  ℹ️ No successful calculations to plot")


    total_time = time.time() - start_time
    calc_time = time.time() - calc_start_time
    print(f"\n✅ All calculations completed!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"⏱️ Calculation time: {calc_time/60:.1f} minutes")
    print("📊 Check the results/ directory for output files")

if __name__ == "__main__":
    main()
