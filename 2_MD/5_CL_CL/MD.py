#!/usr/bin/env python3
"""
Pair-ion MD for CHARMM TIP3P water clusters in OpenMM.

Use cases
---------
(K+, K+), (Cl-, Cl-), and (K+, Cl-) separated by d = 1..8 A.
The ion-pair center is placed at the spherical water-cluster center.
The PDB files are read from --input_dir using names such as:

  pair_K_K_d01A_sel21A_1117w.pdb
  pair_Cl_Cl_d01A_sel21A_1117w.pdb
  pair_K_Cl_d01A_sel21A_1117w.pdb

Main protocol
-------------
- CHARMM force field + CHARMM modified TIP3P water
- NoCutoff
- no electrostatic cutoff
- no LJ cutoff
- LJ switching disabled
- constraints = HBonds
- rigidWater = True
- timestep = 2 fs
- both ion masses are set to zero: ion positions fixed, interactions retained
- ion charges and vdW/LJ parameters are unchanged from CHARMM force field
- spherical wall on water oxygen atoms, R = 20 A
- each distance: minimization + 400 ps equilibration + 10 ns production
- DCD saved every 100 fs

Outputs
-------
out_dir/
  run_settings.txt
  summary.csv
  d01/system.xml
  d01/forcefield_parameters.txt
  d01/equil_state.csv
  d01/traj.dcd
  d01/state.csv
  d01/final.pdb
  d01/checkpoint.chk
  d01/d_info.txt
  ... d02 ... d08

Notes
-----
For CHARMM XML in OpenMM, electrostatics are in NonbondedForce, while CHARMM LJ/vdW
is commonly represented by a CustomNonbondedForce named "LennardJones" with
energy acoef(type1,type2)/r^12 - bcoef(type1,type2)/r^6. This report extracts
charges from NonbondedForce and LJ pair parameters from CustomNonbondedForce.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openmm import (
        CustomExternalForce,
        CustomNonbondedForce,
        HarmonicAngleForce,
        HarmonicBondForce,
        LangevinMiddleIntegrator,
        NonbondedForce,
        Platform,
        Vec3,
        XmlSerializer,
    )
    from openmm.app import (
        NoCutoff,
        CutoffNonPeriodic,
        DCDReporter,
        ForceField,
        HBonds,
        PDBFile,
        Simulation,
        StateDataReporter,
        element,
    )
    from openmm.unit import (
        angstrom,
        dalton,
        elementary_charge,
        femtosecond,
        kelvin,
        kilocalorie_per_mole,
        kilojoule_per_mole,
        nanometer,
        picosecond,
        radian,
    )
except ImportError:
    from simtk.openmm import (  # type: ignore
        CustomExternalForce,
        CustomNonbondedForce,
        HarmonicAngleForce,
        HarmonicBondForce,
        LangevinMiddleIntegrator,
        NonbondedForce,
        Platform,
        Vec3,
        XmlSerializer,
    )
    from simtk.openmm.app import (  # type: ignore
        NoCutoff,
        CutoffNonPeriodic,
        DCDReporter,
        ForceField,
        HBonds,
        PDBFile,
        Simulation,
        StateDataReporter,
        element,
    )
    from simtk.unit import (  # type: ignore
        angstrom,
        dalton,
        elementary_charge,
        femtosecond,
        kelvin,
        kilocalorie_per_mole,
        kilojoule_per_mole,
        nanometer,
        picosecond,
        radian,
    )


ION_RESNAMES = {
    "K": {"POT"},
    "Cl": {"CLA"},
}

ION_ELEMENTS = {
    "K": element.potassium,
    "Cl": element.chlorine,
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL"}
OXYGEN_NAMES = {"O", "OH2", "OW", "OT"}


# ------------------------- general helpers -------------------------


def d_label(d: int) -> str:
    return f"d{int(d):02d}"


def steps_from_time(time_value: float, time_unit: str, timestep_fs: float) -> int:
    if time_unit == "ps":
        fs = time_value * 1000.0
    elif time_unit == "ns":
        fs = time_value * 1_000_000.0
    elif time_unit == "fs":
        fs = time_value
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")
    steps = fs / timestep_fs
    if abs(steps - round(steps)) > 1e-8:
        raise ValueError(f"{time_value} {time_unit} is not an integer number of steps at {timestep_fs} fs")
    return int(round(steps))


def ensure_clean_dir(path: Path, overwrite: bool = False):
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Output directory exists: {path}. Use --overwrite to replace it.")
    path.mkdir(parents=True, exist_ok=True)


def force_name(force) -> str:
    try:
        name = force.getName()
        if name:
            return name
    except Exception:
        pass
    return force.__class__.__name__


def format_float(x, ndigits=10) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{ndigits}f}"
    return str(x)


# ------------------------- topology/atom identification -------------------------


def is_ion_atom(atom, ion_label: str) -> bool:
    allowed_resnames = ION_RESNAMES[ion_label]
    expected_element = ION_ELEMENTS[ion_label]
    return atom.residue.name in allowed_resnames or atom.element == expected_element


def find_pair_ion_atoms(topology, pair: Sequence[str]) -> List[int]:
    """Return ion atom indices in the requested order. Handles K/K, Cl/Cl, K/Cl."""
    ion1, ion2 = pair[0], pair[1]
    if ion1 not in ION_RESNAMES or ion2 not in ION_RESNAMES:
        raise ValueError("Only K and Cl are supported for pair-ion runs.")

    if ion1 == ion2:
        candidates = [atom for atom in topology.atoms() if is_ion_atom(atom, ion1)]
        if len(candidates) != 2:
            details = [(a.index, a.name, a.residue.name, str(a.element)) for a in candidates]
            raise RuntimeError(f"Expected exactly two {ion1} ions, found {len(candidates)}: {details}")
        return [a.index for a in sorted(candidates, key=lambda a: a.index)]

    out = []
    for ion in [ion1, ion2]:
        candidates = [atom for atom in topology.atoms() if is_ion_atom(atom, ion)]
        if len(candidates) != 1:
            details = [(a.index, a.name, a.residue.name, str(a.element)) for a in candidates]
            raise RuntimeError(f"Expected exactly one {ion} ion, found {len(candidates)}: {details}")
        out.append(candidates[0].index)
    return out


def is_water_residue(residue) -> bool:
    atoms = list(residue.atoms())
    if residue.name in WATER_RESNAMES:
        return True
    n_o = 0
    n_h = 0
    for atom in atoms:
        if atom.element == element.oxygen or atom.name in OXYGEN_NAMES:
            n_o += 1
        elif atom.element == element.hydrogen or atom.name.upper().startswith("H"):
            n_h += 1
    return len(atoms) == 3 and n_o == 1 and n_h == 2


def find_water_atoms(topology) -> Tuple[List[int], List[Tuple[int, int, int, object]]]:
    """Return water oxygen indices and water triplets (O,H1,H2,residue)."""
    oxygens = []
    waters = []
    for residue in topology.residues():
        if not is_water_residue(residue):
            continue
        atoms = list(residue.atoms())
        o_atoms = [a for a in atoms if a.element == element.oxygen or a.name in OXYGEN_NAMES]
        h_atoms = [a for a in atoms if a.element == element.hydrogen or a.name.upper().startswith("H")]
        if len(o_atoms) != 1 or len(h_atoms) != 2:
            raise RuntimeError(f"Cannot identify water atoms in residue {residue.name} {residue.id}")
        o = o_atoms[0]
        h_atoms = sorted(h_atoms, key=lambda a: a.name)
        h1, h2 = h_atoms[0], h_atoms[1]
        oxygens.append(o.index)
        waters.append((o.index, h1.index, h2.index, residue))
    if not waters:
        raise RuntimeError("No water molecules detected.")
    return oxygens, waters


def topology_atom_by_index(topology, atom_index: int):
    for atom in topology.atoms():
        if atom.index == atom_index:
            return atom
    raise RuntimeError(f"Atom index not found in topology: {atom_index}")


def recenter_positions_on_pair_center(positions, ion_indices: Sequence[int]):
    if len(ion_indices) != 2:
        raise ValueError("Need exactly two ion indices for pair centering")
    p0 = positions[ion_indices[0]].value_in_unit(angstrom)
    p1 = positions[ion_indices[1]].value_in_unit(angstrom)
    center = [(float(p0[i]) + float(p1[i])) / 2.0 for i in range(3)]
    centered = []
    for p in positions:
        v = p.value_in_unit(angstrom)
        centered.append(Vec3(float(v[0]) - center[0], float(v[1]) - center[1], float(v[2]) - center[2]))
    return centered * angstrom


def distance_between_atoms_A(positions, i: int, j: int) -> float:
    pi = positions[i].value_in_unit(angstrom)
    pj = positions[j].value_in_unit(angstrom)
    return math.sqrt(sum((float(pi[k]) - float(pj[k])) ** 2 for k in range(3)))


# ------------------------- force identification -------------------------


def get_nonbonded_force(system) -> NonbondedForce:
    matches = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one NonbondedForce, found {len(matches)}")
    return matches[0]


def get_lj_force(system) -> Optional[CustomNonbondedForce]:
    candidates = []
    for f in system.getForces():
        if isinstance(f, CustomNonbondedForce):
            energy = f.getEnergyFunction()
            name = force_name(f)
            if "acoef" in energy and "bcoef" in energy:
                candidates.append(f)
            elif name.lower() == "lennardjones":
                candidates.append(f)
    if not candidates:
        return None
    if len(candidates) > 1:
        for f in candidates:
            if force_name(f).lower() == "lennardjones":
                return f
    return candidates[0]


def get_harmonic_bond_force(system) -> Optional[HarmonicBondForce]:
    for f in system.getForces():
        if isinstance(f, HarmonicBondForce):
            return f
    return None


def get_harmonic_angle_force(system) -> Optional[HarmonicAngleForce]:
    for f in system.getForces():
        if isinstance(f, HarmonicAngleForce):
            return f
    return None


# ------------------------- LJ extraction from CustomNonbondedForce -------------------------


def custom_force_particle_type(force: CustomNonbondedForce, atom_index: int) -> int:
    param_names = [force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters())]
    if "type" not in param_names:
        raise RuntimeError(f"CustomNonbondedForce {force_name(force)} has no per-particle parameter named 'type'")
    type_pos = param_names.index("type")
    values = force.getParticleParameters(atom_index)
    val = values[type_pos]
    try:
        return int(round(float(val)))
    except TypeError:
        return int(round(float(val._value)))


def get_discrete2d_table(force: CustomNonbondedForce, table_name: str):
    for i in range(force.getNumTabulatedFunctions()):
        name = force.getTabulatedFunctionName(i)
        if name != table_name:
            continue
        func = force.getTabulatedFunction(i)
        if not hasattr(func, "getFunctionParameters"):
            raise RuntimeError(f"Tabulated function {table_name} has no getFunctionParameters()")
        params = func.getFunctionParameters()
        if len(params) != 3:
            raise RuntimeError(f"Unexpected function parameter format for {table_name}: {params}")
        xsize, ysize, values = params
        return int(xsize), int(ysize), list(values)
    raise RuntimeError(f"CustomNonbondedForce {force_name(force)} has no tabulated function named {table_name}")


def table2d_value(xsize: int, ysize: int, values: Sequence[float], i: int, j: int) -> float:
    if i < 0 or i >= xsize or j < 0 or j >= ysize:
        raise IndexError(f"Table index out of range: ({i}, {j}) size=({xsize}, {ysize})")
    return float(values[i + xsize * j])


def lj_pair_from_acoef_bcoef(force: CustomNonbondedForce, type_i: int, type_j: int) -> Dict[str, Optional[float]]:
    ax, ay, avals = get_discrete2d_table(force, "acoef")
    bx, by, bvals = get_discrete2d_table(force, "bcoef")
    a = table2d_value(ax, ay, avals, type_i, type_j)
    b = table2d_value(bx, by, bvals, type_i, type_j)

    if a <= 0.0 or b <= 0.0:
        return {
            "type_i": type_i,
            "type_j": type_j,
            "acoef": a,
            "bcoef": b,
            "sigma_nm": None,
            "sigma_A": None,
            "epsilon_kJ_mol": None,
            "epsilon_kcal_mol_positive": None,
            "epsilon_charmm_kcal_mol": None,
            "Rmin_over_2_A": None,
        }

    sigma_nm = (a / b) ** (1.0 / 6.0)
    epsilon_kJ = (b * b) / (4.0 * a)
    sigma_A = sigma_nm * 10.0
    rmin_over_2_A = (2.0 ** (1.0 / 6.0)) * sigma_A / 2.0
    epsilon_kcal = epsilon_kJ / 4.184

    return {
        "type_i": type_i,
        "type_j": type_j,
        "acoef": a,
        "bcoef": b,
        "sigma_nm": sigma_nm,
        "sigma_A": sigma_A,
        "epsilon_kJ_mol": epsilon_kJ,
        "epsilon_kcal_mol_positive": epsilon_kcal,
        "epsilon_charmm_kcal_mol": -epsilon_kcal,
        "Rmin_over_2_A": rmin_over_2_A,
    }


# ------------------------- parameters from forces -------------------------


def charge_e_from_nonbonded(nbforce: NonbondedForce, atom_index: int) -> float:
    charge, sigma, epsilon = nbforce.getParticleParameters(atom_index)
    return float(charge.value_in_unit(elementary_charge))


def mass_da(system, atom_index: int) -> float:
    return float(system.getParticleMass(atom_index).value_in_unit(dalton))


def set_cutoffs_and_switching(system, cutoff_a: float, switch_a: float):
    cutoff = cutoff_a * angstrom
    switch = switch_a * angstrom
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            force.setNonbondedMethod(NonbondedForce.CutoffNonPeriodic)
            force.setCutoffDistance(cutoff)
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(switch)
        elif isinstance(force, CustomNonbondedForce):
            force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
            force.setCutoffDistance(cutoff)
            if hasattr(force, "setUseSwitchingFunction"):
                force.setUseSwitchingFunction(True)
            if hasattr(force, "setSwitchingDistance"):
                force.setSwitchingDistance(switch)


def set_no_cutoff(system):
    """Use direct pairwise nonbonded interactions without cutoff or switching."""
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            force.setNonbondedMethod(NonbondedForce.NoCutoff)
            force.setUseSwitchingFunction(False)
        elif isinstance(force, CustomNonbondedForce):
            force.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
            if hasattr(force, "setUseSwitchingFunction"):
                force.setUseSwitchingFunction(False)


def add_spherical_wall(system, water_oxygen_indices: Iterable[int], radius_a: float, k_kcal_mol_A2: float):
    # Coordinates are in nm in OpenMM energy expressions.
    # kcal/mol/A^2 -> kJ/mol/nm^2: multiply by 4.184 / (0.1 nm)^2 = 418.4
    k_kJ_mol_nm2 = k_kcal_mol_A2 * 418.4
    radius_nm = radius_a / 10.0
    wall = CustomExternalForce("k*step(r-R)*(r-R)^2; r=sqrt(x*x+y*y+z*z)")
    wall.setName("WaterO_SphericalWall")
    wall.addGlobalParameter("k", k_kJ_mol_nm2)
    wall.addGlobalParameter("R", radius_nm)
    n = 0
    for idx in water_oxygen_indices:
        wall.addParticle(int(idx), [])
        n += 1
    system.addForce(wall)
    return n


def bond_terms_for_atoms(system, atom_indices: Sequence[int]) -> List[Dict[str, object]]:
    atom_set = set(atom_indices)
    out = []
    force = get_harmonic_bond_force(system)
    if force is None:
        return out
    for i in range(force.getNumBonds()):
        p1, p2, length, k = force.getBondParameters(i)
        if p1 in atom_set and p2 in atom_set:
            k_kJ_nm2 = float(k.value_in_unit(kilojoule_per_mole / nanometer**2))
            out.append({
                "p1": p1,
                "p2": p2,
                "length_A": float(length.value_in_unit(angstrom)),
                "k_OpenMM_kJ_mol_nm2": k_kJ_nm2,
                "k_OpenMM_kcal_mol_A2": k_kJ_nm2 / 4.184 / 100.0,
                "k_CHARMM_style_kcal_mol_A2": k_kJ_nm2 / 4.184 / 100.0 / 2.0,
            })
    return out


def angle_terms_for_atoms(system, atom_indices: Sequence[int]) -> List[Dict[str, object]]:
    atom_set = set(atom_indices)
    out = []
    force = get_harmonic_angle_force(system)
    if force is None:
        return out
    for i in range(force.getNumAngles()):
        p1, p2, p3, theta0, k = force.getAngleParameters(i)
        if p1 in atom_set and p2 in atom_set and p3 in atom_set:
            k_kJ_rad2 = float(k.value_in_unit(kilojoule_per_mole / radian**2))
            theta_rad = float(theta0.value_in_unit(radian))
            out.append({
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "theta0_deg": theta_rad * 180.0 / math.pi,
                "k_OpenMM_kJ_mol_rad2": k_kJ_rad2,
                "k_OpenMM_kcal_mol_rad2": k_kJ_rad2 / 4.184,
                "k_CHARMM_style_kcal_mol_rad2": k_kJ_rad2 / 4.184 / 2.0,
            })
    return out


def constraint_terms_for_atoms(system, atom_indices: Sequence[int]) -> List[Dict[str, object]]:
    atom_set = set(atom_indices)
    out = []
    for i in range(system.getNumConstraints()):
        p1, p2, dist = system.getConstraintParameters(i)
        if p1 in atom_set and p2 in atom_set:
            out.append({
                "p1": p1,
                "p2": p2,
                "distance_A": float(dist.value_in_unit(angstrom)),
            })
    return out


def geometry_from_positions(positions, o_idx: int, h1_idx: int, h2_idx: int) -> Dict[str, float]:
    def xyz(idx):
        v = positions[idx].value_in_unit(angstrom)
        return (float(v[0]), float(v[1]), float(v[2]))

    def dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def angle(a, b, c):
        ba = [a[i] - b[i] for i in range(3)]
        bc = [c[i] - b[i] for i in range(3)]
        dot = sum(ba[i] * bc[i] for i in range(3))
        nba = math.sqrt(sum(x * x for x in ba))
        nbc = math.sqrt(sum(x * x for x in bc))
        cosang = max(-1.0, min(1.0, dot / (nba * nbc)))
        return math.degrees(math.acos(cosang))

    o = xyz(o_idx)
    h1 = xyz(h1_idx)
    h2 = xyz(h2_idx)
    return {
        "OH1_A": dist(o, h1),
        "OH2_A": dist(o, h2),
        "HH_A": dist(h1, h2),
        "HOH_deg": angle(h1, o, h2),
    }


def write_lj_pair_section(f, title: str, p: Dict[str, Optional[float]]):
    f.write(f"{title}:\n")
    f.write(f"  type_i = {p['type_i']}\n")
    f.write(f"  type_j = {p['type_j']}\n")
    f.write(f"  acoef = {format_float(p['acoef'], 16)}\n")
    f.write(f"  bcoef = {format_float(p['bcoef'], 16)}\n")
    if p["sigma_A"] is None:
        f.write("  equivalent_sigma_A = NA\n")
        f.write("  equivalent_epsilon_kJ_mol = NA\n")
        f.write("  equivalent_CHARMM_Rmin_over_2_A = NA\n")
        f.write("  equivalent_CHARMM_epsilon_kcal_mol = NA\n")
    else:
        f.write(f"  equivalent_sigma_A = {p['sigma_A']:.10f}\n")
        f.write(f"  equivalent_epsilon_kJ_mol = {p['epsilon_kJ_mol']:.10f}\n")
        f.write(f"  equivalent_epsilon_kcal_mol_positive = {p['epsilon_kcal_mol_positive']:.10f}\n")
        f.write(f"  equivalent_CHARMM_Rmin_over_2_A = {p['Rmin_over_2_A']:.10f}\n")
        f.write(f"  equivalent_CHARMM_epsilon_kcal_mol = {p['epsilon_charmm_kcal_mol']:.10f}\n")
    f.write("\n")


# ------------------------- reports -------------------------


def write_forcefield_parameter_report(
    out_dir: Path,
    topology,
    system,
    centered_positions,
    pair: Sequence[str],
    ion_indices: Sequence[int],
    original_ion_masses_da: Sequence[float],
    ff_files: Sequence[str],
    md_settings: Dict[str, object],
):
    nbforce = get_nonbonded_force(system)
    ljforce = get_lj_force(system)
    water_oxygen_indices, waters = find_water_atoms(topology)
    first_o, first_h1, first_h2, first_residue = waters[0]

    ion_atoms = [topology_atom_by_index(topology, idx) for idx in ion_indices]
    o_atom = topology_atom_by_index(topology, first_o)
    h1_atom = topology_atom_by_index(topology, first_h1)
    h2_atom = topology_atom_by_index(topology, first_h2)

    atom_indices_for_lj = {
        "ion1": ion_indices[0],
        "ion2": ion_indices[1],
        "water_O": first_o,
        "water_H1": first_h1,
        "water_H2": first_h2,
    }

    lj_type = {}
    pair_lj = {}
    if ljforce is not None:
        for key, idx in atom_indices_for_lj.items():
            lj_type[key] = custom_force_particle_type(ljforce, idx)
        pairs = [
            ("ion1-ion1", "ion1", "ion1"),
            ("ion2-ion2", "ion2", "ion2"),
            ("ion1-ion2", "ion1", "ion2"),
            ("ion1-water_O", "ion1", "water_O"),
            ("ion1-water_H", "ion1", "water_H1"),
            ("ion2-water_O", "ion2", "water_O"),
            ("ion2-water_H", "ion2", "water_H1"),
            ("water_O-water_O", "water_O", "water_O"),
            ("water_O-water_H", "water_O", "water_H1"),
            ("water_H-water_H", "water_H1", "water_H1"),
        ]
        for label, a, b in pairs:
            pair_lj[label] = lj_pair_from_acoef_bcoef(ljforce, lj_type[a], lj_type[b])

    atom_triplet = [first_o, first_h1, first_h2]
    bonds = bond_terms_for_atoms(system, atom_triplet)
    angles = angle_terms_for_atoms(system, atom_triplet)
    constraints = constraint_terms_for_atoms(system, atom_triplet)
    geom = geometry_from_positions(centered_positions, first_o, first_h1, first_h2)
    ion_distance = distance_between_atoms_A(centered_positions, ion_indices[0], ion_indices[1])

    out_path = out_dir / "forcefield_parameters.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Force-field parameter report\n")
        f.write("============================\n\n")

        f.write("Force field files:\n")
        for ff in ff_files:
            f.write(f"  {ff}\n")
        f.write("\n")

        f.write("OpenMM System force list:\n")
        for i, force in enumerate(system.getForces()):
            f.write(f"  {i} {force.__class__.__name__} {force_name(force)}\n")
            if hasattr(force, "getEnergyFunction"):
                energy = force.getEnergyFunction().replace("\n", " ")
                f.write(f"    energy_function = {energy[:300]}\n")
            if isinstance(force, CustomNonbondedForce):
                names = [force.getPerParticleParameterName(j) for j in range(force.getNumPerParticleParameters())]
                f.write(f"    per_particle_parameters = {' '.join(names)}\n")
                tab_names = [force.getTabulatedFunctionName(j) for j in range(force.getNumTabulatedFunctions())]
                f.write(f"    tabulated_functions = {' '.join(tab_names)}\n")
        f.write("\n")

        f.write("Important CHARMM/OpenMM note:\n")
        f.write("  In this CHARMM/OpenMM System, electrostatic charges are in NonbondedForce.\n")
        f.write("  The sigma/epsilon values in NonbondedForce are placeholders for this CHARMM setup.\n")
        f.write("  The actual CHARMM LJ/vdW terms are in CustomNonbondedForce named LennardJones.\n")
        f.write("  This report therefore reads charges from NonbondedForce and LJ from LennardJones.\n\n")

        f.write("MD settings:\n")
        for key, val in md_settings.items():
            f.write(f"  {key} = {val}\n")
        f.write("\n")

        f.write("Ion pair\n")
        f.write("--------\n")
        f.write(f"pair = {pair[0]} {pair[1]}\n")
        f.write(f"ion_distance_A = {ion_distance:.10f}\n")
        for n, (label, idx, atom, mass0) in enumerate(zip(pair, ion_indices, ion_atoms, original_ion_masses_da), start=1):
            f.write(f"ion{n}_label = {label}\n")
            f.write(f"ion{n}_atom_index = {idx}\n")
            f.write(f"ion{n}_residue_name = {atom.residue.name}\n")
            f.write(f"ion{n}_atom_name = {atom.name}\n")
            f.write(f"ion{n}_element = {atom.element}\n")
            f.write(f"ion{n}_original_mass_Da = {mass0:.10f}\n")
            f.write(f"ion{n}_MD_mass_Da = {mass_da(system, idx):.10f}\n")
            f.write(f"ion{n}_charge_from_NonbondedForce_e = {charge_e_from_nonbonded(nbforce, idx):.10f}\n")
            if ljforce is not None:
                f.write(f"ion{n}_LJ_particle_type_index = {lj_type[f'ion{n}']}\n")
        if ljforce is not None:
            write_lj_pair_section(f, "ion1-ion1 LJ from CustomNonbondedForce", pair_lj["ion1-ion1"])
            write_lj_pair_section(f, "ion2-ion2 LJ from CustomNonbondedForce", pair_lj["ion2-ion2"])
            write_lj_pair_section(f, "ion1-ion2 LJ from CustomNonbondedForce", pair_lj["ion1-ion2"])
            write_lj_pair_section(f, "ion1-water_O LJ from CustomNonbondedForce", pair_lj["ion1-water_O"])
            write_lj_pair_section(f, "ion1-water_H LJ from CustomNonbondedForce", pair_lj["ion1-water_H"])
            write_lj_pair_section(f, "ion2-water_O LJ from CustomNonbondedForce", pair_lj["ion2-water_O"])
            write_lj_pair_section(f, "ion2-water_H LJ from CustomNonbondedForce", pair_lj["ion2-water_H"])
        else:
            f.write("LJ report: no CustomNonbondedForce LennardJones found.\n")
        f.write("\n")

        f.write("Water sample\n")
        f.write("------------\n")
        f.write(f"residue_name = {first_residue.name}\n")
        f.write(f"residue_id = {first_residue.id}\n")
        f.write(f"oxygen_atom_index = {first_o}\n")
        f.write(f"oxygen_atom_name = {o_atom.name}\n")
        f.write(f"hydrogen1_atom_index = {first_h1}\n")
        f.write(f"hydrogen1_atom_name = {h1_atom.name}\n")
        f.write(f"hydrogen2_atom_index = {first_h2}\n")
        f.write(f"hydrogen2_atom_name = {h2_atom.name}\n\n")

        f.write("Water atom masses and charges:\n")
        for label, idx, atom in [("O", first_o, o_atom), ("H1", first_h1, h1_atom), ("H2", first_h2, h2_atom)]:
            f.write(
                f"  {label}: atom={atom.name}, index={idx}, "
                f"mass_Da={mass_da(system, idx):.10f}, "
                f"charge_e={charge_e_from_nonbonded(nbforce, idx):.10f}\n"
            )
        f.write("\n")

        if ljforce is not None:
            f.write("Water/ion LJ particle type indices from CustomNonbondedForce:\n")
            for key in ["ion1", "ion2", "water_O", "water_H1", "water_H2"]:
                f.write(f"  {key} = {lj_type[key]}\n")
            f.write("\n")
            write_lj_pair_section(f, "water_O-water_O LJ from CustomNonbondedForce", pair_lj["water_O-water_O"])
            write_lj_pair_section(f, "water_O-water_H LJ from CustomNonbondedForce", pair_lj["water_O-water_H"])
            write_lj_pair_section(f, "water_H-water_H LJ from CustomNonbondedForce", pair_lj["water_H-water_H"])

        f.write("Water intramolecular geometry from centered PDB:\n")
        f.write(f"  O-H1_A = {geom['OH1_A']:.10f}\n")
        f.write(f"  O-H2_A = {geom['OH2_A']:.10f}\n")
        f.write(f"  H1-H2_A = {geom['HH_A']:.10f}\n")
        f.write(f"  H-O-H_deg = {geom['HOH_deg']:.10f}\n\n")

        f.write("Water constraints actually present in OpenMM System:\n")
        if constraints:
            for c in constraints:
                f.write(f"  atoms {c['p1']}-{c['p2']}: distance_A = {c['distance_A']:.10f}\n")
        else:
            f.write("  none found for first water molecule\n")
        f.write("\n")

        f.write("Water HarmonicBondForce terms present in OpenMM System for first water:\n")
        if bonds:
            for b in bonds:
                f.write(
                    f"  atoms {b['p1']}-{b['p2']}: length_A={b['length_A']:.10f}, "
                    f"k_OpenMM_kJ_mol_nm2={b['k_OpenMM_kJ_mol_nm2']:.10f}, "
                    f"k_OpenMM_kcal_mol_A2={b['k_OpenMM_kcal_mol_A2']:.10f}, "
                    f"k_CHARMM_style_kcal_mol_A2={b['k_CHARMM_style_kcal_mol_A2']:.10f}\n"
                )
        else:
            f.write("  none found for first water molecule\n")
        f.write("\n")

        f.write("Water HarmonicAngleForce terms present in OpenMM System for first water:\n")
        if angles:
            for a in angles:
                f.write(
                    f"  atoms {a['p1']}-{a['p2']}-{a['p3']}: theta0_deg={a['theta0_deg']:.10f}, "
                    f"k_OpenMM_kJ_mol_rad2={a['k_OpenMM_kJ_mol_rad2']:.10f}, "
                    f"k_OpenMM_kcal_mol_rad2={a['k_OpenMM_kcal_mol_rad2']:.10f}, "
                    f"k_CHARMM_style_kcal_mol_rad2={a['k_CHARMM_style_kcal_mol_rad2']:.10f}\n"
                )
        else:
            f.write("  none found for first water molecule\n")


def write_run_settings(out_dir: Path, args, ff_files: Sequence[str]):
    path = out_dir / "run_settings.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Pair-ion MD run settings\n")
        f.write("========================\n\n")
        f.write(f"input_dir = {args.input_dir}\n")
        f.write(f"pair = {args.pair[0]} {args.pair[1]}\n")
        f.write(f"pdb_pattern = {args.pdb_pattern}\n")
        f.write("force_field_files = " + " ".join(ff_files) + "\n")
        f.write(f"d_start = {args.d_start}\n")
        f.write(f"d_end = {args.d_end}\n")
        f.write(f"d_step = {args.d_step}\n")
        f.write(f"timestep_fs = {args.timestep_fs}\n")
        f.write(f"equil_ps = {args.equil_ps}\n")
        f.write(f"prod_ns = {args.prod_ns}\n")
        f.write(f"traj_interval_fs = {args.traj_interval_fs}\n")
        f.write(f"state_interval_ps = {args.state_interval_ps}\n")
        f.write(f"temperature_K = {args.temperature_K}\n")
        f.write(f"friction_per_ps = {args.friction_per_ps}\n")
        f.write("cutoff_A = disabled by NoCutoff\n")
        f.write("switch_A = disabled by NoCutoff\n")
        f.write(f"wall_radius_A = {args.wall_radius_a}\n")
        f.write(f"wall_k_kcal_mol_A2 = {args.wall_k_kcal_mol_A2}\n")
        f.write(f"platform = {args.platform}\n")
        f.write(f"precision = {args.precision}\n")
        f.write(f"test_short = {args.test_short}\n")


def write_d_info(d_dir: Path, args, pdb_path: Path, d: int, ion_indices: Sequence[int], ion_distance_A: float, equil_steps: int, prod_steps: int, dcd_interval: int, state_interval: int):
    path = d_dir / "d_info.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Distance-state information\n")
        f.write("==========================\n\n")
        f.write(f"pair = {args.pair[0]} {args.pair[1]}\n")
        f.write(f"input_pdb = {pdb_path}\n")
        f.write(f"d_label = {d_label(d)}\n")
        f.write(f"requested_d_A = {d}\n")
        f.write(f"actual_ion_distance_A = {ion_distance_A:.10f}\n")
        f.write(f"ion_atom_indices = {ion_indices[0]} {ion_indices[1]}\n")
        f.write(f"equil_steps = {equil_steps}\n")
        f.write(f"prod_steps = {prod_steps}\n")
        f.write(f"dcd_interval_steps = {dcd_interval}\n")
        f.write(f"state_interval_steps = {state_interval}\n")
        f.write(f"timestep_fs = {args.timestep_fs}\n")
        f.write("cutoff_A = disabled by NoCutoff\n")
        f.write("switch_A = disabled by NoCutoff\n")
        f.write(f"wall_radius_A = {args.wall_radius_a}\n")
        f.write(f"wall_k_kcal_mol_A2 = {args.wall_k_kcal_mol_A2}\n")
        f.write("system_xml = system.xml\n")


# ------------------------- OpenMM platform -------------------------


def get_platform_and_properties(args):
    if args.platform.lower() == "auto":
        return None, {}
    platform = Platform.getPlatformByName(args.platform)
    props = {}
    if args.platform.upper() in {"CUDA", "OPENCL"}:
        if args.precision:
            props["Precision"] = args.precision
        if args.device_index is not None:
            props["DeviceIndex"] = str(args.device_index)
    return platform, props


# ------------------------- main workflow -------------------------


def pair_filename(input_dir: Path, pair: Sequence[str], d: int, pattern: str) -> Path:
    return input_dir / pattern.format(ion1=pair[0], ion2=pair[1], d=d)


def run_one_distance(args, ff, ff_files: Sequence[str], d: int, platform, props) -> Dict[str, object]:
    input_dir = Path(args.input_dir)
    pdb_path = pair_filename(input_dir, args.pair, d, args.pdb_pattern)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {pdb_path}")

    label = d_label(d)
    d_dir = Path(args.out_dir) / label
    d_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Pair {args.pair[0]} {args.pair[1]}, {label}, input={pdb_path.name} =====")

    pdb = PDBFile(str(pdb_path))
    ion_indices = find_pair_ion_atoms(pdb.topology, args.pair)
    water_oxygen_indices, waters = find_water_atoms(pdb.topology)
    centered_positions = recenter_positions_on_pair_center(pdb.positions, ion_indices)
    ion_distance_A = distance_between_atoms_A(centered_positions, ion_indices[0], ion_indices[1])

    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        rigidWater=True,
        removeCMMotion=False,
    )
    set_no_cutoff(system)

    nbforce = get_nonbonded_force(system)
    original_masses = [mass_da(system, idx) for idx in ion_indices]
    original_charges = [charge_e_from_nonbonded(nbforce, idx) for idx in ion_indices]

    n_wall = add_spherical_wall(system, water_oxygen_indices, args.wall_radius_a, args.wall_k_kcal_mol_A2)

    # Fix both ions at their PDB positions while preserving all interactions.
    for idx in ion_indices:
        system.setParticleMass(idx, 0.0 * dalton)

    md_settings = {
        "timestep_fs": args.timestep_fs,
        "equil_ps": args.equil_ps,
        "prod_ns": args.prod_ns,
        "traj_interval_fs": args.traj_interval_fs,
        "state_interval_ps": args.state_interval_ps,
        "temperature_K": args.temperature_K,
        "friction_per_ps": args.friction_per_ps,
        "cutoff_A": "disabled by NoCutoff",
        "switch_A": "disabled by NoCutoff",
        "wall_radius_A": args.wall_radius_a,
        "wall_k_kcal_mol_A2": args.wall_k_kcal_mol_A2,
        "constraints": "HBonds",
        "rigidWater": "True",
        "ion_masses_set_to_zero": "True",
        "original_ion_masses_Da": " ".join(f"{x:.10f}" for x in original_masses),
        "original_ion_charges_e": " ".join(f"{x:.10f}" for x in original_charges),
        "nonbondedMethod": "NoCutoff",
        "requested_d_A": d,
        "actual_ion_distance_A": f"{ion_distance_A:.10f}",
        "waters_detected": len(waters),
        "water_oxygens_with_wall": n_wall,
    }

    write_forcefield_parameter_report(
        out_dir=d_dir,
        topology=pdb.topology,
        system=system,
        centered_positions=centered_positions,
        pair=args.pair,
        ion_indices=ion_indices,
        original_ion_masses_da=original_masses,
        ff_files=ff_files,
        md_settings=md_settings,
    )

    if not args.no_system_xml:
        with (d_dir / "system.xml").open("w", encoding="utf-8") as f:
            f.write(XmlSerializer.serialize(system))

    timestep = args.timestep_fs * femtosecond
    integrator = LangevinMiddleIntegrator(
        args.temperature_K * kelvin,
        args.friction_per_ps / picosecond,
        timestep,
    )

    if platform is None:
        simulation = Simulation(pdb.topology, system, integrator)
        platform_name = simulation.context.getPlatform().getName()
    else:
        simulation = Simulation(pdb.topology, system, integrator, platform, props)
        platform_name = platform.getName()

    simulation.context.setPositions(centered_positions)

    equil_steps = steps_from_time(args.equil_ps, "ps", args.timestep_fs)
    prod_steps = steps_from_time(args.prod_ns, "ns", args.timestep_fs)
    dcd_interval = steps_from_time(args.traj_interval_fs, "fs", args.timestep_fs)
    state_interval = steps_from_time(args.state_interval_ps, "ps", args.timestep_fs)

    write_d_info(d_dir, args, pdb_path, d, ion_indices, ion_distance_A, equil_steps, prod_steps, dcd_interval, state_interval)

    print(f"  platform: {platform_name}")
    print(f"  ion indices: {ion_indices[0]}, {ion_indices[1]}")
    print(f"  ion charges from FF: {original_charges[0]:.3f}, {original_charges[1]:.3f} e")
    print(f"  ion distance after centering: {ion_distance_A:.6f} A")
    print(f"  waters detected: {len(waters)}")
    print(f"  water O spherical wall particles: {n_wall}")

    print(f"  Minimizing energy, maxIterations={args.minimize_iterations} ...")
    simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
    simulation.context.setVelocitiesToTemperature(args.temperature_K * kelvin)

    simulation.reporters = []
    simulation.reporters.append(
        StateDataReporter(
            str(d_dir / "equil_state.csv"),
            state_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            separator=",",
        )
    )
    print(f"  Equilibration: {args.equil_ps} ps ({equil_steps} steps)")
    simulation.step(equil_steps)

    simulation.reporters = []
    simulation.reporters.append(DCDReporter(str(d_dir / "traj.dcd"), dcd_interval))
    simulation.reporters.append(
        StateDataReporter(
            str(d_dir / "state.csv"),
            state_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=prod_steps,
            separator=",",
        )
    )
    print(f"  Production: {args.prod_ns} ns ({prod_steps} steps)")
    simulation.step(prod_steps)

    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    final_positions = state.getPositions()
    with (d_dir / "final.pdb").open("w", encoding="utf-8") as f:
        PDBFile.writeFile(pdb.topology, final_positions, f, keepIds=True)
    simulation.saveCheckpoint(str(d_dir / "checkpoint.chk"))

    pot = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
    kin = state.getKineticEnergy().value_in_unit(kilocalorie_per_mole)
    print(f"  Completed {label}. Final PE = {pot:.6f} kcal/mol")

    return {
        "d_label": label,
        "requested_d_A": d,
        "actual_ion_distance_A": ion_distance_A,
        "input_pdb": str(pdb_path),
        "ion1_index": ion_indices[0],
        "ion2_index": ion_indices[1],
        "ion1_charge_e": original_charges[0],
        "ion2_charge_e": original_charges[1],
        "waters": len(waters),
        "equil_steps": equil_steps,
        "prod_steps": prod_steps,
        "dcd_interval_steps": dcd_interval,
        "final_potential_kcal_mol": pot,
        "final_kinetic_kcal_mol": kin,
        "folder": str(d_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Pair-ion MD in CHARMM TIP3P water cluster.")
    parser.add_argument("--pair", nargs=2, required=True, choices=["K", "Cl"], help="Ion pair, e.g. --pair K K, --pair Cl Cl, --pair K Cl")
    parser.add_argument("--input_dir", required=True, help="Directory containing pair PDB files")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--ff", nargs="+", default=["charmm36.xml", "charmm36/water.xml"], help="OpenMM CHARMM XML files")
    parser.add_argument(
        "--pdb_pattern",
        default="pair_{ion1}_{ion2}_d{d:02d}A_sel21A_1117w.pdb",
        help="Input PDB filename pattern. Available fields: {ion1}, {ion2}, {d:02d}",
    )
    parser.add_argument("--d_start", type=int, default=1)
    parser.add_argument("--d_end", type=int, default=8)
    parser.add_argument("--d_step", type=int, default=1)

    parser.add_argument("--timestep_fs", type=float, default=2.0)
    parser.add_argument("--equil_ps", type=float, default=400.0)
    parser.add_argument("--prod_ns", type=float, default=10.0)
    parser.add_argument("--traj_interval_fs", type=float, default=100.0)
    parser.add_argument("--state_interval_ps", type=float, default=0.2)
    parser.add_argument("--temperature_K", type=float, default=300.0)
    parser.add_argument("--friction_per_ps", type=float, default=1.0)
    parser.add_argument("--cutoff_a", type=float, default=12.0)
    parser.add_argument("--switch_a", type=float, default=10.0)
    parser.add_argument("--wall_radius_a", type=float, default=20.0)
    parser.add_argument("--wall_k_kcal_mol_A2", type=float, default=10.0)
    parser.add_argument("--minimize_iterations", type=int, default=10000)

    parser.add_argument("--platform", default="auto", choices=["auto", "CUDA", "OpenCL", "CPU"], help="OpenMM platform")
    parser.add_argument("--precision", default="mixed", choices=["single", "mixed", "double"], help="GPU precision")
    parser.add_argument("--device_index", default=None, help="CUDA/OpenCL device index")

    parser.add_argument("--test_short", action="store_true", help="Short test: equil 1 ps, production 2 ps")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    parser.add_argument("--no_system_xml", action="store_true", help="Do not write system.xml files")
    args = parser.parse_args()

    if args.test_short:
        args.equil_ps = 1.0
        args.prod_ns = 0.002
        args.traj_interval_fs = 100.0
        args.state_interval_ps = 0.2
        print("TEST MODE: equil_ps=1, prod_ps=2, traj_interval_fs=100")

    if args.d_step <= 0:
        raise ValueError("--d_step must be positive")
    if args.d_end < args.d_start:
        raise ValueError("--d_end must be >= --d_start")

    out_dir = Path(args.out_dir)
    ensure_clean_dir(out_dir, args.overwrite)

    ff = ForceField(*args.ff)
    ff_files = tuple(args.ff)
    print("Loaded CHARMM force field files:", ", ".join(ff_files))
    print("Input directory:", args.input_dir)
    print("Pair:", args.pair[0], args.pair[1])
    print("This script does not modify ion charges. It fixes ion positions by setting ion masses to zero.")

    write_run_settings(out_dir, args, ff_files)
    platform, props = get_platform_and_properties(args)

    print("\nMD settings")
    print(f"  timestep: {args.timestep_fs} fs")
    print(f"  equilibration: {args.equil_ps} ps")
    print(f"  production: {args.prod_ns} ns")
    print(f"  DCD interval: {args.traj_interval_fs} fs")
    print("  nonbonded method: NoCutoff")
    print("  nonbonded cutoff: disabled")
    print("  LJ switching: disabled")
    print(f"  spherical wall: R = {args.wall_radius_a} A, k = {args.wall_k_kcal_mol_A2} kcal/mol/A^2")
    print("  both ion masses set to zero: ion positions fixed, interactions retained")

    records = []
    for d in range(args.d_start, args.d_end + 1, args.d_step):
        rec = run_one_distance(args, ff, ff_files, d, platform, props)
        records.append(rec)

    summary_path = out_dir / "summary.csv"
    if records:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    print("\nDone.")
    print(f"Output directory: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
