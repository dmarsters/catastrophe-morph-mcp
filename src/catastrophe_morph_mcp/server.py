"""
Catastrophe Morph MCP Server - Phase 1A + Phase 2.6 + Phase 2.7 + Forced Orbit Enhanced
========================================================================================

Maps René Thom's seven elementary catastrophes to visual aesthetic parameters
with integrated trajectory computation, rhythmic composition, and attractor
visualization prompt generation.

Architecture:
  Layer 1: Pure taxonomy (catastrophe_theory.yaml) - 0 tokens
  Layer 2: Deterministic operations + trajectory dynamics + rhythmic composition
           + attractor visualization prompts - 0 tokens  
  Layer 3: Claude synthesis - ~100-200 tokens

Phase 1A Enhancement:
  - RK4 trajectory integration between catastrophe types
  - Smooth aesthetic transitions through parameter space
  - Convergence validation and path efficiency metrics

Phase 2.6 Enhancement:
  - Rhythmic composition between catastrophe morphologies
  - Oscillatory patterns: sinusoidal, triangular, square wave
  - 5 curated presets for catastrophe temporal patterns
  - Bifurcation cycles and stability oscillations

Phase 2.7 Enhancement:
  - Attractor visualization prompt generation from 5D coordinates
  - 7 visual types derived from catastrophe olog vocabulary
  - Nearest-neighbor vocabulary extraction with optical properties
  - Composite, split-view, and sequence prompt modes
  - Preset attractor catalog from Tier 4D multi-domain discovery
  - Keyframe prompt extraction from rhythmic preset trajectories

Forced Orbit Integration (v1.2.0):
  - Production-ready phase-space integration for limit cycles
  - Perfect cycle closure with zero drift
  - Supports multi-domain compositional limit cycles
  - Compatible with aesthetic-dynamics-core v1.0.0+
"""

from fastmcp import FastMCP
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List, Optional
import sys
import re

mcp = FastMCP("Catastrophe Morph")

# ============================================================================
# PHASE 1A + PHASE 2.6: Import aesthetic-dynamics-core (with graceful degradation)
# ============================================================================

try:
    from aesthetic_dynamics_core import (
        _integrate_trajectory_impl,
        _compute_gradient_field_impl,
        _analyze_convergence_impl,
        _validate_round_trip_impl,
        _integrate_forced_limit_cycle_single_domain_impl,
        _integrate_forced_limit_cycle_multi_domain_impl
    )
    DYNAMICS_AVAILABLE = True
    FORCED_ORBIT_AVAILABLE = True
except ImportError:
    # Graceful degradation if aesthetic-dynamics-core not installed
    DYNAMICS_AVAILABLE = False
    FORCED_ORBIT_AVAILABLE = False
    _integrate_trajectory_impl = None
    _compute_gradient_field_impl = None
    _analyze_convergence_impl = None
    _validate_round_trip_impl = None
    _integrate_forced_limit_cycle_single_domain_impl = None
    _integrate_forced_limit_cycle_multi_domain_impl = None

print("=== CATASTROPHE MORPH SERVER STARTING ===", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"Numpy available: {np is not None}", file=sys.stderr)
print(f"Dynamics available: {DYNAMICS_AVAILABLE}", file=sys.stderr)

# ============================================================================
# Server Configuration
# ============================================================================

SERVER_VERSION = "1.2.0-phase2.7-attractor-viz"
VALIDATION_DATE = "2026-02-06"

# ============================================================================
# LOAD TAXONOMY FROM OLOG
# ============================================================================

OLOG_PATH = Path(__file__).parent / "ologs" / "catastrophe_theory.yaml"

def load_olog():
    """Load the catastrophe theory taxonomy from YAML."""
    with open(OLOG_PATH, "r") as f:
        return yaml.safe_load(f)

# Load at module level for performance
OLOG = load_olog()
CATASTROPHE_MORPHOLOGIES = OLOG["morphologies"]
INTENSITY_PROFILES = OLOG["intensity_profiles"]
EMPHASIS_MAPPING = OLOG["emphasis_mapping"]


# ============================================================================
# LAYER 1: Catastrophe State Definitions
# ============================================================================

# Define canonical states for the 7 elementary catastrophes
# Parameters represent position in the unified catastrophe morphospace

CATASTROPHE_STATES = {
    "fold": {
        "name": "Fold Catastrophe (A2)",
        "description": "Simple bifurcation - smooth to discontinuous",
        "coordinates": {
            "control_complexity": 0.0,    # 1 control parameter (simplest)
            "geometric_sharpness": 0.2,   # Smooth folds
            "surface_tension": 0.3,       # Low tension
            "optical_intensity": 0.4,     # Matte, diffuse
            "aesthetic_intensity": 0.5    # Moderate presence
        },
        "mathematical_properties": {
            "control_parameters": 1,
            "behavior_dimensions": 1
        }
    },
    
    "cusp": {
        "name": "Cusp Catastrophe (A3)",
        "description": "Sharp bifurcation with hysteresis",
        "coordinates": {
            "control_complexity": 0.25,   # 2 control parameters
            "geometric_sharpness": 0.9,   # Sharp, angular
            "surface_tension": 0.7,       # High tension
            "optical_intensity": 0.8,     # Specular, reflective
            "aesthetic_intensity": 0.7    # Strong presence
        },
        "mathematical_properties": {
            "control_parameters": 2,
            "behavior_dimensions": 1
        }
    },
    
    "swallowtail": {
        "name": "Swallowtail Catastrophe (A4)",
        "description": "Complex topology with multiple stable states",
        "coordinates": {
            "control_complexity": 0.5,    # 3 control parameters
            "geometric_sharpness": 0.4,   # Flowing curves
            "surface_tension": 0.5,       # Moderate tension
            "optical_intensity": 0.7,     # Iridescent
            "aesthetic_intensity": 0.8    # High presence
        },
        "mathematical_properties": {
            "control_parameters": 3,
            "behavior_dimensions": 1
        }
    },
    
    "butterfly": {
        "name": "Butterfly Catastrophe (A5)",
        "description": "Four-dimensional control with compromise pocket",
        "coordinates": {
            "control_complexity": 0.75,   # 4 control parameters
            "geometric_sharpness": 0.6,   # Symmetric structures
            "surface_tension": 0.6,       # Balanced tension
            "optical_intensity": 0.85,    # Structural color
            "aesthetic_intensity": 0.9    # Very strong presence
        },
        "mathematical_properties": {
            "control_parameters": 4,
            "behavior_dimensions": 1
        }
    },
    
    "elliptic_umbilic": {
        "name": "Elliptic Umbilic (D4-)",
        "description": "Wave-like bifurcation with three-fold symmetry",
        "coordinates": {
            "control_complexity": 0.5,    # 3 control parameters
            "geometric_sharpness": 0.3,   # Wave undulation
            "surface_tension": 0.4,       # Oscillating tension
            "optical_intensity": 0.65,    # Metallic waves
            "aesthetic_intensity": 0.75   # Strong presence
        },
        "mathematical_properties": {
            "control_parameters": 3,
            "behavior_dimensions": 2
        }
    },
    
    "hyperbolic_umbilic": {
        "name": "Hyperbolic Umbilic (D4+)",
        "description": "Saddle-point bifurcation with stress concentration",
        "coordinates": {
            "control_complexity": 0.5,    # 3 control parameters
            "geometric_sharpness": 0.7,   # Tension lines
            "surface_tension": 0.9,       # Maximum tension
            "optical_intensity": 0.7,     # Metallic stress
            "aesthetic_intensity": 0.85   # Very strong presence
        },
        "mathematical_properties": {
            "control_parameters": 3,
            "behavior_dimensions": 2
        }
    },
    
    "parabolic_umbilic": {
        "name": "Parabolic Umbilic (D5)",
        "description": "Mushroom catastrophe - most complex elementary form",
        "coordinates": {
            "control_complexity": 1.0,    # 4 control parameters (most complex)
            "geometric_sharpness": 0.5,   # Organic curves
            "surface_tension": 0.55,      # Moderate-high tension
            "optical_intensity": 0.6,     # Satin finish
            "aesthetic_intensity": 0.95   # Maximum presence
        },
        "mathematical_properties": {
            "control_parameters": 4,
            "behavior_dimensions": 2
        }
    }
}

# Parameter names in order (for trajectory computation)
PARAMETER_NAMES = [
    "control_complexity",
    "geometric_sharpness", 
    "surface_tension",
    "optical_intensity",
    "aesthetic_intensity"
]

# Parameter bounds (all normalized to [0, 1])
PARAMETER_BOUNDS = [0.0, 1.0]


# ============================================================================
# LAYER 1: Pure Taxonomy Tools
# ============================================================================

@mcp.tool()
def get_catastrophe_types() -> dict:
    """
    List all seven elementary catastrophes with descriptions.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Returns:
        Dictionary mapping catastrophe IDs to their properties
    """
    taxonomy = load_taxonomy()
    return {
        catastrophe_id: {
            "mathematical_name": data["mathematical_name"],
            "control_parameters": data["control_parameters"],
            "description": data["description"],
            "keywords": data["keywords"]
        }
        for catastrophe_id, data in taxonomy["morphologies"].items()
    }


@mcp.tool()
def get_catastrophe_specifications(catastrophe_id: str) -> dict:
    """
    Get complete visual specifications for a catastrophe type.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Args:
        catastrophe_id: One of the 7 catastrophe types
    
    Returns:
        Complete visual vocabulary and properties
    """
    taxonomy = load_taxonomy()
    
    if catastrophe_id not in taxonomy["morphologies"]:
        return {
            "error": f"Unknown catastrophe: {catastrophe_id}",
            "available": list(taxonomy["morphologies"].keys())
        }
    
    return taxonomy["morphologies"][catastrophe_id]


# ============================================================================
# LAYER 2: Deterministic Parameter Mapping
# ============================================================================

def _map_catastrophe_parameters_impl(
    catastrophe_id: str,
    intensity: str = "moderate",
    emphasis: str = "surface"
) -> dict:
    """Core implementation of parameter mapping."""
    taxonomy = load_taxonomy()
    
    if catastrophe_id not in taxonomy["morphologies"]:
        return {
            "error": f"Unknown catastrophe: {catastrophe_id}",
            "available": list(taxonomy["morphologies"].keys())
        }
    
    catastrophe_data = taxonomy["morphologies"][catastrophe_id]
    intensity_profile = taxonomy["intensity_profiles"][intensity]
    emphasis_mapping = taxonomy["emphasis_mapping"][emphasis]
    
    # Get canonical coordinates
    canonical_coords = CATASTROPHE_STATES[catastrophe_id]["coordinates"]
    
    # Apply intensity scaling
    scaled_coords = canonical_coords.copy()
    scaled_coords["aesthetic_intensity"] = intensity_profile["optical_influence"]
    
    return {
        "catastrophe_id": catastrophe_id,
        "catastrophe_name": catastrophe_data["mathematical_name"],
        "coordinates": scaled_coords,
        "visual_vocabulary": catastrophe_data["visual_vocabulary"],
        "intensity_profile": intensity_profile,
        "emphasis": emphasis_mapping,
        "color_associations": catastrophe_data["color_associations"],
        "optical_properties": catastrophe_data["optical_properties"]
    }


@mcp.tool()
def map_catastrophe_parameters(
    catastrophe_id: str,
    intensity: str = "moderate",
    emphasis: str = "surface"
) -> dict:
    """
    Map catastrophe type to visual parameters.
    
    Layer 2: Deterministic operation (0 tokens)
    
    Args:
        catastrophe_id: Which catastrophe (fold, cusp, swallowtail, etc.)
        intensity: "subtle", "moderate", or "dramatic"
        emphasis: "surface", "form", "light", "movement", or "edges"
    
    Returns:
        Complete parameter set for visual synthesis
    """
    return _map_catastrophe_parameters_impl(catastrophe_id, intensity, emphasis)


# ============================================================================
# LAYER 2: TRAJECTORY DYNAMICS (Phase 1A)
# ============================================================================

def _compute_trajectory_between_catastrophes_impl(
    start_catastrophe_id: str,
    end_catastrophe_id: str,
    num_steps: int = 30,
    return_analysis: bool = True
) -> dict:
    """
    Core implementation of trajectory computation between catastrophe types.
    
    This is the Phase 1A enhancement - uses aesthetic-dynamics-core
    to compute smooth RK4 trajectories through catastrophe morphospace.
    """
    if not DYNAMICS_AVAILABLE:
        return {
            "error": "aesthetic-dynamics-core not installed",
            "message": "Install with: pip install aesthetic-dynamics-core --break-system-packages",
            "fallback": "Use map_catastrophe_parameters for individual states"
        }
    
    # Validate catastrophe IDs
    if start_catastrophe_id not in CATASTROPHE_STATES:
        return {
            "error": f"Unknown start catastrophe: {start_catastrophe_id}",
            "available": list(CATASTROPHE_STATES.keys())
        }
    
    if end_catastrophe_id not in CATASTROPHE_STATES:
        return {
            "error": f"Unknown end catastrophe: {end_catastrophe_id}",
            "available": list(CATASTROPHE_STATES.keys())
        }
    
    # Get catastrophe states
    start_state = CATASTROPHE_STATES[start_catastrophe_id]
    end_state = CATASTROPHE_STATES[end_catastrophe_id]
    
    start_coords = start_state["coordinates"]
    end_coords = end_state["coordinates"]
    
    # Check for same catastrophe
    if start_catastrophe_id == end_catastrophe_id:
        return {
            "warning": "Start and end are the same catastrophe",
            "suggestion": "Try different catastrophe types for meaningful trajectory"
        }
    
    # Compute trajectory using aesthetic-dynamics-core
    trajectory_result = _integrate_trajectory_impl(
        start_state=start_coords,
        target_state=end_coords,
        parameter_names=PARAMETER_NAMES,
        num_steps=num_steps,
        bounds=PARAMETER_BOUNDS,
        convergence_threshold=0.01
    )
    
    # Compute mathematical distance (weighted by parameter importance)
    # Weight control_complexity higher as it's the defining characteristic
    weights = {
        "control_complexity": 2.0,
        "geometric_sharpness": 1.0,
        "surface_tension": 1.0,
        "optical_intensity": 0.8,
        "aesthetic_intensity": 0.5
    }
    
    def weighted_distance(state1, state2):
        return np.sqrt(sum(
            weights[p] * (state1[p] - state2[p])**2
            for p in PARAMETER_NAMES
        ))
    
    mathematical_distance = weighted_distance(start_coords, end_coords)
    
    # Prepare response
    response = {
        "start_catastrophe": {
            "id": start_catastrophe_id,
            "name": start_state["name"],
            "description": start_state["description"],
            "coordinates": start_coords,
            "control_parameters": start_state["mathematical_properties"]["control_parameters"]
        },
        "end_catastrophe": {
            "id": end_catastrophe_id,
            "name": end_state["name"],
            "description": end_state["description"],
            "coordinates": end_coords,
            "control_parameters": end_state["mathematical_properties"]["control_parameters"]
        },
        "trajectory": {
            "states": trajectory_result["trajectory"],
            "num_steps": trajectory_result["num_steps"],
            "parameter_names": PARAMETER_NAMES
        },
        "convergence": {
            "converged": trajectory_result["converged"],
            "convergence_step": trajectory_result["convergence_step"],
            "final_distance": trajectory_result["final_distance"],
            "convergence_threshold": 0.01
        },
        "path_metrics": {
            "geodesic_length": trajectory_result["path_length"],
            "euclidean_distance": trajectory_result["initial_distance"],
            "mathematical_distance": float(mathematical_distance),
            "path_efficiency": trajectory_result["initial_distance"] / trajectory_result["path_length"] if trajectory_result["path_length"] > 0 else 1.0
        },
        "transition_characteristics": {
            "control_parameter_change": abs(
                end_state["mathematical_properties"]["control_parameters"] - 
                start_state["mathematical_properties"]["control_parameters"]
            ),
            "complexity_direction": "increasing" if end_coords["control_complexity"] > start_coords["control_complexity"] else "decreasing",
            "sharpness_evolution": "sharpening" if end_coords["geometric_sharpness"] > start_coords["geometric_sharpness"] else "smoothing"
        },
        "dynamics_info": {
            "integration_method": "RK4 (Runge-Kutta 4th order)",
            "bounds": str(PARAMETER_BOUNDS),
            "cost": "0 tokens (pure Layer 2)"
        }
    }
    
    # Optional: Add convergence analysis
    if return_analysis and trajectory_result["converged"]:
        analysis = _analyze_convergence_impl(
            trajectory=trajectory_result["trajectory"],
            target_state=end_coords,
            parameter_names=PARAMETER_NAMES,
            threshold=0.01
        )
        
        response["convergence_analysis"] = {
            "monotonic_decrease": analysis["monotonic_decrease"],
            "oscillation_count": analysis["oscillation_count"],
            "convergence_rate": analysis["convergence_rate"],
            "distance_reduction": analysis["distance_reduction"]
        }
    
    return response


@mcp.tool()
def compute_trajectory_between_catastrophes(
    start_catastrophe_id: str,
    end_catastrophe_id: str,
    num_steps: int = 30,
    return_analysis: bool = True
) -> dict:
    """
    Compute smooth trajectory between two catastrophe types in morphospace.
    
    NEW PHASE 1A TOOL: Uses aesthetic-dynamics-core for zero-cost trajectory
    integration via RK4. Enables visualization of smooth aesthetic transitions
    through catastrophe theory parameter space.
    
    This answers questions like:
    - "What's the smoothest path from fold to butterfly?"
    - "How does geometric sharpness evolve from cusp to swallowtail?"
    - "What intermediate states exist between elliptic and hyperbolic umbilics?"
    
    Args:
        start_catastrophe_id: Starting catastrophe ("fold", "cusp", etc.)
        end_catastrophe_id: Target catastrophe
        num_steps: Number of integration steps (default: 30)
        return_analysis: Include convergence analysis (default: True)
    
    Returns:
        Dictionary with trajectory data, convergence metrics, and transition analysis
    
    Cost: 0 tokens (pure Layer 2 deterministic computation)
    
    Example:
        >>> compute_trajectory_between_catastrophes(
        ...     "fold",
        ...     "cusp",
        ...     num_steps=20
        ... )
        {
            "start_catastrophe": {"name": "Fold Catastrophe (A2)", ...},
            "end_catastrophe": {"name": "Cusp Catastrophe (A3)", ...},
            "trajectory": [...],  # 21 intermediate states
            "converged": true,
            "path_metrics": {
                "geodesic_length": 0.847,
                "euclidean_distance": 0.831,
                "path_efficiency": 0.981
            },
            "transition_characteristics": {
                "control_parameter_change": 1,
                "complexity_direction": "increasing",
                "sharpness_evolution": "sharpening"
            }
        }
    """
    return _compute_trajectory_between_catastrophes_impl(
        start_catastrophe_id, end_catastrophe_id, num_steps, return_analysis
    )


# ============================================================================
# LAYER 2: Distance and Similarity Metrics
# ============================================================================

@mcp.tool()
def compute_catastrophe_distance(
    catastrophe_id_1: str,
    catastrophe_id_2: str,
    metric: str = "mathematical"
) -> dict:
    """
    Compute distance between two catastrophe types.
    
    Layer 2: Pure distance computation (0 tokens)
    
    Args:
        catastrophe_id_1: First catastrophe
        catastrophe_id_2: Second catastrophe
        metric: "mathematical" (weighted) or "euclidean" (unweighted)
    
    Returns:
        Distance value and components
    """
    if catastrophe_id_1 not in CATASTROPHE_STATES:
        return {"error": f"Unknown catastrophe: {catastrophe_id_1}"}
    if catastrophe_id_2 not in CATASTROPHE_STATES:
        return {"error": f"Unknown catastrophe: {catastrophe_id_2}"}
    
    coords1 = CATASTROPHE_STATES[catastrophe_id_1]["coordinates"]
    coords2 = CATASTROPHE_STATES[catastrophe_id_2]["coordinates"]
    
    if metric == "mathematical":
        # Weighted distance emphasizing mathematical properties
        weights = {
            "control_complexity": 2.0,
            "geometric_sharpness": 1.0,
            "surface_tension": 1.0,
            "optical_intensity": 0.8,
            "aesthetic_intensity": 0.5
        }
        distance = np.sqrt(sum(
            weights[p] * (coords1[p] - coords2[p])**2
            for p in PARAMETER_NAMES
        ))
    else:
        # Euclidean distance
        distance = np.sqrt(sum(
            (coords1[p] - coords2[p])**2
            for p in PARAMETER_NAMES
        ))
    
    # Compute per-parameter differences
    differences = {
        p: abs(coords1[p] - coords2[p])
        for p in PARAMETER_NAMES
    }
    
    return {
        "catastrophe_1": catastrophe_id_1,
        "catastrophe_2": catastrophe_id_2,
        "distance": float(distance),
        "metric": metric,
        "parameter_differences": differences,
        "most_different_parameter": max(differences, key=differences.get)
    }


# ============================================================================
# LAYER 3: Visualization Context (for Claude synthesis)
# ============================================================================

@mcp.tool()
def prepare_transition_visualization(
    trajectory_result: dict,
    visualization_style: str = "parameter_evolution"
) -> dict:
    """
    Prepare structured context for Claude to synthesize trajectory visualization.
    
    Layer 3 Interface: Assembles deterministic parameters for creative synthesis.
    
    Args:
        trajectory_result: Output from compute_trajectory_between_catastrophes
        visualization_style: "parameter_evolution", "morphospace_path", or "phase_diagram"
    
    Returns:
        Complete synthesis context with data and instructions
    
    Cost: 0 tokens (deterministic) + ~100-200 tokens (Claude synthesis)
    """
    if "error" in trajectory_result:
        return trajectory_result
    
    start = trajectory_result["start_catastrophe"]
    end = trajectory_result["end_catastrophe"]
    trajectory = trajectory_result["trajectory"]["states"]
    
    context = {
        "visualization_type": visualization_style,
        "data": {
            "start_point": {
                "name": start["name"],
                "coordinates": start["coordinates"]
            },
            "end_point": {
                "name": end["name"],
                "coordinates": end["coordinates"]
            },
            "trajectory": trajectory,
            "parameter_names": PARAMETER_NAMES
        },
        "metrics": trajectory_result["path_metrics"],
        "convergence": trajectory_result["convergence"],
        "synthesis_instructions": {}
    }
    
    if visualization_style == "parameter_evolution":
        context["synthesis_instructions"] = {
            "format": "Line plot with one line per parameter",
            "x_axis": "Step number",
            "y_axis": "Parameter value [0, 1]",
            "title": f"Parameter Evolution: {start['name']} → {end['name']}",
            "highlight": "Show where each parameter changes most rapidly",
            "annotations": [
                f"Start: {start['name']}",
                f"End: {end['name']}",
                f"Steps: {len(trajectory)}",
                f"Converged: {trajectory_result['convergence']['converged']}"
            ]
        }
    
    elif visualization_style == "morphospace_path":
        context["synthesis_instructions"] = {
            "format": "2D projection of 5D trajectory (PCA or t-SNE)",
            "points": "Show all 7 catastrophe types as reference",
            "path": "Draw trajectory from start to end",
            "markers": "Mark start (green), end (red), intermediate (gray)",
            "title": f"Morphospace Path: {start['name']} → {end['name']}",
            "annotations": [
                "Path shows smooth transition through parameter space",
                f"Efficiency: {trajectory_result['path_metrics']['path_efficiency']:.2f}"
            ]
        }
    
    elif visualization_style == "phase_diagram":
        context["synthesis_instructions"] = {
            "format": "2D heatmap of morphospace",
            "axes": ["control_complexity", "geometric_sharpness"],
            "overlays": [
                "7 catastrophe types as points",
                "Trajectory as line",
                "Gradient field as arrows"
            ],
            "title": "Catastrophe Morphospace Phase Diagram",
            "colormap": "Viridis (complexity) or Plasma (intensity)"
        }
    
    return context


# ============================================================================
# PHASE 2.6: RHYTHMIC COMPOSITION
# ============================================================================

# Oscillation generators (compact versions from microscopy/nuclear)
def _gen_sin_osc(steps, cycles, phase=0.0):
    t = np.linspace(0, 2*np.pi*cycles, steps)
    return 0.5 * (1 + np.sin(t + phase*2*np.pi))

def _gen_tri_osc(steps, cycles, phase=0.0):
    t = np.linspace(0, cycles, steps) + phase
    t_cycle = t % 1.0
    return np.where(t_cycle < 0.5, 2*t_cycle, 2*(1-t_cycle))

def _gen_sqr_osc(steps, cycles, phase=0.0):
    t = np.linspace(0, cycles, steps)
    t = (t + phase) % 1.0
    return np.where(t < 0.5, 0.0, 1.0)


# Catastrophe Rhythmic Presets
CATASTROPHE_RHYTHMIC_PRESETS = {
    "bifurcation_cycle": {
        "description": "Oscillation between smooth and sharp transitions",
        "state_a_id": "fold",
        "state_b_id": "cusp",
        "oscillation_pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 20,
        "use_case": "Smooth to sharp bifurcation cycling",
        "visual_effect": "Undulating between continuous and discontinuous forms"
    },
    "complexity_scan": {
        "description": "Control complexity variation (simple ↔ complex)",
        "state_a_id": "fold",
        "state_b_id": "parabolic_umbilic",
        "oscillation_pattern": "triangular",
        "num_cycles": 3,
        "steps_per_cycle": 25,
        "use_case": "Linear scanning through complexity space",
        "visual_effect": "Ramping from minimal to maximum control parameters"
    },
    "umbilic_alternation": {
        "description": "Wave vs saddle point oscillation",
        "state_a_id": "elliptic_umbilic",
        "state_b_id": "hyperbolic_umbilic",
        "oscillation_pattern": "sinusoidal",
        "num_cycles": 5,
        "steps_per_cycle": 18,
        "use_case": "Alternating between wave and tension patterns",
        "visual_effect": "Smooth transitions between undulation and stress concentration"
    },
    "symmetry_pulse": {
        "description": "Symmetry order variation",
        "state_a_id": "cusp",
        "state_b_id": "butterfly",
        "oscillation_pattern": "square",
        "num_cycles": 4,
        "steps_per_cycle": 15,
        "use_case": "Abrupt symmetry changes",
        "visual_effect": "Toggle between 2-fold and 4-fold symmetry"
    },
    "tension_breathing": {
        "description": "Surface tension rhythmic variation",
        "state_a_id": "fold",
        "state_b_id": "hyperbolic_umbilic",
        "oscillation_pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 22,
        "use_case": "Tension oscillation for stress-field animation",
        "visual_effect": "Breathing between minimal and maximum surface tension"
    }
}


@mcp.tool()
def generate_rhythmic_catastrophe_sequence(
    state_a_id: str,
    state_b_id: str,
    oscillation_pattern: str = "sinusoidal",
    num_cycles: int = 3,
    steps_per_cycle: int = 20,
    phase_offset: float = 0.0
) -> dict:
    """
    Generate rhythmic oscillation between two catastrophe morphologies.
    
    PHASE 2.6 TOOL: Temporal composition for catastrophe aesthetics.
    Creates periodic transitions cycling between catastrophe types.
    
    Args:
        state_a_id: Starting catastrophe (fold, cusp, swallowtail, etc.)
        state_b_id: Alternating catastrophe
        oscillation_pattern: "sinusoidal" | "triangular" | "square"
        num_cycles: Number of complete A→B→A cycles
        steps_per_cycle: Samples per cycle
        phase_offset: Starting phase (0.0 = A, 0.5 = B)
    
    Returns:
        Sequence with states, pattern info, and phase points
    
    Cost: 0 tokens (Layer 2)
    """
    if state_a_id not in CATASTROPHE_STATES:
        return {"error": f"Unknown catastrophe: {state_a_id}", "available": list(CATASTROPHE_STATES.keys())}
    if state_b_id not in CATASTROPHE_STATES:
        return {"error": f"Unknown catastrophe: {state_b_id}", "available": list(CATASTROPHE_STATES.keys())}
    
    coords_a = CATASTROPHE_STATES[state_a_id]["coordinates"]
    coords_b = CATASTROPHE_STATES[state_b_id]["coordinates"]
    
    total_steps = num_cycles * steps_per_cycle
    
    # Generate oscillation
    if oscillation_pattern == "sinusoidal":
        alpha = _gen_sin_osc(total_steps, num_cycles, phase_offset)
    elif oscillation_pattern == "triangular":
        alpha = _gen_tri_osc(total_steps, num_cycles, phase_offset)
    elif oscillation_pattern == "square":
        alpha = _gen_sqr_osc(total_steps, num_cycles, phase_offset)
    else:
        return {"error": f"Unknown pattern: {oscillation_pattern}"}
    
    # Interpolate states
    sequence = []
    for a in alpha:
        state = {
            param: float((1-a)*coords_a[param] + a*coords_b[param])
            for param in PARAMETER_NAMES
        }
        sequence.append(state)
    
    return {
        "sequence": sequence,
        "pattern_type": oscillation_pattern,
        "num_cycles": num_cycles,
        "steps_per_cycle": steps_per_cycle,
        "total_steps": total_steps,
        "phase_offset": phase_offset,
        "state_a": {"id": state_a_id, "name": CATASTROPHE_STATES[state_a_id]["name"]},
        "state_b": {"id": state_b_id, "name": CATASTROPHE_STATES[state_b_id]["name"]},
        "aesthetic_flow": f"{oscillation_pattern.title()} oscillation between {state_a_id} and {state_b_id}"
    }


@mcp.tool()
def apply_catastrophe_rhythmic_preset(preset_name: str) -> dict:
    """
    Apply curated catastrophe rhythmic pattern preset.
    
    PHASE 2.6 CONVENIENCE TOOL: Pre-configured patterns.
    
    Available Presets:
    - bifurcation_cycle: fold ↔ cusp (smooth/sharp cycling)
    - complexity_scan: fold ↔ parabolic_umbilic (complexity ramp)
    - umbilic_alternation: elliptic ↔ hyperbolic (wave/saddle)
    - symmetry_pulse: cusp ↔ butterfly (symmetry toggle)
    - tension_breathing: fold ↔ hyperbolic (tension oscillation)
    
    Cost: 0 tokens
    """
    if preset_name not in CATASTROPHE_RHYTHMIC_PRESETS:
        return {"error": f"Unknown preset: {preset_name}", "available": list(CATASTROPHE_RHYTHMIC_PRESETS.keys())}
    
    preset = CATASTROPHE_RHYTHMIC_PRESETS[preset_name]
    
    result = generate_rhythmic_catastrophe_sequence(
        preset["state_a_id"],
        preset["state_b_id"],
        preset["oscillation_pattern"],
        preset["num_cycles"],
        preset["steps_per_cycle"]
    )
    
    result["preset_info"] = {
        "name": preset_name,
        "description": preset["description"],
        "use_case": preset["use_case"],
        "visual_effect": preset["visual_effect"]
    }
    
    return result


@mcp.tool()
def list_catastrophe_rhythmic_presets() -> dict:
    """List all available catastrophe rhythmic presets."""
    return {
        "presets": {
            name: {
                "description": cfg["description"],
                "states": f"{cfg['state_a_id']} ↔ {cfg['state_b_id']}",
                "pattern": cfg["oscillation_pattern"],
                "period": cfg["steps_per_cycle"],
                "use_case": cfg["use_case"]
            }
            for name, cfg in CATASTROPHE_RHYTHMIC_PRESETS.items()
        },
        "total_presets": len(CATASTROPHE_RHYTHMIC_PRESETS)
    }


# ============================================================================
# PHASE 2.7: ATTRACTOR VISUALIZATION PROMPT GENERATION
# ============================================================================

# Visual Vocabulary Extractors - map parameter coordinates to image gen keywords
# Each visual type has canonical coordinates and associated keywords from the olog

CATASTROPHE_VISUAL_TYPES = {
    "fold_smooth": {
        "coords": {
            "control_complexity": 0.0,
            "geometric_sharpness": 0.2,
            "surface_tension": 0.3,
            "optical_intensity": 0.4,
            "aesthetic_intensity": 0.5
        },
        "keywords": [
            "fold singularities",
            "smooth transitions",
            "soft folds",
            "gradual collapse",
            "matte diffuse finish",
            "earth tones",
            "paper-like creases",
            "compression ridges"
        ]
    },
    "cusp_crystalline": {
        "coords": {
            "control_complexity": 0.25,
            "geometric_sharpness": 0.9,
            "surface_tension": 0.7,
            "optical_intensity": 0.8,
            "aesthetic_intensity": 0.7
        },
        "keywords": [
            "cusp points",
            "sharp vertices",
            "faceted planes",
            "crystalline precision",
            "knife edges",
            "specular reflective facets",
            "metallic silvers",
            "hard-edged shadows"
        ]
    },
    "swallowtail_flowing": {
        "coords": {
            "control_complexity": 0.5,
            "geometric_sharpness": 0.4,
            "surface_tension": 0.5,
            "optical_intensity": 0.7,
            "aesthetic_intensity": 0.8
        },
        "keywords": [
            "flowing curves",
            "trailing forms",
            "sweeping surfaces",
            "cascading transitions",
            "iridescent refraction",
            "feathered edges",
            "aurora colors",
            "dynamic light play"
        ]
    },
    "butterfly_radiant": {
        "coords": {
            "control_complexity": 0.75,
            "geometric_sharpness": 0.6,
            "surface_tension": 0.6,
            "optical_intensity": 0.85,
            "aesthetic_intensity": 0.9
        },
        "keywords": [
            "bilateral symmetry",
            "wing-like structures",
            "radial expansion",
            "mirrored patterns",
            "structural color iridescence",
            "scalloped edges",
            "morpho blues",
            "emergence bloom"
        ]
    },
    "umbilic_wave": {
        "coords": {
            "control_complexity": 0.5,
            "geometric_sharpness": 0.3,
            "surface_tension": 0.4,
            "optical_intensity": 0.65,
            "aesthetic_intensity": 0.75
        },
        "keywords": [
            "wave propagation",
            "ripple interference patterns",
            "undulating surfaces",
            "periodic structures",
            "metallic wave-like reflection",
            "rhythmic bands",
            "mercury silvers",
            "harmonic oscillation"
        ]
    },
    "umbilic_tension": {
        "coords": {
            "control_complexity": 0.5,
            "geometric_sharpness": 0.7,
            "surface_tension": 0.9,
            "optical_intensity": 0.7,
            "aesthetic_intensity": 0.85
        },
        "keywords": [
            "saddle surfaces",
            "hyperbolic paraboloids",
            "stress concentration lines",
            "tension forces",
            "directional metallic reflection",
            "tensioned curves",
            "steel grays",
            "equilibrium strain"
        ]
    },
    "mushroom_organic": {
        "coords": {
            "control_complexity": 1.0,
            "geometric_sharpness": 0.5,
            "surface_tension": 0.55,
            "optical_intensity": 0.6,
            "aesthetic_intensity": 0.95
        },
        "keywords": [
            "mushroom caps",
            "parabolic domes",
            "canopy structures",
            "cap curvature",
            "satin dome focusing",
            "curling margins",
            "dome golds",
            "organic emergence"
        ]
    }
}


# Preset attractor states from Tier 4D discovery research
# These are the emergent attractors found in multi-domain composition

CATASTROPHE_ATTRACTOR_PRESETS = {
    "period_30": {
        "name": "Period 30 — Universal Sync",
        "description": "Dominant LCM synchronization across microscopy, diatom, and heraldic domains. Most stable multi-domain attractor.",
        "basin_size": 0.116,
        "classification": "lcm_sync",
        "state": {
            "control_complexity": 0.45,
            "geometric_sharpness": 0.50,
            "surface_tension": 0.55,
            "optical_intensity": 0.65,
            "aesthetic_intensity": 0.75
        }
    },
    "period_29": {
        "name": "Period 29 — Emergent Resonance",
        "description": "Five-domain specific emergence. Multi-domain LCM synchronization pathway discovered only in 5+ domain composition.",
        "basin_size": 0.084,
        "classification": "lcm_sync",
        "state": {
            "control_complexity": 0.50,
            "geometric_sharpness": 0.55,
            "surface_tension": 0.50,
            "optical_intensity": 0.70,
            "aesthetic_intensity": 0.80
        }
    },
    "period_19": {
        "name": "Period 19 — Gap Flow",
        "description": "Novel gap-filler attractor (18-20). Resilient across 4- and 5-domain systems. Fundamental aesthetic intermediate.",
        "basin_size": 0.074,
        "classification": "novel_gap_filler",
        "state": {
            "control_complexity": 0.30,
            "geometric_sharpness": 0.45,
            "surface_tension": 0.40,
            "optical_intensity": 0.55,
            "aesthetic_intensity": 0.65
        }
    },
    "period_28": {
        "name": "Period 28 — Composite Beat",
        "description": "Novel composite beat mechanism (60 - 2×16 = 28). Demonstrates attractor-attractor interaction. Fragile to domain additions.",
        "basin_size": 0.024,
        "classification": "novel_composite_beat",
        "state": {
            "control_complexity": 0.40,
            "geometric_sharpness": 0.60,
            "surface_tension": 0.65,
            "optical_intensity": 0.75,
            "aesthetic_intensity": 0.80
        }
    },
    "period_60": {
        "name": "Period 60 — Harmonic Hub",
        "description": "Major LCM hub in 4-domain system. Weakened in 5-domain (-66%) but still present. Complex multi-scale synchronization.",
        "basin_size": 0.040,
        "classification": "harmonic",
        "state": {
            "control_complexity": 0.55,
            "geometric_sharpness": 0.65,
            "surface_tension": 0.60,
            "optical_intensity": 0.75,
            "aesthetic_intensity": 0.85
        }
    },
    "bifurcation_edge": {
        "name": "Bifurcation Edge — Cusp Threshold",
        "description": "Aesthetic state at the cusp bifurcation boundary. Maximum geometric tension before catastrophic snap.",
        "basin_size": None,
        "classification": "curated",
        "state": {
            "control_complexity": 0.20,
            "geometric_sharpness": 0.85,
            "surface_tension": 0.75,
            "optical_intensity": 0.80,
            "aesthetic_intensity": 0.70
        }
    },
    "organic_complexity": {
        "name": "Organic Complexity — Mushroom Emergence",
        "description": "Maximum control complexity with organic character. Parabolic umbilic region with dome curvature.",
        "basin_size": None,
        "classification": "curated",
        "state": {
            "control_complexity": 0.90,
            "geometric_sharpness": 0.50,
            "surface_tension": 0.55,
            "optical_intensity": 0.60,
            "aesthetic_intensity": 0.90
        }
    }
}


def _extract_catastrophe_visual_vocabulary(
    state: dict,
    strength: float = 1.0
) -> dict:
    """
    Extract visual vocabulary from catastrophe parameter coordinates.
    
    Uses nearest-neighbor matching against canonical visual types.
    
    Args:
        state: Parameter coordinates dict
        strength: Keyword weight multiplier [0.0, 1.0]
    
    Returns:
        Dict with nearest_type, distance, keywords, optical_properties
    
    Cost: 0 tokens (pure distance computation)
    """
    # Build coordinate vectors
    state_vec = np.array([state.get(p, 0.5) for p in PARAMETER_NAMES])
    
    min_dist = float('inf')
    nearest_type = None
    
    for type_name, type_data in CATASTROPHE_VISUAL_TYPES.items():
        type_vec = np.array([type_data["coords"][p] for p in PARAMETER_NAMES])
        dist = float(np.linalg.norm(state_vec - type_vec))
        
        if dist < min_dist:
            min_dist = dist
            nearest_type = type_name
    
    type_data = CATASTROPHE_VISUAL_TYPES[nearest_type]
    
    # Get optical properties from the olog for the nearest morphology
    # Map visual type back to catastrophe morphology
    type_to_morphology = {
        "fold_smooth": "fold",
        "cusp_crystalline": "cusp",
        "swallowtail_flowing": "swallowtail",
        "butterfly_radiant": "butterfly",
        "umbilic_wave": "elliptic_umbilic",
        "umbilic_tension": "hyperbolic_umbilic",
        "mushroom_organic": "parabolic_umbilic"
    }
    
    morphology_id = type_to_morphology.get(nearest_type, "fold")
    morphology_data = CATASTROPHE_MORPHOLOGIES.get(morphology_id, {})
    optical = morphology_data.get("optical_properties", {})
    colors = morphology_data.get("color_associations", [])
    intentionality = morphology_data.get("intentionality", "")
    
    return {
        "nearest_type": nearest_type,
        "morphology": morphology_id,
        "distance": min_dist,
        "strength": strength,
        "keywords": type_data["keywords"],
        "optical_properties": optical,
        "color_associations": colors,
        "intentionality": intentionality.strip()
    }


def _build_composite_prompt(
    attractor_state: dict,
    style_modifier: str = "",
    include_optical: bool = True
) -> str:
    """
    Build a composite image generation prompt from attractor state.
    
    Translates 5D catastrophe coordinates into descriptive visual prompt
    suitable for image generation models.
    
    Args:
        attractor_state: Parameter coordinate dict
        style_modifier: Optional style prefix ("photorealistic", "abstract", etc.)
        include_optical: Include optical property descriptors
    
    Returns:
        Prompt string for image generation
    
    Cost: 0 tokens (deterministic string assembly)
    """
    vocab = _extract_catastrophe_visual_vocabulary(attractor_state)
    
    # Build keyword set weighted by strength
    keywords = vocab["keywords"][:6]  # Top 6 keywords
    
    # Add optical descriptors
    optical_parts = []
    if include_optical and vocab["optical_properties"]:
        opt = vocab["optical_properties"]
        if "finish" in opt:
            optical_parts.append(f"{opt['finish']} finish")
        if "light_behavior" in opt:
            optical_parts.append(opt["light_behavior"])
        if "shadow_quality" in opt:
            optical_parts.append(f"{opt['shadow_quality']} shadows")
    
    # Add color associations
    colors = vocab.get("color_associations", [])[:3]
    
    # Assemble prompt
    parts = []
    if style_modifier:
        parts.append(style_modifier)
    
    parts.extend(keywords)
    
    if optical_parts:
        parts.extend(optical_parts)
    
    if colors:
        parts.append(", ".join(colors))
    
    return ", ".join(parts)


def _build_sequence_prompts(
    sequence_states: list,
    keyframe_count: int = 4,
    style_modifier: str = ""
) -> list:
    """
    Build sequence of prompts from trajectory/rhythmic sequence states.
    
    Samples keyframes evenly from the sequence and generates a prompt
    for each keyframe.
    
    Args:
        sequence_states: List of parameter state dicts
        keyframe_count: Number of keyframes to extract
        style_modifier: Optional style prefix
    
    Returns:
        List of {keyframe_index, step, state, prompt} dicts
    
    Cost: 0 tokens
    """
    total = len(sequence_states)
    if total == 0:
        return []
    
    # Sample evenly spaced keyframes
    indices = np.linspace(0, total - 1, keyframe_count, dtype=int)
    
    keyframes = []
    for i, idx in enumerate(indices):
        state = sequence_states[idx]
        prompt = _build_composite_prompt(state, style_modifier)
        vocab = _extract_catastrophe_visual_vocabulary(state)
        
        keyframes.append({
            "keyframe_index": i,
            "step": int(idx),
            "state": state,
            "nearest_type": vocab["nearest_type"],
            "morphology": vocab["morphology"],
            "prompt": prompt
        })
    
    return keyframes


@mcp.tool()
def extract_catastrophe_visual_vocabulary(
    state: dict,
    strength: float = 1.0
) -> dict:
    """
    Extract visual vocabulary from catastrophe parameter coordinates.
    
    PHASE 2.7 TOOL: Maps a 5D parameter state to the nearest canonical
    catastrophe visual type and returns image-generation-ready keywords.
    
    Uses nearest-neighbor matching against 7 visual types derived from
    the catastrophe theory olog vocabulary.
    
    Args:
        state: Parameter coordinates dict with keys:
            control_complexity, geometric_sharpness, surface_tension,
            optical_intensity, aesthetic_intensity
        strength: Keyword weight multiplier [0.0, 1.0] (default: 1.0)
    
    Returns:
        Dict with nearest_type, morphology, distance, keywords,
        optical_properties, color_associations, intentionality
    
    Cost: 0 tokens (pure Layer 2 computation)
    
    Example:
        >>> extract_catastrophe_visual_vocabulary({
        ...     "control_complexity": 0.25,
        ...     "geometric_sharpness": 0.85,
        ...     "surface_tension": 0.70,
        ...     "optical_intensity": 0.80,
        ...     "aesthetic_intensity": 0.70
        ... })
        {
            "nearest_type": "cusp_crystalline",
            "morphology": "cusp",
            "distance": 0.071,
            "keywords": ["cusp points", "sharp vertices", ...],
            "optical_properties": {"finish": "specular", ...},
            "color_associations": ["metallic silvers", "obsidian blacks", ...]
        }
    """
    return _extract_catastrophe_visual_vocabulary(state, strength)


@mcp.tool()
def generate_catastrophe_attractor_prompt(
    attractor_id: str = "",
    custom_state: dict = None,
    mode: str = "composite",
    style_modifier: str = "",
    keyframe_count: int = 4
) -> dict:
    """
    Generate image generation prompt from attractor state or custom coordinates.
    
    PHASE 2.7 TOOL: Translates mathematical attractor coordinates into
    visual prompts suitable for image generation (ComfyUI, Stable Diffusion,
    DALL-E, etc.).
    
    Modes:
        composite: Single blended prompt from attractor state
        split_view: Separate prompt per catastrophe morphology component
        sequence: Multiple keyframe prompts from rhythmic preset trajectory
    
    Args:
        attractor_id: Preset attractor name (period_30, period_19, etc.)
            Use "" with custom_state for arbitrary coordinates.
        custom_state: Optional custom parameter coordinates dict.
            Overrides attractor_id if provided.
        mode: "composite" | "split_view" | "sequence"
        style_modifier: Optional prefix ("photorealistic", "oil painting", etc.)
        keyframe_count: Number of keyframes for sequence mode (default: 4)
    
    Returns:
        Dict with prompt(s), vocabulary details, and attractor metadata
    
    Cost: 0 tokens (Layer 2 deterministic)
    
    Available attractor presets:
        period_30: Universal Sync (11.6% basin, LCM)
        period_29: Emergent Resonance (8.4% basin, LCM)
        period_19: Gap Flow (7.4% basin, novel)
        period_28: Composite Beat (2.4% basin, novel)
        period_60: Harmonic Hub (4.0% basin, harmonic)
        bifurcation_edge: Cusp threshold (curated)
        organic_complexity: Mushroom emergence (curated)
    
    Example:
        >>> generate_catastrophe_attractor_prompt("period_28", mode="composite")
        {
            "prompt": "cusp points, sharp vertices, faceted planes, ...",
            "attractor": {"name": "Period 28 — Composite Beat", ...},
            "vocabulary": {"nearest_type": "cusp_crystalline", ...}
        }
    """
    import json
    
    # Resolve state
    if custom_state is not None:
        state = custom_state
        attractor_meta = {
            "name": "Custom State",
            "description": "User-provided parameter coordinates",
            "classification": "custom"
        }
    elif attractor_id and attractor_id in CATASTROPHE_ATTRACTOR_PRESETS:
        preset = CATASTROPHE_ATTRACTOR_PRESETS[attractor_id]
        state = preset["state"]
        attractor_meta = {
            "name": preset["name"],
            "description": preset["description"],
            "basin_size": preset["basin_size"],
            "classification": preset["classification"]
        }
    else:
        return {
            "error": f"Provide attractor_id or custom_state. Available presets: {list(CATASTROPHE_ATTRACTOR_PRESETS.keys())}"
        }
    
    # Validate state has required parameters
    missing = [p for p in PARAMETER_NAMES if p not in state]
    if missing:
        return {"error": f"Missing parameters: {missing}", "required": PARAMETER_NAMES}
    
    if mode == "composite":
        vocab = _extract_catastrophe_visual_vocabulary(state)
        prompt = _build_composite_prompt(state, style_modifier)
        
        return {
            "mode": "composite",
            "prompt": prompt,
            "attractor": attractor_meta,
            "vocabulary": {
                "nearest_type": vocab["nearest_type"],
                "morphology": vocab["morphology"],
                "distance": vocab["distance"],
                "keywords": vocab["keywords"],
                "optical_properties": vocab["optical_properties"],
                "color_associations": vocab["color_associations"],
                "intentionality": vocab["intentionality"]
            },
            "state": state,
            "cost": "0 tokens"
        }
    
    elif mode == "split_view":
        # Generate separate prompts for nearby morphology extremes
        # Find the two closest visual types for the split
        state_vec = np.array([state.get(p, 0.5) for p in PARAMETER_NAMES])
        
        distances = []
        for type_name, type_data in CATASTROPHE_VISUAL_TYPES.items():
            type_vec = np.array([type_data["coords"][p] for p in PARAMETER_NAMES])
            dist = float(np.linalg.norm(state_vec - type_vec))
            distances.append((type_name, dist))
        
        distances.sort(key=lambda x: x[1])
        
        panels = []
        for type_name, dist in distances[:3]:  # Top 3 closest types
            type_data = CATASTROPHE_VISUAL_TYPES[type_name]
            panel_prompt = _build_composite_prompt(type_data["coords"], style_modifier)
            
            type_to_morph = {
                "fold_smooth": "fold", "cusp_crystalline": "cusp",
                "swallowtail_flowing": "swallowtail", "butterfly_radiant": "butterfly",
                "umbilic_wave": "elliptic_umbilic", "umbilic_tension": "hyperbolic_umbilic",
                "mushroom_organic": "parabolic_umbilic"
            }
            
            panels.append({
                "visual_type": type_name,
                "morphology": type_to_morph.get(type_name, type_name),
                "distance_from_state": dist,
                "prompt": panel_prompt,
                "keywords": type_data["keywords"]
            })
        
        return {
            "mode": "split_view",
            "panels": panels,
            "attractor": attractor_meta,
            "state": state,
            "note": "Each panel represents a nearby catastrophe visual type. Blend or display separately.",
            "cost": "0 tokens"
        }
    
    elif mode == "sequence":
        # Generate keyframe sequence from the nearest rhythmic preset
        # Find which preset is closest to this attractor state
        best_preset = None
        best_dist = float('inf')
        
        for preset_name, preset_cfg in CATASTROPHE_RHYTHMIC_PRESETS.items():
            # Check midpoint of the preset trajectory
            a_coords = CATASTROPHE_STATES[preset_cfg["state_a_id"]]["coordinates"]
            b_coords = CATASTROPHE_STATES[preset_cfg["state_b_id"]]["coordinates"]
            mid = {p: (a_coords[p] + b_coords[p]) / 2 for p in PARAMETER_NAMES}
            mid_vec = np.array([mid[p] for p in PARAMETER_NAMES])
            state_vec_seq = np.array([state.get(p, 0.5) for p in PARAMETER_NAMES])
            dist = float(np.linalg.norm(mid_vec - state_vec_seq))
            
            if dist < best_dist:
                best_dist = dist
                best_preset = preset_name
        
        # Generate the sequence from that preset
        preset_cfg = CATASTROPHE_RHYTHMIC_PRESETS[best_preset]
        seq_result = generate_rhythmic_catastrophe_sequence(
            preset_cfg["state_a_id"],
            preset_cfg["state_b_id"],
            preset_cfg["oscillation_pattern"],
            preset_cfg["num_cycles"],
            preset_cfg["steps_per_cycle"]
        )
        
        if "error" in seq_result:
            return seq_result
        
        keyframes = _build_sequence_prompts(
            seq_result["sequence"],
            keyframe_count=keyframe_count,
            style_modifier=style_modifier
        )
        
        return {
            "mode": "sequence",
            "preset_used": best_preset,
            "preset_distance": best_dist,
            "preset_info": {
                "description": preset_cfg["description"],
                "states": f"{preset_cfg['state_a_id']} ↔ {preset_cfg['state_b_id']}",
                "pattern": preset_cfg["oscillation_pattern"],
                "period": preset_cfg["steps_per_cycle"],
                "total_steps": preset_cfg["num_cycles"] * preset_cfg["steps_per_cycle"]
            },
            "keyframes": keyframes,
            "attractor": attractor_meta,
            "cost": "0 tokens"
        }
    
    else:
        return {"error": f"Unknown mode: {mode}", "available": ["composite", "split_view", "sequence"]}


@mcp.tool()
def list_catastrophe_attractor_presets() -> dict:
    """
    List all available catastrophe attractor presets for visualization.
    
    PHASE 2.7 TOOL: Shows discovered and curated attractor configurations
    available for prompt generation.
    
    Returns:
        Dict with preset names, descriptions, basin sizes, and classifications
    
    Cost: 0 tokens
    """
    presets = {}
    for pid, pdata in CATASTROPHE_ATTRACTOR_PRESETS.items():
        vocab = _extract_catastrophe_visual_vocabulary(pdata["state"])
        presets[pid] = {
            "name": pdata["name"],
            "description": pdata["description"],
            "basin_size": pdata["basin_size"],
            "classification": pdata["classification"],
            "nearest_visual_type": vocab["nearest_type"],
            "morphology": vocab["morphology"],
            "distance_to_type": round(vocab["distance"], 4)
        }
    
    return {
        "attractor_presets": presets,
        "total_presets": len(presets),
        "visual_types_available": len(CATASTROPHE_VISUAL_TYPES),
        "modes": ["composite", "split_view", "sequence"],
        "usage": "generate_catastrophe_attractor_prompt(attractor_id='period_30', mode='composite')"
    }


@mcp.tool()
def generate_catastrophe_sequence_prompts(
    preset_name: str,
    keyframe_count: int = 4,
    style_modifier: str = ""
) -> dict:
    """
    Generate keyframe prompts from a Phase 2.6 rhythmic preset.
    
    PHASE 2.7 TOOL: Extracts evenly-spaced keyframes from a rhythmic
    oscillation sequence and generates an image prompt for each.
    
    Useful for:
    - Storyboard generation from rhythmic compositions
    - Animation keyframe specification
    - Multi-panel visualization of temporal aesthetic evolution
    
    Args:
        preset_name: Phase 2.6 preset (bifurcation_cycle, complexity_scan, etc.)
        keyframe_count: Number of keyframes to extract (default: 4)
        style_modifier: Optional style prefix for all prompts
    
    Returns:
        Dict with keyframes, each containing step, state, prompt, and vocabulary
    
    Cost: 0 tokens
    
    Example:
        >>> generate_catastrophe_sequence_prompts("bifurcation_cycle", keyframe_count=6)
        {
            "preset": "bifurcation_cycle",
            "keyframes": [
                {"step": 0, "prompt": "fold singularities, smooth transitions, ...", ...},
                {"step": 16, "prompt": "cusp points, sharp vertices, ...", ...},
                ...
            ]
        }
    """
    if preset_name not in CATASTROPHE_RHYTHMIC_PRESETS:
        return {
            "error": f"Unknown preset: {preset_name}",
            "available": list(CATASTROPHE_RHYTHMIC_PRESETS.keys())
        }
    
    preset_cfg = CATASTROPHE_RHYTHMIC_PRESETS[preset_name]
    
    # Generate the full sequence
    seq_result = generate_rhythmic_catastrophe_sequence(
        preset_cfg["state_a_id"],
        preset_cfg["state_b_id"],
        preset_cfg["oscillation_pattern"],
        preset_cfg["num_cycles"],
        preset_cfg["steps_per_cycle"]
    )
    
    if "error" in seq_result:
        return seq_result
    
    # Extract keyframe prompts
    keyframes = _build_sequence_prompts(
        seq_result["sequence"],
        keyframe_count=keyframe_count,
        style_modifier=style_modifier
    )
    
    return {
        "preset": preset_name,
        "preset_info": {
            "description": preset_cfg["description"],
            "states": f"{preset_cfg['state_a_id']} ↔ {preset_cfg['state_b_id']}",
            "pattern": preset_cfg["oscillation_pattern"],
            "period": preset_cfg["steps_per_cycle"],
            "total_steps": seq_result["total_steps"],
            "visual_effect": preset_cfg["visual_effect"]
        },
        "keyframes": keyframes,
        "total_keyframes": len(keyframes),
        "cost": "0 tokens"
    }


# ============================================================================
# Server Metadata
# ============================================================================

@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the Catastrophe Morph MCP server.
    
    Returns server metadata, capabilities, and Phase 1A enhancements.
    """
    return {
        "name": "Catastrophe Morph MCP",
        "version": SERVER_VERSION,
        "validation_date": VALIDATION_DATE,
        "description": "Visual aesthetics derived from René Thom's catastrophe theory with trajectory dynamics and attractor visualization",
        "catastrophe_theory": {
            "elementary_catastrophes": 7,
            "catastrophe_types": list(CATASTROPHE_STATES.keys()),
            "parameter_space_dimensions": len(PARAMETER_NAMES),
            "reference": "René Thom - Structural Stability and Morphogenesis (1972)"
        },
        "capabilities": {
            "layer_1_taxonomy": [
                "get_catastrophe_types - Pure lookup",
                "get_catastrophe_specifications - Complete visual vocabulary"
            ],
            "layer_2_deterministic": [
                "map_catastrophe_parameters - Parameter mapping",
                "compute_catastrophe_distance - Distance metrics",
                "compute_trajectory_between_catastrophes - RK4 trajectory integration (Phase 1A)",
                "generate_rhythmic_catastrophe_sequence - Oscillatory composition (Phase 2.6)",
                "apply_catastrophe_rhythmic_preset - Curated rhythmic patterns (Phase 2.6)",
                "list_catastrophe_rhythmic_presets - Available preset catalog (Phase 2.6)",
                "extract_catastrophe_visual_vocabulary - Parameter→keyword mapping (Phase 2.7)",
                "generate_catastrophe_attractor_prompt - Attractor→image prompt (Phase 2.7)",
                "list_catastrophe_attractor_presets - Attractor catalog (Phase 2.7)",
                "generate_catastrophe_sequence_prompts - Keyframe prompts (Phase 2.7)"
            ],
            "layer_3_synthesis": [
                "prepare_transition_visualization - Visualization context"
            ]
        },
        "phase_1a_enhancements": {
            "dynamics_available": DYNAMICS_AVAILABLE,
            "integration_method": "RK4 (Runge-Kutta 4th order)" if DYNAMICS_AVAILABLE else "Not available",
            "trajectory_features": [
                "Zero-cost morphospace navigation",
                "Smooth catastrophe transitions",
                "Convergence analysis",
                "Path efficiency metrics",
                "Transition characterization"
            ] if DYNAMICS_AVAILABLE else [],
            "installation": "pip install aesthetic-dynamics-core --break-system-packages" if not DYNAMICS_AVAILABLE else "Installed"
        },
        "compatible_bricks": [
            "aesthetic-dynamics-core - Phase 1A trajectory computation (required for dynamics)",
            "composition-graph-mcp - Multi-domain composition analysis",
            "diatom-morph-mcp - Microscopic aesthetic composition"
        ],
        "cost_profile": {
            "layer_1": "0 tokens (pure lookup)",
            "layer_2": "0 tokens (deterministic computation + RK4 + prompt generation)",
            "layer_3": "~100-200 tokens (Claude synthesis)"
        },
        "parameter_space": {
            "parameters": PARAMETER_NAMES,
            "bounds": PARAMETER_BOUNDS,
            "dimensionality": len(PARAMETER_NAMES)
        },
        "phase_2_6_enhancements": {
            "rhythmic_composition": True,
            "oscillation_patterns": ["sinusoidal", "triangular", "square"],
            "available_presets": len(CATASTROPHE_RHYTHMIC_PRESETS),
            "preset_names": list(CATASTROPHE_RHYTHMIC_PRESETS.keys()),
            "forced_orbit_available": FORCED_ORBIT_AVAILABLE,
            "temporal_features": [
                "Bifurcation cycling (smooth ↔ sharp)",
                "Complexity scanning (simple ↔ complex)",
                "Umbilic alternation (wave ↔ saddle)",
                "Symmetry pulsing (2-fold ↔ 4-fold)",
                "Tension breathing (minimal ↔ maximum)"
            ],
            "use_cases": [
                "Temporal bifurcation sequences",
                "Control complexity animation",
                "Umbilic pattern oscillation",
                "Symmetry order variation",
                "Surface tension dynamics"
            ],
            "forced_orbit_integration": {
                "status": "PRODUCTION READY" if FORCED_ORBIT_AVAILABLE else "NOT AVAILABLE",
                "method": "Phase-space integration (S¹) for perfect limit cycles",
                "advantages": [
                    "Perfect cycle closure (zero drift)",
                    "No parameter tuning required",
                    "Deterministic phase advancement",
                    "Multi-domain composition ready"
                ] if FORCED_ORBIT_AVAILABLE else [],
                "integration_note": "Use aesthetic-dynamics-core v1.0.0+ for forced orbit support"
            }
        },
        "phase_2_7_enhancements": {
            "attractor_visualization": True,
            "supported_visual_types": len(CATASTROPHE_VISUAL_TYPES),
            "visual_type_names": list(CATASTROPHE_VISUAL_TYPES.keys()),
            "attractor_presets": len(CATASTROPHE_ATTRACTOR_PRESETS),
            "attractor_preset_names": list(CATASTROPHE_ATTRACTOR_PRESETS.keys()),
            "prompt_modes": ["composite", "split_view", "sequence"],
            "features": [
                "Nearest-neighbor vocabulary extraction from 5D coordinates",
                "7 visual types mapped from catastrophe olog vocabulary",
                "Composite prompt blending with optical properties",
                "Split-view multi-panel prompt generation",
                "Sequence keyframe extraction from rhythmic presets",
                "Preset attractor catalog from Tier 4D discovery",
                "Custom coordinate support for arbitrary states"
            ],
            "vocabulary_source": "catastrophe_theory.yaml olog (7 morphologies × 8 keyword categories)",
            "image_gen_compatibility": [
                "ComfyUI / Stable Diffusion",
                "DALL-E",
                "Midjourney (prompt adaptation may be needed)"
            ]
        }
    }

# ============================================================================
# STRATEGIC ANALYSIS VOCABULARY
# ============================================================================

STRATEGIC_MAPPINGS = {
    "catastrophe_type": {
        "fold": {
            "pattern": "binary_clarity",
            "confidence": 0.85,
            "keywords": ["either", "or", "binary", "yes/no", "clear choice", "simple decision", 
                        "two options", "straightforward", "black and white"],
            "description": "Strategy exhibits clear binary choices - decisions are straightforward with minimal ambiguity",
            "categorical_family": "constraints"
        },
        "cusp": {
            "pattern": "threshold_sensitivity", 
            "confidence": 0.85,
            "keywords": ["tipping point", "threshold", "critical mass", "point of no return",
                        "irreversible", "commitment", "once we", "after which", "decisive"],
            "description": "Strategy contains threshold effects where crossing certain points triggers irreversible changes",
            "categorical_family": "morphisms"
        },
        "swallowtail": {
            "pattern": "multi_path_complexity",
            "confidence": 0.80,
            "keywords": ["multiple approaches", "various paths", "depending on", "context-dependent",
                        "trade-offs", "competing priorities", "balance between", "tension"],
            "description": "Strategy shows complex topology with multiple valid paths and context-dependent choices",
            "categorical_family": "objects"
        },
        "butterfly": {
            "pattern": "emergent_coordination",
            "confidence": 0.80,
            "keywords": ["emergence", "coordination", "alignment", "synthesis", "integration",
                        "bringing together", "unified", "holistic", "metamorphosis"],
            "description": "Strategy emphasizes emergent properties from coordinating multiple dimensions",
            "categorical_family": "morphisms"
        },
        "elliptic_umbilic": {
            "pattern": "cyclical_rhythm",
            "confidence": 0.75,
            "keywords": ["cycle", "rhythm", "periodic", "recurring", "wave", "oscillate",
                        "ebb and flow", "seasonal", "iterative"],
            "description": "Strategy exhibits cyclical patterns or rhythmic repetition",
            "categorical_family": "constraints"
        },
        "hyperbolic_umbilic": {
            "pattern": "tension_equilibrium",
            "confidence": 0.80,
            "keywords": ["tension", "balance", "opposing", "countervailing", "trade-off",
                        "push-pull", "competing", "balanced forces", "equilibrium"],
            "description": "Strategy reveals tensions between opposing forces seeking equilibrium",
            "categorical_family": "constraints"
        },
        "parabolic_umbilic": {
            "pattern": "organic_emergence",
            "confidence": 0.75,
            "keywords": ["organic", "natural", "emerge", "grow", "develop", "evolve",
                        "bottom-up", "grassroots", "cultivate"],
            "description": "Strategy emphasizes organic growth and natural emergence patterns",
            "categorical_family": "morphisms"
        }
    },
    
    "complexity_dimension": {
        "single_control": {
            "control_params": 1,
            "pattern": "simple_variable",
            "keywords": ["single", "one", "primary", "main", "key", "core", "central"],
            "description": "Strategy controlled by single primary variable",
            "categorical_family": "objects"
        },
        "dual_control": {
            "control_params": 2,
            "pattern": "two_factors",
            "keywords": ["two", "dual", "both", "pair", "couple"],
            "description": "Strategy governed by two independent factors",
            "categorical_family": "objects"
        },
        "triple_control": {
            "control_params": 3,
            "pattern": "three_dimensions",
            "keywords": ["three", "triple", "triad"],
            "description": "Strategy operating across three dimensions",
            "categorical_family": "objects"
        },
        "quad_control": {
            "control_params": 4,
            "pattern": "four_factors",
            "keywords": ["four", "quad", "multiple", "many", "various"],
            "description": "Strategy involving four or more independent variables",
            "categorical_family": "objects"
        }
    },
    
    "optical_manifestation": {
        "diffuse": {
            "pattern": "distributed_execution",
            "keywords": ["distributed", "decentralized", "spread", "diffuse", "widespread"],
            "description": "Strategy execution is distributed across multiple actors/locations",
            "categorical_family": "morphisms"
        },
        "specular": {
            "pattern": "focused_precision",
            "keywords": ["focused", "targeted", "precise", "sharp", "concentrated"],
            "description": "Strategy shows precise, focused execution",
            "categorical_family": "morphisms"
        },
        "iridescent": {
            "pattern": "adaptive_response",
            "keywords": ["adapt", "flexible", "responsive", "dynamic", "changing"],
            "description": "Strategy adapts appearance based on context",
            "categorical_family": "morphisms"
        },
        "metallic": {
            "pattern": "structured_reflection",
            "keywords": ["reflect", "mirror", "systematic", "structured", "methodical"],
            "description": "Strategy exhibits systematic, reflective processes",
            "categorical_family": "morphisms"
        }
    }
}


def _analyze_catastrophe_strategy(text: str) -> list:
    """
    Layer 2 deterministic analysis - pattern match strategy text through catastrophe vocabulary.
    
    Returns list of findings in tomographic format.
    Cost: 0 tokens (pure pattern matching)
    """
    text_lower = text.lower()
    findings = []
    
    # Analyze catastrophe type patterns
    type_scores = {}
    for cat_type, mapping in STRATEGIC_MAPPINGS["catastrophe_type"].items():
        matches = [kw for kw in mapping["keywords"] if kw in text_lower]
        if matches:
            score = len(matches) / len(mapping["keywords"])
            type_scores[cat_type] = {
                "score": score,
                "matches": matches,
                "mapping": mapping
            }
    
    # Report strongest catastrophe type pattern
    if type_scores:
        best_type = max(type_scores, key=lambda x: type_scores[x]["score"])
        data = type_scores[best_type]
        
        findings.append({
            "dimension": "structural_stability",
            "pattern": data["mapping"]["pattern"],
            "confidence": min(0.95, data["mapping"]["confidence"] * (0.5 + data["score"] * 0.5)),
            "evidence": f"Found {len(data['matches'])} stability indicators: {data['matches'][:3]}",
            "categorical_family": data["mapping"]["categorical_family"],
            "catastrophe_type": best_type,
            "control_parameters": CATASTROPHE_MORPHOLOGIES[best_type]["control_parameters"]
        })
    
    # Analyze complexity dimension
    complexity_matches = []
    for complexity, mapping in STRATEGIC_MAPPINGS["complexity_dimension"].items():
        matches = [kw for kw in mapping["keywords"] if kw in text_lower]
        if matches:
            complexity_matches.append({
                "complexity": complexity,
                "control_params": mapping["control_params"],
                "matches": matches
            })
    
    if complexity_matches:
        # Use highest control parameter count
        best = max(complexity_matches, key=lambda x: x["control_params"])
        findings.append({
            "dimension": "strategic_complexity",
            "pattern": f"{best['control_params']}_parameter_system",
            "confidence": 0.80,
            "evidence": f"Complexity indicators: {best['matches'][:3]}",
            "categorical_family": "objects",
            "control_parameters": best["control_params"]
        })
    
    # Analyze optical manifestation (execution characteristics)
    optical_scores = {}
    for optical, mapping in STRATEGIC_MAPPINGS["optical_manifestation"].items():
        matches = [kw for kw in mapping["keywords"] if kw in text_lower]
        if matches:
            optical_scores[optical] = {
                "matches": matches,
                "mapping": mapping
            }
    
    if optical_scores:
        best_optical = max(optical_scores, key=lambda x: len(optical_scores[x]["matches"]))
        data = optical_scores[best_optical]
        
        findings.append({
            "dimension": "execution_manifestation",
            "pattern": data["mapping"]["pattern"],
            "confidence": 0.75,
            "evidence": f"Execution indicators: {data['matches'][:3]}",
            "categorical_family": "morphisms",
            "optical_quality": best_optical
        })
    
    # Check for singularity indicators (structural instability)
    singularity_keywords = [
        "unclear", "ambiguous", "undefined", "uncertain", "conflicting",
        "contradiction", "inconsistent", "incompatible", "unstable"
    ]
    singularity_matches = [kw for kw in singularity_keywords if kw in text_lower]
    
    if len(singularity_matches) >= 2:
        findings.append({
            "dimension": "structural_singularity",
            "pattern": "instability_detected",
            "confidence": 0.85,
            "evidence": f"Singularity indicators: {singularity_matches[:3]}",
            "categorical_family": "constraints",
            "warning": "Strategy contains singularity points where small changes may trigger large effects"
        })
    
    return findings


@mcp.tool()
def analyze_strategy_document_tool(strategy_text: str) -> str:
    """
    Analyze a strategy document through catastrophe theory structural lens.
    
    Projects strategy text through catastrophe vocabulary to detect structural
    patterns: stability type (catastrophe morphology), complexity (control params),
    execution characteristics (optical properties), and singularity points.
    
    This is LAYER 2 deterministic analysis with ZERO LLM cost - pure pattern
    matching against catastrophe taxonomy.
    
    Args:
        strategy_text: Full text of strategy document
        
    Returns:
        JSON string with findings format:
        {
            "domain": "catastrophe_morph",
            "findings": [
                {
                    "dimension": "structural_stability",
                    "pattern": "threshold_sensitivity",
                    "confidence": 0.85,
                    "evidence": "Found 5 threshold indicators: ['tipping point', ...]",
                    "categorical_family": "morphisms",
                    "catastrophe_type": "cusp",
                    "control_parameters": 2
                },
                ...
            ],
            "total_findings": 4,
            "methodology": "deterministic_pattern_matching",
            "llm_cost_tokens": 0
        }
        
    Example:
        >>> result = analyze_strategy_document_tool(strategy_pdf_text)
        >>> findings = json.loads(result)["findings"]
    """
    import json
    
    findings = _analyze_catastrophe_strategy(strategy_text)
    
    return json.dumps({
        "domain": "catastrophe_morph",
        "findings": findings,
        "total_findings": len(findings),
        "methodology": "deterministic_pattern_matching",
        "llm_cost_tokens": 0
    }, indent=2)

# ============================================================================
# ENTRY POINT
# ============================================================================

def create_server():
    """Entry point for FastMCP Cloud deployment."""
    return mcp


if __name__ == "__main__":
    mcp.run()
