#!/usr/bin/env python3
"""
CatastropheMorph MCP Server (FastMCP Format)
Apply catastrophe theory morphologies to enhance prompts.

Uses FastMCP decorators for clean, Anthropic-standard MCP implementation.
"""

import json
import random
from fastmcp import FastMCP

# ============================================================================
# CATASTROPHE THEORY TAXONOMY
# ============================================================================

CATASTROPHE_MORPHOLOGIES = {
    "fold": {
        "keywords": ["collapse", "fold", "crease", "smooth"],
        "description": "Smooth bifurcation with one control parameter",
        "visual_features": [
            "fold singularities", "cusp formations", "bifurcation surfaces",
            "smooth transitions", "crease patterns", "compression ridges"
        ],
        "surface_emphasis": ["edges", "ridges", "folds"],
        "optical_properties": ["matte", "diffuse", "soft_shadows"]
    },
    
    "cusp": {
        "keywords": ["cusp", "sharp", "geometric", "angular"],
        "description": "Sharp bifurcation with two control parameters",
        "visual_features": [
            "cusp points", "sharp edges", "angular boundaries",
            "geometric precision", "faceted surfaces", "crystalline sharpness"
        ],
        "surface_emphasis": ["vertices", "edges", "angles"],
        "optical_properties": ["specular", "reflective", "sharp_shadows"]
    },
    
    "swallowtail": {
        "keywords": ["tail", "swallow", "flowing", "dynamic"],
        "description": "Three control parameters with complex topology",
        "visual_features": [
            "flowing curves", "dynamic forms", "sweeping transitions",
            "organic trails", "momentum lines", "cascade effects"
        ],
        "surface_emphasis": ["curves", "flow", "transition"],
        "optical_properties": ["iridescent", "dynamic", "flowing_light"]
    },
    
    "butterfly": {
        "keywords": ["butterfly", "wings", "symmetry", "expansion"],
        "description": "Four control parameters with symmetrical expansion",
        "visual_features": [
            "bilateral symmetry", "wing-like structures", "radial expansion",
            "mirrored patterns", "growth formations", "fractal symmetry"
        ],
        "surface_emphasis": ["symmetry", "radiance", "expansion"],
        "optical_properties": ["iridescent", "symmetric", "radiant"]
    },
    
    "elliptic_umbilic": {
        "keywords": ["elliptic", "wave", "ripple", "undulate"],
        "description": "Wave-like bifurcations with continuous deformation",
        "visual_features": [
            "wave propagation", "ripple patterns", "undulating surfaces",
            "periodic structures", "oscillating forms", "harmonic waves"
        ],
        "surface_emphasis": ["waves", "ripples", "undulation"],
        "optical_properties": ["metallic", "reflective_waves", "shimmering"]
    },
    
    "hyperbolic_umbilic": {
        "keywords": ["hyperbolic", "saddle", "tension", "stress"],
        "description": "Saddle-point bifurcation with stress concentration",
        "visual_features": [
            "saddle formations", "stress concentration", "tension lines",
            "hyperbolic surfaces", "stress patterns", "directional forces"
        ],
        "surface_emphasis": ["stress", "tension", "direction"],
        "optical_properties": ["metallic", "directional", "tensioned"]
    }
}

# ============================================================================
# MCP SERVER SETUP
# ============================================================================

mcp = FastMCP("catastrophe-morph")

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

@mcp.tool()
def enhance_catastrophe_aesthetic(
    prompt: str,
    catastrophe_type: str = "auto",
    emphasis: str = "surface",
    intensity: str = "moderate"
) -> str:
    """
    Apply catastrophe theory aesthetic enhancements to a prompt.
    
    Augments a prompt with visual descriptors based on catastrophe theory
    morphologies, creating semantically coherent transformations.
    
    Args:
        prompt: Base prompt or previous layer output
        catastrophe_type: Type of catastrophe ("auto" to detect, or specific type)
        emphasis: Visual emphasis area ("surface", "edges", "light", "form")
        intensity: Enhancement intensity ("subtle", "moderate", "dramatic")
    
    Returns:
        Enhanced prompt with catastrophe aesthetic terms
    """
    
    if catastrophe_type == "auto":
        # Auto-detect from prompt
        prompt_lower = prompt.lower()
        scores = {
            cat_type: sum(1 for kw in data["keywords"] if kw in prompt_lower)
            for cat_type, data in CATASTROPHE_MORPHOLOGIES.items()
        }
        catastrophe_type = max(scores, key=scores.get) if any(scores.values()) else "fold"
    
    if catastrophe_type not in CATASTROPHE_MORPHOLOGIES:
        catastrophe_type = "fold"
    
    morph_data = CATASTROPHE_MORPHOLOGIES[catastrophe_type]
    
    # Select features based on emphasis and intensity
    intensity_map = {
        "subtle": 1,
        "moderate": 2,
        "dramatic": 3
    }
    num_features = intensity_map.get(intensity, 2)
    
    selected_features = random.sample(
        morph_data["visual_features"],
        min(num_features, len(morph_data["visual_features"]))
    )
    
    enhancement = f"Enhanced with {catastrophe_type} catastrophe - Adding: {', '.join(selected_features)}"
    return f"{prompt}\n{enhancement}"


@mcp.tool()
def map_catastrophe_parameters(
    prompt: str,
    intensity: float = 0.5
) -> dict:
    """
    Map aesthetic parameters from prompt to catastrophe theory space.
    
    Analyzes a prompt and returns structured parameters suitable for
    catastrophe theory visualization and transformation.
    
    Args:
        prompt: Text to analyze
        intensity: Parameter intensity (0.0-1.0)
    
    Returns:
        Dictionary with catastrophe parameters and control values
    """
    
    prompt_lower = prompt.lower()
    
    # Classify into catastrophe type
    scores = {
        cat_type: sum(1 for kw in data["keywords"] if kw in prompt_lower)
        for cat_type, data in CATASTROPHE_MORPHOLOGIES.items()
    }
    best_cat = max(scores, key=scores.get) if any(scores.values()) else "fold"
    
    return {
        "catastrophe_type": best_cat,
        "description": CATASTROPHE_MORPHOLOGIES[best_cat]["description"],
        "surface_emphasis": CATASTROPHE_MORPHOLOGIES[best_cat]["surface_emphasis"],
        "optical_properties": CATASTROPHE_MORPHOLOGIES[best_cat]["optical_properties"],
        "intensity": intensity,
        "visual_features": CATASTROPHE_MORPHOLOGIES[best_cat]["visual_features"]
    }


@mcp.tool()
def recommend_optical_treatment(catastrophe_type: str) -> dict:
    """
    Recommend optical treatment for a catastrophe morphology.
    
    Returns appropriate lighting, material, and visual treatments
    that complement the catastrophe aesthetic.
    
    Args:
        catastrophe_type: The catastrophe morphology type
    
    Returns:
        Optical treatment recommendations
    """
    
    if catastrophe_type not in CATASTROPHE_MORPHOLOGIES:
        catastrophe_type = "fold"
    
    morph = CATASTROPHE_MORPHOLOGIES[catastrophe_type]
    
    return {
        "catastrophe_type": catastrophe_type,
        "optical_properties": morph["optical_properties"],
        "recommended_lighting": "emphasize" if "sharp" in morph["optical_properties"] else "diffuse",
        "material_suggestion": "metallic" if "reflective" in str(morph["optical_properties"]) else "matte",
        "surface_treatment": morph["surface_emphasis"]
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    mcp.run()
