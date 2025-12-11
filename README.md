# CatastropheMorph MCP Server

Apply catastrophe theory aesthetic enhancements to prompts.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python3 catastrophe_morph.py
```

Or import directly:

```python
from catastrophe_morph import enhance_catastrophe_aesthetic

result = enhance_catastrophe_aesthetic(
    "base prompt",
    catastrophe_type="fold",
    emphasis="surface",
    intensity="dramatic"
)
```

## Tools

- **enhance_catastrophe_aesthetic(prompt, catastrophe_type, emphasis, intensity)** - Apply catastrophe morphology
- **map_catastrophe_parameters(prompt, intensity)** - Map aesthetic parameters
- **recommend_optical_treatment(catastrophe_type)** - Recommend lighting/materials

## Catastrophe Types

- fold
- cusp
- swallowtail
- butterfly
- elliptic_umbilic
- hyperbolic_umbilic

## License

MIT
