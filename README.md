# SITA: A Framework for Structure-to-Instance  Theorem Autoformalization

The project introduces SITA, a framework that automates the formalization of mathematical theorems in Lean by rigorously bridging abstract structures and their concrete instances in advanced mathematical domains.

## 📁 Project Structure

```
SITA/
├── Code/
│   ├── data/
│   │   ├── api/                 # Configuration for external model calls
│   │   ├── bugfix/              # Known Lean error patterns and fixes
│   │   ├── theorem_database/    # Theorem knowledge base (JSON)
|   |   └── problem/             # Test problem
│   └── lean/
│   │   ├── lakefile.lean        # Lean project entry
│   │   └── Optlib/
│   │       ├── Autoformalization/Example   # Autoformalization example
|   |       ├── Autoformalization/Template  # Autoformalization template
│   │       └── ...                         # Adapted from https://github.com/optsuite/optlib, 
|   |                                       # licensed under Apache 2.0 license as described in the file LICENSE_Optlib.
|   |                                       # Attribution will be restored after review.
|   └── src/                     # Generation Code
|   └── tool/model_downloader.py # Download the model needed
```

## 🚀 Getting Started

### Build & Run

```bash
cd SITA/Code/lean
lake build
```

To interact with the formalization:

```bash
lean --run Optlib/Autoformalization/Example/GD_example.lean
```

> Ensure all `.json` files in `data/` are accessible to any scripts or tools that perform autoformalization.

## 📚 Use Cases

- **Integration with LLMs**: (Suggested by `api/config.json`) for proof suggestion and statement completion
  
  run `Code/src/main.py` for generation.

