# SITA: A Framework for Structure-to-Instance Theorem Autoformalization

SITA is a framework that automates the formalization of mathematical theorems in Lean by bridging abstract structures and their concrete instances.

## ✨ Highlights

- Accepted at AAAI 2026.
- Structure-to-instance formalization workflow for Lean.
- Template-driven abstraction to reuse formalized structures.
- LLM-assisted generation with feedback-guided refinement.

## 🧭 Pipeline

<p align="center">
  <img src="fig/pipeline.svg" alt="SITA Pipeline" width="85%">
</p>

The pipeline starts from informal theorem statements, applies structure-aware parsing and template matching, and then performs LLM-assisted generation with rule-based fixes and verification feedback.

## 📁 Project Structure

```
SITA/
├── fig/
│   ├── pipeline.pdf            # Original pipeline diagram
│   └── pipeline.svg            # Vector image for README
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
|   └── src/                     # Generation code and pipeline modules
|   |   ├── main.py              # End-to-end entry point
|   |   ├── first_generation.py  # Initial formalization pass
|   |   ├── majorty_voting.py    # Voting-based validation
|   |   └── rulebased_fixer.py   # Rule-based Lean fixes
|   └── tool/model_downloader.py # Download the model needed
```

## Use Case

One representative structure-to-instance use case is **Lasso + Proximal Gradient Method (PGM)**:

- **Step 1: Abstract structure in Lean templates.**  
  SITA starts from a formalized composite optimization template:
  - problem class: `composite_pro`  
    This describes the optimization problem form itself.
  - algorithm class: `pg`  
    This defines the proximal-gradient iteration scheme.
  - abstract theorem: `pg_method_converge` (under convexity + smoothness + step-size assumptions)  
    This is the generic convergence guarantee for the abstract `pg` method, reusable across instances once assumptions are verified.

- **Step 2: Concrete instance for Lasso.**  
  Lasso objective is formalized as

  $$
  \min_x \frac{1}{2}\|Ax-b\|^2 + \mu\|x\|_1
  $$

  with explicit proximal-gradient update

  $$
  z = x_k - t A^\top (Ax_k-b),
  $$

  $$
  (x_{k+1})_i = \text{sign}(z_i)\max(|z_i|-t\mu,0).
  $$

  In code, this corresponds to instance-specific classes such as `Lasso_pro` and `pg_Lasso`.

- **Step 3: Structure-to-instance linking.**  
  SITA generates instance declarations from Lasso classes to abstract classes (e.g., from `Lasso_pro` to `composite_pro`, and from `pg_Lasso` to `pg`) so that generic theorems become reusable.

- **Step 4: Theorem transfer.**  
  After proving required assumptions (e.g., convexity and Lipschitz gradient lemmas), SITA applies the abstract theorem directly to obtain the instance theorem `Lasso_convergence`.

This is the core idea of SITA: verify assumptions once per instance, then reuse formalized abstract theorems instead of reproving from scratch.

## 📊 Core Results

Table header meanings:
- `Def`: percentage of syntactically correct generated definitions.
- `Thm`: percentage of syntactically correct theorem statements.
- `Instance`: percentage of syntactically correct Lean instance declarations.
- `File`: percentage of end-to-end successful whole-file generation.
- `MV`: majority-voting based semantic score used in the paper.

| Method | Def | Thm | Instance | File | MV |
| --- | --- | --- | --- | --- | --- |
| Direct-V3 | 27.9% | 28.0% | 22.8% | 0.0% | 50.2 |
| Direct-R1 | 62.8% | 25.6% | 25.7% | 0.0% | 46.0 |
| SITA-V3 | 91.0% | 86.7% | 90.8% | 27.2% | 66.1 |
| SITA-R1 | 93.8% | 95.6% | 95.4% | 57.14% | 76.9 |

Interpretation:
- SITA significantly improves all structure-level components (`Def`, `Thm`, `Instance`) over direct generation.
- The biggest gap is at full-file level: direct generation is `0.0%`, while SITA-R1 reaches `57.14%`, showing that staged generation + error-fix + proof refinement is critical for end-to-end Lean files.

## 🚀 Getting Started

### Build

```bash
# Build the Lean project (run from repo root).
cd SITA/Code/lean
lake build
```

> Notes:
> - Ensure all `.json` files in `Code/data/` are accessible to any scripts or tools that perform autoformalization.
> - If you have not downloaded the model needed for error classification yet, run `python Code/tool/model_downloader.py` first.

## 🧪 Usage

- Integration with LLMs: (by `api/config.json`) for statement completion and proof suggestion.

- Run generation:

  ```bash
  # Generates formalizations based on config and input problems.
  python Code/src/main.py
  ```

## Contact

- Chenyi Li: `lichenyi@stu.pku.edu.cn`
- Zaiwen Wen: `wenzw at pku dot edu dot cn`

## Citation

If you find our paper or our code useful, we would appreciate it if you could cite our work:

```bibtex
@misc{li2025sita,
  title={SITA: A Framework for Structure-to-Instance Theorem Autoformalization}, 
  author={Chenyi Li and Wanli Ma and Zichen Wang and Zaiwen Wen},
  year={2025},
  eprint={2511.10356},
  archivePrefix={arXiv},
}
```

## ⚖️ License

This project is released under the MIT License. See `LICENSE`.
