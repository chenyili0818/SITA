import json
from typing import Any, Dict, Iterable, Tuple
from openai import OpenAI
import re
import os

CONFIG: Dict[str, Any] = {
    "model_name": "deepseek-reasoner",
    "temperature": 0.5
}
output_root = "natural_language_output"

with open('./data/api/config.json', 'r', encoding='utf-8') as f:
    external_config = json.load(f)

CONFIG.update({
    "api_key": external_config.get("api_key"),
    "base_url": external_config.get("base_url")
})

client: OpenAI = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

TRANS_PROMPT_TEMPLATE = """Translate the following Lean4 code into a single LaTeX section with strict adherence to:

1. **Content Preservation**:
   - Maintain all definitions, lemmas, theorems in original order
   - Preserve proof structures exactly as in code
   - For 'sorry' proofs: Mark "Proof incomplete: [missing statement]"

2. **Proof Translation**:
   - Reconstruct every proof step in natural language
   - Convert tactics to mathematical reasoning:
     * "apply X" → "By application of X"
     * "rw Y" → "Rewriting using Y"
     * "exact Z" → "Direct application of Z"
   - Never complete partial proofs

3. **LaTeX Formatting**:
   - Output a SINGLE \section{...} environment
   - Use ONLY standard environments:
     \begin definition , \begin lemma , \begin theorem , \begin proof 
   - Avoid custom commands, packages, or environments
   - Use standard mathematical notation without new symbols

4. **Structural Constraints**:
   - No subsections or subsubsections
   - No itemize/enumerate lists
   - Maintain original component order as continuous text
   - Preserve all variable names exactly

Translate the following Lean code block into LaTeX:
```lean
{lean_code}
```"""
    
def translate_lean_content(lean_code: str) -> str:
    prompt = TRANS_PROMPT_TEMPLATE.replace("{lean_code}", lean_code)
    try:
        response = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=8000,
            temperature=CONFIG["temperature"],
        )
        return response.choices[0].message.content
    except Exception as exc:  
        print("DeepSeek API error: %s", exc)
        return ""

def slugify(text: str) -> str:
    """Convert *text* to a safe path fragment (alnum + underscore)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    result = slug or "unnamed"
    return result

if __name__ == "__main__":
    record = {"problem_name": "wavelet decomposition model", "problem_statement": "\\min_{d \\in \\mathbb{R}^n} \\|\\lambda \\odot d\\|_1 + \\frac{1}{2}\\|A W^{T} d - b\\|^2", "class": "Nesterov", "algorithm": "Use Nesterov acceleration with fixed \\(t_k = 1/L\\), where \\(L = \\|A W^T\\|^2\\) is the Lipschitz constant of \\(\\nabla f\\). Define:\n\\[\nz^{k}=(1-\\gamma_{k})d^{k-1}+\\gamma_{k}y^{k-1},\n\\]\n\\[\ny^{k}=\\operatorname{SoftThreshold}\\left(y^{k-1}-\\frac{t_{k}}{\\gamma_{k}}\\nabla f(z^{k}), \\frac{t_k}{\\gamma_k} \\lambda\\right),\n\\]\n\\[\nd^{k}=(1-\\gamma_{k})d^{k-1}+\\gamma_{k}y^{k}.\n\\]\nSoftThreshold operator: \\[ \\operatorname{SoftThreshold}(v, \\tau) = \\operatorname{sign}(v) \\cdot \\max(|v| - \\tau, 0). \\]"}
    problem = record.get("problem_statement", "")
    pname = slugify(record.get("problem_name", ""))
    pclass = slugify(record.get("class", ""))
    problem_description = f"Problem name: {pname}\nProblem statement: {problem}\nAlgorithm Class: {pclass}"
    
    sita_root = "lean/Optlib/Optbench/Solution/wavelet_decomposition_model_Nesterov_fixed_0.lean"
    with open(sita_root, 'r', encoding='utf-8') as f:
        lean_content = f.read()
    
    translate_output = translate_lean_content(lean_content)
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "translate_output.txt"), 'w', encoding='utf-8') as f:
        f.write(translate_output)
    