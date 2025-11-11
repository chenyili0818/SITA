from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from SITA.Code.src.rulebased_fixer import process_lean_file
from interact import compile_file, parse_lean_output_with_context

# ---------------------------------------------------------------------------
# Configuration (model + generation parameters)
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "max_workers": 16,
    "initial_generations": 3,  
    "proof_generations": 3,
    "model_name": "deepseek-reasoner",
    "temperature": 0.7
}

with open('./data/api/config.json', 'r', encoding='utf-8') as f:
    external_config = json.load(f)

CONFIG.update({
    "api_key": external_config.get("api_key"),
    "base_url": external_config.get("base_url")
})

# Initialize OpenAI client
client: OpenAI = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

# ---------------------------------------------------------------------------
# logger Setup
# ---------------------------------------------------------------------------

def read_text(path: os.PathLike | str, logger) -> str:
    """Return file contents or an empty string if the file is missing."""
    try:
        logger.debug(f"📖 Reading file: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        logger.debug(f"✅ Successfully read {len(content)} characters from {path}")
        return content
    except FileNotFoundError:
        logger.error(f"❌ File not found: {path}")
        return ""
    except Exception as e:
        logger.error(f"❌ Error reading file {path}: {e}")
        return ""

def write_text(path: os.PathLike | str, content: str, logger: logging.Logger) -> None:
    """Write content to file, creating directories as needed."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        logger.debug(f"💾 Written {len(content)} characters to {path}")
    except Exception as e:
        logger.error(f"❌ Error writing to file {path}: {e}")
        raise

def slugify(text: str, logger: logging.Logger) -> str:
    """Convert *text* to a safe path fragment (alnum + underscore)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    result = slug or "unnamed"
    logger.debug(f"🔄 Slugified '{text}' → '{result}'")
    return result

def load_problem_from_jsonl(jsonl_path: os.PathLike | str, logger: logging.Logger) -> list[dict]:
    """Load problems from JSONL file with progress tracking."""
    logger.info(f"📂 Loading problems from: {jsonl_path}")
    data_list = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line.strip():  # Skip empty lines
                    try:
                        data_list.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️  Skipping malformed JSON on line {line_num}: {e}")
        
        logger.info(f"✅ Successfully loaded {len(data_list)} problems")
        return data_list
    except Exception as e:
        logger.error(f"❌ Error loading JSONL file {jsonl_path}: {e}")
        return []

def get_lean_errors(file_path: os.PathLike | str, logger: logging.Logger) -> Tuple[list[dict], int]:
    """Compile `file_path` with Lean and return (errors, n_errors)."""
    logger.debug(f"🔍 Compiling Lean file: {file_path}")
    start_time = time.time()
    
    try:
        _stdout, json_info, _stderr = compile_file(str(file_path))
        _timings, errors = parse_lean_output_with_context(json_info)
        
        compilation_time = time.time() - start_time
        error_count = len(errors) if errors else 0
        
        if error_count == 0:
            logger.debug(f"✅ Lean compilation successful ({compilation_time:.2f}s): {file_path}")
        else:
            logger.debug(f"⚠️  Lean compilation found {error_count} errors ({compilation_time:.2f}s): {file_path}")
            
        return errors, error_count
    except Exception as e:
        logger.error(f"❌ Error compiling Lean file {file_path}: {e}")
        return [], 0

# ---------------------------------------------------------------------------
# OpenAI (DeepSeek) wrapper
# ---------------------------------------------------------------------------

def call_deepseek_api(prompt: str, logger: logging.Logger) -> str:
    """Call the DeepSeek API with retry logic and detailed logger."""
    logger.debug(f"🤖 Calling DeepSeek API with {len(prompt)} character prompt")
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            # max_tokens=16_000,
            temperature=CONFIG["temperature"],
        )
        
        api_time = time.time() - start_time
        content = response.choices[0].message.content
        
        logger.debug(f"✅ DeepSeek API call successful ({api_time:.2f}s), "
                     f"received {len(content) if content else 0} characters")
        
        # Log the complete model output to both file and console
        if content:
            logger.info("🤖 Model Response:")
            logger.info("=" * 60)
            logger.info(f"{content}")
            logger.info("=" * 60)
        
        return content or ""
    except Exception as exc:
        api_time = time.time() - start_time
        logger.error(f"❌ DeepSeek API error after {api_time:.2f}s: {exc}")
        return ""

def extract_lean_code(raw_output: str, logger: logging.Logger) -> str:
    """Strip ```lean blocks from LLM output (if any)."""
    logger.debug(f"🔧 Extracting Lean code from {len(raw_output)} character output")
    
    raw_output = raw_output.strip()
    original_length = len(raw_output)
    
    if raw_output.startswith("```lean4"):
        raw_output = raw_output[8:].lstrip()  # len("```lean4") == 8
    elif raw_output.startswith("```lean"):
        raw_output = raw_output[7:].lstrip()
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]
    
    extracted = raw_output.strip()
    logger.debug(f"🔧 Extracted {len(extracted)} characters "
                 f"(removed {original_length - len(extracted)} formatting chars)")
    
    return extracted

# ---------------------------------------------------------------------------
# Code generation pipeline (single problem)
# ---------------------------------------------------------------------------

def generate_initial_version(
    lean_structure: str,
    lean_example: str,
    problem: str,
    trial_path: os.PathLike | str,
    logger: logging.Logger
) -> Tuple[str | None, int, list[dict] | None]:
    """Generate one Lean file and return (code, error_count, error_json)."""
    logger.info(f"🎯 Generating initial version: {trial_path}")
    start_time = time.time()
    
    prompt = f"""
    As a mathematical formalization expert and Lean 4 programming expert, you possess extensive experience and deep understanding in formalizing optimization problems and are proficient in Lean 4 programming. You are capable of defining new classes, definitions, and instances in Lean 4 and can derive theorems for specific problems based on the general structure.

    You need to generate a complete Lean 4 formalization for a specific optimization problem instance. We already has a general formalization structure for the optimization method and the class of problems it applies to, and requires the creation of a formalization for a specific problem instance.

    Your task is to generate a complete Lean 4 formalization of this specific optimization problem instance, strictly following the structure and the style of the provided structure reference Lean 4 file. You need to define new classes for the optimization problems and methods, define suitable definitions based on the classes. Besides, you need to link the formalization of the specific problem to the structure reference using instance in Lean4. Finally, you need to state theorems specialized from the structure reference under the setting of the concrete problems, and try to prove it based on the structure reference. If you cannot prove it, just write `sorry` in the proof.

    The definition of the problem class should contain all the needed variables in ([name] : [Type]) format, no matter whether they are used in the properties here or not. Do not use `variable` block with explicit definitions. If you use "let" to define things, please give the corresponding type explicitly. You may need to use "let" to define some intermediate variables. Please note that the output of matrix vector multiplication is defined using type of "Fin n \\to ℝ". You may need to use "let" to give the type as EuclideanSpace ℝ (Fin n).

    Requirements:
    1. Strictly follow the structure of the reference file:
       - Problem definition (variables, objective function)
       - Algorithm implementation (parameters, iteration format)
       - Convergence theorem (statement only, proofs as `sorry`)
    2. **Replace All proofs with `sorry`**
    3. Preserve all mathematical notation, naming conventions, and code style from the reference.
    4. Ensure the generated code is syntactically correct Lean 4 code. Do not use functions not defined in Lean4.
    5. An example of a Lean 4 formalization of abstract method applied to the concrete problem is provided. Your output must imitate its structure.
    6. You should not add unneeded assumptions for the theorems.
    7. Use same imports and namespaces as reference. Do not change the imports. You should not need to repeat the template.

    Problem description:
    ```jsonl""" + problem + """
    ```

    Structure Reference Lean4 code:
    ```lean""" + lean_structure + """
    ```

    Example Lean4 code:
    ```lean""" + lean_example + """
    ```

    Output ONLY the complete Lean4 code WITHOUT any explanations.
    """

    logger.debug(f"📝 Sending prompt to API ({len(prompt)} chars)")
    raw_code = call_deepseek_api(prompt, logger)
    if not raw_code:
        logger.error(f"❌ API call failed for {trial_path}")
        return None, 0, None

    logger.debug(f"🔧 Processing generated code")
    clean_code = extract_lean_code(raw_code, logger)
    write_text(trial_path, clean_code, logger)
    
    logger.debug(f"🛠️  Processing Lean file with fix_code")
    process_lean_file(trial_path, inplace=True)  # Mutates file in‑place
    
    logger.debug(f"✅ Compiling generated file")
    errors, error_count = get_lean_errors(trial_path, logger)
    
    generation_time = time.time() - start_time
    if error_count == 0:
        logger.info(f"✅ Successfully generated error-free code ({generation_time:.2f}s): {trial_path}")
    else:
        logger.info(f"⚠️  Generated code with {error_count} errors ({generation_time:.2f}s): {trial_path}")
    
    return clean_code, error_count, errors

def analyze_sorry_positions(lean_content: str) -> Tuple[List[str], int]:
    """Analyze and extract sorry positions from Lean code."""
    import re
    
    # Find all sorry occurrences with context
    sorry_pattern = r'(theorem|lemma|def|instance)\s+([^:]+):\s*([^:=]+):=\s*sorry'
    matches = re.findall(sorry_pattern, lean_content, re.MULTILINE | re.DOTALL)
    
    sorry_contexts = []
    for match in matches:
        kind, name, type_sig = match
        name = name.strip()
        type_sig = type_sig.strip()
        sorry_contexts.append(f"{kind} {name} : {type_sig}")
    
    # Also find simple sorry patterns
    simple_sorry_pattern = r'sorry'
    simple_matches = re.findall(simple_sorry_pattern, lean_content)
    
    total_sorries = len(sorry_contexts) + len(simple_matches) - len(sorry_contexts)  # Avoid double counting
    
    return sorry_contexts, total_sorries

def validate_sorry_replacement(generated_code: str, original_sorry_count: int) -> Tuple[bool, int, str]:
    """Validate that sorry statements have been properly replaced."""
    import re
    
    # Count remaining sorry statements
    sorry_pattern = r'\bsorry\b'
    remaining_sorries = len(re.findall(sorry_pattern, generated_code))
    
    success = remaining_sorries == 0
    
    if success:
        result_msg = f"✅ All {original_sorry_count} sorry statements successfully replaced"
    else:
        result_msg = f"⚠️  Still has {remaining_sorries} sorry statements (started with {original_sorry_count})"
    
    return success, remaining_sorries, result_msg

def generate_initial_proof_version(
    lean_content : str,
    example_content : str,
    trial_path: os.PathLike | str,
    logger: logging.Logger
) -> Tuple[str | None, int, list[dict] | None]:
    """Generate one Lean file and return (code, error_count, error_json)."""
    logger.info(f"🎯 Generating proof version: {trial_path}")
    start_time = time.time()
    
    # Analyze sorry positions first
    sorry_contexts, total_sorries = analyze_sorry_positions(lean_content)
    logger.info(f"🔍 Found {total_sorries} sorry placeholders to replace")
    
    if total_sorries == 0:
        logger.info("✅ No sorry placeholders found - file may already be complete")
        write_text(trial_path, lean_content, logger)
        process_lean_file(trial_path, inplace=True)
        errors, error_count = get_lean_errors(trial_path, logger)
        return lean_content, error_count, errors
    
    # Create detailed sorry analysis for the prompt
    sorry_analysis = "\n".join([f"- {ctx}" for ctx in sorry_contexts])
    if sorry_analysis:
        sorry_section = f"\n\nSORRY ANALYSIS - These specific items need proofs:\n{sorry_analysis}"
    else:
        sorry_section = f"\n\nFound {total_sorries} sorry statements that need to be replaced with proofs."
    
    prompt = f"""
    As a mathematical formalization expert and Lean 4 programming expert, you possess extensive experience and deep understanding in formalizing optimization problems and are proficient in Lean 4 programming. You are proving lemmas and theorems in Lean4.

    **CRITICAL TASK**: You MUST replace ALL `sorry` placeholders with actual mathematical proofs. In Lean 4, `sorry` is a placeholder that should be replaced with real proofs. Your job is to provide complete, rigorous proofs for each theorem and lemma.

    **IMPORTANT**: Do NOT output `sorry` in your response. Every `sorry` you see must be replaced with a proper proof. If you cannot complete a proof, use tactics like `simp`, `rfl`, `trivial`, `assumption`, `exact`, or provide step-by-step proof tactics.
    {sorry_section}

    Requirements:
    1. **MANDATORY**: Replace every single `sorry` with actual proof tactics or proof terms
    2. Keep all other parts of the file unchanged (imports, definitions, theorem statements)
    3. Use the reference file to understand proof patterns and tactics
    4. Wrap your complete code in ```lean4 ``` block

    The file you need to prove (REPLACE ALL `sorry` WITH REAL PROOFS):
    ```lean""" + lean_content + """
    ```

    Structure Reference Lean4 code (for proof patterns and tactics):
    ```lean""" + example_content + """
    ```

    Remember: Your output must have ZERO `sorry` statements. Every theorem must have a complete proof!
    Output ONLY the complete Lean4 code WITHOUT any explanations.
    """

    logger.debug(f"📝 Sending proof prompt to API ({len(prompt)} chars)")
    raw_code = call_deepseek_api(prompt, logger)
    if not raw_code:
        logger.error(f"❌ Proof API call failed for {trial_path}")
        return None, 0, None

    logger.debug(f"🔧 Processing generated proof code")
    clean_code = extract_lean_code(raw_code, logger)
    
    # Validate sorry replacement before saving
    validation_success, remaining_sorries, validation_msg = validate_sorry_replacement(clean_code, total_sorries)
    logger.info(validation_msg)
    
    if not validation_success and remaining_sorries > 0:
        logger.warning(f"🔄 Generated code still contains {remaining_sorries} sorry statements - this may need manual review")
    
    write_text(trial_path, clean_code, logger)
    
    
    logger.debug(f"🛠️  Processing proof Lean file with fix_code")
    process_lean_file(trial_path, inplace=True)  # Mutates file in‑place
    
    logger.debug(f"✅ Compiling generated proof file")
    errors, error_count = get_lean_errors(trial_path, logger)
    
    generation_time = time.time() - start_time
    if error_count == 0:
        logger.info(f"✅ Successfully generated error-free proof ({generation_time:.2f}s): {trial_path}")
    else:
        logger.info(f"⚠️  Generated proof with {error_count} errors ({generation_time:.2f}s): {trial_path}")
    
    return clean_code, error_count, errors

# ---------------------------------------------------------------------------
# Parallel generation (keep best variant)
# ---------------------------------------------------------------------------

def generate_lean_code_parallel(
    *,
    problem: str,
    trial_root: os.PathLike | str,
    lean_example: str,
    lean_structure: str,
    logger: logging.Logger
):
    """Generate multiple variants in parallel and return the best one."""
    logger.info(f"🚀 Starting parallel generation for problem")
    logger.info(f"📂 Output directory: {trial_root}")
    logger.info(f"🔄 Generating {CONFIG['initial_generations']} variants")
    
    best_code: str | None = None
    best_errs = float("inf")
    start_time = time.time()

    os.makedirs(trial_root, exist_ok=True)
    max_workers = max(1, min(CONFIG["max_workers"], CONFIG["initial_generations"]))
    
    def _task(idx: int, logger: logging.Logger):
        file_path = Path(trial_root) / f"Trial{idx}.lean"
        logger.debug(f"🎯 Starting generation task {idx}")
        return idx, *generate_initial_version(lean_structure, lean_example, problem, file_path, logger)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, i + 1, logger): i + 1 for i in range(CONFIG["initial_generations"])}
        completed_count = 0
        
        for fut in as_completed(futures):
            idx, code, n_err, _ = fut.result()
            completed_count += 1
            
            if code is None:
                logger.warning(f"❌ Variant #{idx} failed to generate")
                continue
                
            logger.info(f"✅ Variant #{idx} completed → {n_err} error(s) "
                        f"({completed_count}/{CONFIG['initial_generations']})")
            
            if n_err < best_errs:
                best_code, best_errs = code, n_err
                logger.info(f"🏆 New best variant: #{idx} with {n_err} errors")
                
                if n_err == 0:
                    logger.info("🎉 Found error-free variant - this is optimal!")

    total_time = time.time() - start_time
    
    if best_code is None:
        logger.error(f"❌ All generation attempts failed after {total_time:.2f}s")
        return None
    
    logger.info(f"🏁 Generation completed in {total_time:.2f}s")
    logger.info(f"🏆 Best result: {best_errs} errors")
    
    return best_code

# ---------------------------------------------------------------------------
# Command‑line interface
# ---------------------------------------------------------------------------

def generate_lean_code_parallel_proof(
    *,
    trial_root: os.PathLike | str,
    lean_example: str,
    lean_content: str,
    logger: logging.Logger
):
    """Generate multiple proof variants in parallel and return the best one."""
    logger.info(f"🚀 Starting parallel proof generation")
    logger.info(f"📂 Output directory: {trial_root}")
    logger.info(f"🔄 Generating {CONFIG['proof_generations']} proof variants")
    
    best_code: str | None = None
    best_errs = float("inf")
    start_time = time.time()

    os.makedirs(trial_root, exist_ok=True)
    max_workers = max(1, min(CONFIG["max_workers"], CONFIG["proof_generations"]))

    def _task(idx: int, logger: logging.Logger):
        file_path = Path(trial_root) / f"ProofTrial{idx}.lean"
        logger.debug(f"🎯 Starting proof generation task {idx}")
        return idx, *generate_initial_proof_version(
            lean_content=lean_content,
            example_content=lean_example,
            trial_path=file_path,
            logger=logger
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, i + 1, logger): i + 1 for i in range(CONFIG["proof_generations"])}
        completed_count = 0
        
        for fut in as_completed(futures):
            idx, code, n_err, _ = fut.result()
            completed_count += 1
            
            if code is None:
                logger.warning(f"❌ Proof variant #{idx} failed to generate")
                continue
                
            logger.info(f"✅ Proof variant #{idx} completed → {n_err} error(s) "
                        f"({completed_count}/{CONFIG['proof_generations']})")
            
            if n_err < best_errs:
                best_code, best_errs = code, n_err
                logger.info(f"🏆 New best proof variant: #{idx} with {n_err} errors")
                
                if n_err == 0:
                    logger.info("🎉 Found error-free proof - this is optimal!")

    total_time = time.time() - start_time


def process_all_problems(problem_file: str, lean_root: str, root: str, example: str, structure: str, logger: logging.Logger) -> None:
    """Process all problems from a JSONL file with comprehensive logger."""
    # ---------------------------------------------------------------------
    # Setup logger
    # ---------------------------------------------------------------------
    
    logger.info(f"🎯 Starting batch processing of problems")
    logger.info(f"📁 Input file: {problem_file}")
    logger.info(f"📁 Output directory: {lean_root}")
    logger.info(f"📁 Example file: {example}")
    logger.info(f"📁 Structure file: {structure}")

    # ---------------------------------------------------------------------
    # Load problems
    # ---------------------------------------------------------------------
    start_total = time.time()
    records = load_problem_from_jsonl(problem_file, logger)
    if not records:
        logger.error(f"❌ No problems loaded from {problem_file}")
        return
    
    logger.info(f"📊 Processing {len(records)} problems")

    # Load reference files once
    logger.info(f"📖 Loading reference files...")
    lean_example_content = read_text(example, logger)
    lean_structure_content = read_text(structure, logger)
    
    if not lean_example_content:
        logger.error(f"❌ Failed to load example file: {example}")
        return
    if not lean_structure_content:
        logger.error(f"❌ Failed to load structure file: {structure}")
        return
    
    logger.info(f"✅ Reference files loaded successfully")

    # ---------------------------------------------------------------------
    # Define processing task
    # ---------------------------------------------------------------------
    def process_record(record_idx: int, record: dict, logger: logging.Logger):
        """Process a single problem record."""
        problem = record.get("problem_statement", "")
        pname = slugify(record.get("problem_name", f"problem_{record_idx}"), logger)
        pclass = slugify(record.get("class", ""), logger)
        
        logger.info(f"🔍 [{record_idx+1}/{len(records)}] Processing: {pname} (class: {pclass})")

        problem_description = f"Problem name: {pname}\nProblem statement: {problem}\nAlgorithm Class: {pclass}"
        trial_root = Path(lean_root) / f"{pname}_{pclass}"
        
        record_start = time.time()
        best = generate_lean_code_parallel(
            problem=problem_description,
            trial_root=trial_root,
            lean_example=lean_example_content,
            lean_structure=lean_structure_content,
            logger=logger
        )
        record_time = time.time() - record_start
        
        if best:
            logger.info(f"✅ [{record_idx+1}/{len(records)}] {pname} completed successfully ({record_time:.2f}s)")
            return {"status": "success", "time": record_time, "name": pname}
        else:
            logger.error(f"❌ [{record_idx+1}/{len(records)}] {pname} failed ({record_time:.2f}s)")
            return {"status": "failed", "time": record_time, "name": pname}

    # ---------------------------------------------------------------------
    # Parallel execution with progress tracking
    # ---------------------------------------------------------------------
    max_workers = CONFIG.get("max_workers", 8)
    logger.info(f"🚀 Starting parallel execution with {max_workers} workers")
    
    results = {"success": 0, "failed": 0, "skipped": 0, "total_time": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_record, idx, record, logger): idx 
            for idx, record in enumerate(records)
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results[result["status"]] += 1
                    if "time" in result:
                        results["total_time"] += result["time"]
            except Exception as e:
                logger.exception(f"❌ Exception during problem processing: {e}")
                results["failed"] += 1

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------
    total_time = time.time() - start_total
    logger.info("=" * 80)
    logger.info("🏁 BATCH PROCESSING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"📊 Results Summary:")
    logger.info(f"   ✅ Successful: {results['success']}")
    logger.info(f"   ❌ Failed: {results['failed']}")
    logger.info(f"   ⏭️  Skipped: {results['skipped']}")
    logger.info(f"   📝 Total processed: {len(records)}")
    logger.info(f"⏱️  Total time: {total_time:.2f}s")
    logger.info(f"⏱️  Average time per problem: {results['total_time']/max(1, results['success'] + results['failed']):.2f}s")
    logger.info("=" * 80)


def process_all_problems_proof(problem_file: str, root: str, example: str, logger: logging.Logger) -> None:
    """Process proof generation for a single Lean file with comprehensive logger."""
    # ---------------------------------------------------------------------
    # Setup logger  
    # ---------------------------------------------------------------------
    
    logger.info(f"🎯 Starting proof generation process")
    logger.info(f"📁 Input file: {problem_file}")
    logger.info(f"📁 Example file: {example}")
    logger.info(f"📁 Output root: {root}")

    # ---------------------------------------------------------------------
    # Validate input files
    # ---------------------------------------------------------------------
    problem_path = Path(problem_file)
    example_path = Path(example)
    
    if not problem_path.exists():
        logger.error(f"❌ Problem file not found: {problem_file}")
        return
    
    if not example_path.exists():
        logger.error(f"❌ Example file not found: {example}")
        return
    
    # ---------------------------------------------------------------------
    # Load reference content
    # ---------------------------------------------------------------------
    logger.info(f"📖 Loading reference files...")
    lean_content = read_text(problem_path, logger)
    example_content = read_text(example_path, logger)
    
    if not lean_content:
        logger.error(f"❌ Failed to load problem file: {problem_file}")
        return
    if not example_content:
        logger.error(f"❌ Failed to load example file: {example}")
        return
    
    logger.info(f"✅ Reference files loaded successfully")
    logger.info(f"📏 Problem file: {len(lean_content)} characters")
    logger.info(f"📏 Example file: {len(example_content)} characters")

    # ---------------------------------------------------------------------
    # Generate proof
    # ---------------------------------------------------------------------
    trial_root = problem_path.parent / "Proof"
    logger.info(f"📂 Proof output directory: {trial_root}")
    
    start_time = time.time()
    generation_start = time.time()
    generate_lean_code_parallel_proof(
        trial_root=trial_root,
        lean_example=example_content,
        lean_content=lean_content,
        logger=logger
    )
    generation_time = time.time() - generation_start

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------
    total_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("🏁 PROOF GENERATION COMPLETED")
    logger.info("=" * 80)
    
    logger.info(f"⏱️  Generation time: {generation_time:.2f}s")
    logger.info(f"⏱️  Total time: {total_time:.2f}s")
    logger.info("=" * 80)