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
##DeepSeek-R1
CONFIG: Dict[str, Any] = {
    "max_workers": 6,
    "initial_generations": 5, 
    "proof_generations": 1,
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

def setup_logging(log_dir: Path, log_name: str = "generation") -> Path:
    """Setup comprehensive logging with both file and console output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_name}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12s] [%(levelname)-8s] %(name)s: %(message)s"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s"
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simplified)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    logging.info("=" * 80)
    logging.info("🚀 Lean Code Generation Pipeline Started")
    logging.info("=" * 80)
    logging.info(f"📝 Log file: {log_path}")
    logging.info(f"⚙️  Configuration: {CONFIG}")
    
    return log_path

def read_text(path: os.PathLike | str) -> str:
    """Return file contents or an empty string if the file is missing."""
    try:
        logging.debug(f"📖 Reading file: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        logging.debug(f"✅ Successfully read {len(content)} characters from {path}")
        return content
    except FileNotFoundError:
        logging.error(f"❌ File not found: {path}")
        return ""
    except Exception as e:
        logging.error(f"❌ Error reading file {path}: {e}")
        return ""

def write_text(path: os.PathLike | str, content: str) -> None:
    """Write content to file, creating directories as needed."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        logging.debug(f"💾 Written {len(content)} characters to {path}")
    except Exception as e:
        logging.error(f"❌ Error writing to file {path}: {e}")
        raise

def slugify(text: str) -> str:
    """Convert *text* to a safe path fragment (alnum + underscore)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    result = slug or "unnamed"
    logging.debug(f"🔄 Slugified '{text}' → '{result}'")
    return result

def load_problem_from_jsonl(jsonl_path: os.PathLike | str) -> list[dict]:
    """Load problems from JSONL file with progress tracking."""
    logging.info(f"📂 Loading problems from: {jsonl_path}")
    data_list = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line.strip():  # Skip empty lines
                    try:
                        data_list.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logging.warning(f"⚠️  Skipping malformed JSON on line {line_num}: {e}")
        
        logging.info(f"✅ Successfully loaded {len(data_list)} problems")
        return data_list
    except Exception as e:
        logging.error(f"❌ Error loading JSONL file {jsonl_path}: {e}")
        return []

def get_lean_errors(file_path: os.PathLike | str) -> Tuple[list[dict], int]:
    """Compile `file_path` with Lean and return (errors, n_errors)."""
    logging.debug(f"🔍 Compiling Lean file: {file_path}")
    start_time = time.time()
    
    try:
        _stdout, json_info, _stderr = compile_file(str(file_path))
        _timings, errors = parse_lean_output_with_context(json_info)
        
        compilation_time = time.time() - start_time
        error_count = len(errors) if errors else 0
        
        if error_count == 0:
            logging.debug(f"✅ Lean compilation successful ({compilation_time:.2f}s): {file_path}")
        else:
            logging.debug(f"⚠️  Lean compilation found {error_count} errors ({compilation_time:.2f}s): {file_path}")
            
        return errors, error_count
    except Exception as e:
        logging.error(f"❌ Error compiling Lean file {file_path}: {e}")
        return [], 0

# ---------------------------------------------------------------------------
# OpenAI (DeepSeek) wrapper
# ---------------------------------------------------------------------------

def call_deepseek_api(prompt: str) -> str:
    """Call the DeepSeek API with retry logic and detailed logging."""
    logging.debug(f"🤖 Calling DeepSeek API with {len(prompt)} character prompt")
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=16_000,
            temperature=CONFIG["temperature"],
        )
        
        api_time = time.time() - start_time
        content = response.choices[0].message.content
        
        logging.debug(f"✅ DeepSeek API call successful ({api_time:.2f}s), "
                     f"received {len(content) if content else 0} characters")
        
        # Log the complete model output to both file and console
        if content:
            logging.info("🤖 Model Response:")
            logging.info("=" * 60)
            logging.info(f"{content}")
            logging.info("=" * 60)
        
        return content or ""
    except Exception as exc:
        api_time = time.time() - start_time
        logging.error(f"❌ DeepSeek API error after {api_time:.2f}s: {exc}")
        return ""

def extract_lean_code(raw_output: str) -> str:
    """Strip ```lean blocks from LLM output (if any)."""
    logging.debug(f"🔧 Extracting Lean code from {len(raw_output)} character output")
    
    raw_output = raw_output.strip()
    original_length = len(raw_output)
    
    if raw_output.startswith("```lean4"):
        raw_output = raw_output[8:].lstrip()  # len("```lean4") == 8
    elif raw_output.startswith("```lean"):
        raw_output = raw_output[7:].lstrip()
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]
    
    extracted = raw_output.strip()
    logging.debug(f"🔧 Extracted {len(extracted)} characters "
                 f"(removed {original_length - len(extracted)} formatting chars)")
    
    return extracted

# ---------------------------------------------------------------------------
# Code generation pipeline (single problem)  ─────────────────────────────────
# ---------------------------------------------------------------------------

def generate_initial_version(
    lean_structure: str,
    problem: str,
    trial_path: os.PathLike | str,
) -> Tuple[str | None, int, list[dict] | None]:
    """Generate one Lean file and return (code, error_count, error_json)."""
    logging.info(f"🎯 Generating initial version: {trial_path}")
    start_time = time.time()
    
    prompt = f"""
    As a mathematical formalization expert and Lean 4 programming expert, you possess extensive experience and deep understanding in formalizing optimization problems and are proficient in Lean 4 programming. You are capable of defining new classes, definitions, and instances in Lean 4 and can derive theorems for specific problems based on the general structure.

    You need to generate a complete Lean 4 formalization for a specific optimization problem instance. We already has a general formalization structure for the optimization method and the class of problems it applies to, and requires the creation of a formalization for a specific problem instance.

    Your task is to generate a complete Lean 4 formalization of this specific optimization problem instance, strictly following the structure and the style of the provided structure reference Lean 4 file. You need to define new classes for the optimization problems and methods, define suitable definitions based on the classes. Besides, you need to link the formalization of the specific problem to the structure reference using instance in Lean4. Finally, you need to state theorems specialized from the structure reference under the setting of the concrete problems, and try to prove it based on the theorem of the structure reference. If you cannot prove it, just write `sorry` in the proof.

    The definition of the problem class should contain all the needed variables in ([name] : [Type]) format, no matter whether they are used in the properties here or not. Do not use `variable` block with explicit definitions. If you use "let" to define things, please give the corresponding type explicitly. You may need to use "let" to define some intermediate variables. Please note that the output of matrix vector multiplication is defined using type of "Fin n \\to ℝ". You may need to use "let" to give the type as EuclideanSpace ℝ (Fin n).

    Requirements:
    1. Strictly follow the following structure:
       - Problem definition (variables, objective function)
       - Algorithm implementation (parameters, iteration format)
       - Instance linking to the structure reference
       - Convergence theorem (prove base on the structure reference)
    2. Preserve all mathematical notation, naming conventions, and code style from the reference.
    3. Ensure the generated code is syntactically correct Lean 4 code. Do not use functions not defined in Lean4.
    4. You should not add unneeded assumptions for the theorems.
    5. You do not need to repeat the structure reference in the output.
    6. Use same imports and namespaces as reference. Do not change the imports. You should not need to repeat the template.

    Problem description:
    ```""" + problem + """
    ```

    Structure Reference Lean4 code:
    ```lean""" + lean_structure + """
    ```

    Output ONLY the complete Lean4 code WITHOUT any explanations.
    """

    logging.debug(f"📝 Sending prompt to API ({len(prompt)} chars)")
    raw_code = call_deepseek_api(prompt)
    if not raw_code:
        logging.error(f"❌ API call failed for {trial_path}")
        return None, 0, None

    logging.debug(f"🔧 Processing generated code")
    clean_code = extract_lean_code(raw_code)
    write_text(trial_path, clean_code)
    
    logging.debug(f"🛠️  Processing Lean file with fix_code")
    process_lean_file(trial_path, inplace=True)  # Mutates file in‑place
    
    logging.debug(f"✅ Compiling generated file")
    errors, error_count = get_lean_errors(trial_path)
    
    generation_time = time.time() - start_time
    if error_count == 0:
        logging.info(f"✅ Successfully generated error-free code ({generation_time:.2f}s): {trial_path}")
    else:
        logging.info(f"⚠️  Generated code with {error_count} errors ({generation_time:.2f}s): {trial_path}")
    
    return clean_code, error_count, errors

def generate_lean_code_parallel(
    *,
    problem: str,
    trial_root: os.PathLike | str,
    lean_structure: str,
):
    """Generate multiple variants in parallel and return the best one."""
    logging.info(f"🚀 Starting parallel generation for problem")
    logging.info(f"📂 Output directory: {trial_root}")
    existing_trials = len(list(Path(trial_root).glob("Trial*.lean")))
    remaining_generations = max(0, CONFIG["initial_generations"] - existing_trials)

    if remaining_generations == 0:
        logging.info(f"✅ All {CONFIG['initial_generations']} trials already exist, skipping generation")
        return None
    logging.info(f"📄 Found {existing_trials} existing trials")
    logging.info(f"🔄 Generating {remaining_generations} variants")
    
    best_code: str | None = None
    best_errs = float("inf")
    start_time = time.time()

    os.makedirs(trial_root, exist_ok=True)
    max_workers = max(1, min(CONFIG["max_workers"], remaining_generations))
    
    def _task(idx: int):
        file_path = Path(trial_root) / f"Trial{idx}.lean"
        logging.debug(f"🎯 Starting generation task {idx}")
        return idx, *generate_initial_version(lean_structure, problem, file_path)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, i + 1): i + 1 for i in range(remaining_generations)}
        completed_count = 0
        
        for fut in as_completed(futures):
            idx, code, n_err, _ = fut.result()
            completed_count += 1
            
            if code is None:
                logging.warning(f"❌ Variant #{idx} failed to generate")
                continue
                
            logging.info(f"✅ Variant #{idx} completed → {n_err} error(s) "
                        f"({completed_count}/{CONFIG['initial_generations']})")
            
            if n_err < best_errs:
                best_code, best_errs = code, n_err
                logging.info(f"🏆 New best variant: #{idx} with {n_err} errors")
                
                if n_err == 0:
                    logging.info("🎉 Found error-free variant - this is optimal!")

    total_time = time.time() - start_time
    
    if best_code is None:
        logging.error(f"❌ All generation attempts failed after {total_time:.2f}s")
        return None
    
    logging.info(f"🏁 Generation completed in {total_time:.2f}s")
    logging.info(f"🏆 Best result: {best_errs} errors")
    
    return best_code

# ---------------------------------------------------------------------------
# Command‑line interface
# ---------------------------------------------------------------------------

def process_all_problems(problem_file: str, lean_root: str, root: str, structure: str) -> None:
    """Process all problems from a JSONL file with comprehensive logging."""
    # ---------------------------------------------------------------------
    # Setup logging
    # ---------------------------------------------------------------------
    log_dir = Path(root) / "generation_logs"
    log_path = setup_logging(log_dir, "process_all_problems")
    
    logging.info(f"🎯 Starting batch processing of problems")
    logging.info(f"📁 Input file: {problem_file}")
    logging.info(f"📁 Output directory: {lean_root}")
    logging.info(f"📁 Structure dir: {structure}")

    # ---------------------------------------------------------------------
    # Load problems
    # ---------------------------------------------------------------------
    start_total = time.time()
    records = load_problem_from_jsonl(problem_file)
    if not records:
        logging.error(f"❌ No problems loaded from {problem_file}")
        return
    
    logging.info(f"📊 Processing {len(records)} problems")

    # Load reference files once
    logging.info(f"📖 Loading reference files...")

    # ---------------------------------------------------------------------
    # Define processing task
    # ---------------------------------------------------------------------
    def process_record(record_idx: int, record: dict, structure: str = structure):
        """Process a single problem record."""
        problem = record.get("problem_statement", "")
        pname = slugify(record.get("problem_name", f"problem_{record_idx}"))
        pclass = slugify(record.get("class", ""))
        template_map = {
            "proximal gradient": "PGD_template.lean",
            "gradient descent": "GD_template.lean",
            "nesterov": "NesterovFirst_template.lean",
            "bcd": "BCD_template.lean",
            "admm": "ADMM_template.lean"
        }
        class_name = record.get("class", "").strip().lower()
        template_file = template_map.get(class_name)
        if template_file:
            structure = os.path.join(structure, template_file)
        lean_structure_content = read_text(structure)
    
        if not lean_structure_content:
          logging.error(f"❌ Failed to load structure file: {structure}")
          return
    
        logging.info(f"✅ Reference files loaded successfully")
        logging.info(f"🔍 [{record_idx+1}/{len(records)}] Processing: {pname} (class: {pclass})")

        problem_description = f"Problem name: {pname}\nProblem statement: {problem}\nAlgorithm Class: {pclass}"
        trial_root = Path(lean_root) / f"{pname}_{pclass}"
        
        record_start = time.time()
        best = generate_lean_code_parallel(
            problem=problem_description,
            trial_root=trial_root,
            lean_structure=lean_structure_content,
        )
        record_time = time.time() - record_start
        
        if best:
            logging.info(f"✅ [{record_idx+1}/{len(records)}] {pname} completed successfully ({record_time:.2f}s)")
            return {"status": "success", "time": record_time, "name": pname}
        else:
            logging.error(f"❌ [{record_idx+1}/{len(records)}] {pname} failed ({record_time:.2f}s)")
            return {"status": "failed", "time": record_time, "name": pname}

    # ---------------------------------------------------------------------
    # Parallel execution with progress tracking
    # ---------------------------------------------------------------------
    max_workers = CONFIG.get("max_workers", 8)
    logging.info(f"🚀 Starting parallel execution with {max_workers} workers")
    
    results = {"success": 0, "failed": 0, "skipped": 0, "total_time": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_record, idx, record, structure): idx 
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
                logging.exception(f"❌ Exception during problem processing: {e}")
                results["failed"] += 1

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------
    total_time = time.time() - start_total
    logging.info("=" * 80)
    logging.info("🏁 BATCH PROCESSING COMPLETED")
    logging.info("=" * 80)
    logging.info(f"📊 Results Summary:")
    logging.info(f"   ✅ Successful: {results['success']}")
    logging.info(f"   ❌ Failed: {results['failed']}")
    logging.info(f"   ⏭️  Skipped: {results['skipped']}")
    logging.info(f"   📝 Total processed: {len(records)}")
    logging.info(f"⏱️  Total time: {total_time:.2f}s")
    logging.info(f"⏱️  Average time per problem: {results['total_time']/max(1, results['success'] + results['failed']):.2f}s")
    logging.info(f"📝 Log file: {log_path}")
    logging.info("=" * 80)

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    
    problem_file = "data/problem/problem_test.jsonl"
    
    process_all_problems(
        problem_file=problem_file,
        lean_root="lean/Optlib/Autoformalization/Direct/R1",
        root="results/Direct/R1",
        structure="lean/Optlib/Autoformalization/Template",
    )