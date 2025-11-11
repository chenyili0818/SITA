#!/usr/bin/env python3
"""
Main Pipeline for Lean Code Generation and Error Correction

This module orchestrates the complete pipeline for:
1. Generating Lean code from problem specifications
2. Correcting compilation errors
3. Processing proof generation
4. Managing file operations and workflow

Based on enhanced patterns from Bugfixnew.py and first_generation_new.py
"""

from __future__ import annotations
import argparse
import logging
import os
import shutil
import sys
import time
import re
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from SITA.Code.src.automatic_fixer import correct_all_trials_in_folder, fix_all_trials_in_folder
from SITA.Code.src.first_generation import process_all_problems, process_all_problems_proof
import concurrent.futures
# ---------------------------------------------------------------------------
# Configuration and Global Setup
# ---------------------------------------------------------------------------

# Pipeline Configuration
PIPELINE_CONFIG = {
    "max_workers": 16,  # Maximum parallel workers
    "default_example": "./lean/Optlib/Autoformalization/Example/GD_example.lean",
    "default_structure": "./lean/Optlib/Autoformalization/Template/GD_template.lean",
    "default_problem_file": "./data/problem/problem_test.jsonl",
    "default_lean_root": "./lean/Optlib/Autoformalization/R1",
    "default_results_root": "./results/R1",
    "log_model_output": True,  # Whether to log model outputs
}

def setup_logger(
    name: str = "LeanLogger",
    log_dir: str = "./logs",
    log_file: str = None,  # for explicit file path
    console: bool = True,
    level: int = logging.DEBUG
) -> logging.Logger:
    """
    Setup a logger with time-stamped log file.
    
    Args:
        name: logger name
        log_dir: directory for log file (if log_file not specified)
        log_file: full path to log file (overrides log_dir/name)
        console: whether to log to console
        level: logging level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # If logger already has handlers, return it
    if logger.handlers:
        return logger
    
    # Determine log file path
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = Path(log_dir) / f"{name}_{time_str}.log"
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"📄 Logging initialized → {log_path}")
    return logger

logger_cur = setup_logger(name="MainPipeline", log_dir=PIPELINE_CONFIG["default_results_root"]+"/run_logs")

# ---------------------------------------------------------------------------
# Enhanced Pipeline Functions
# ---------------------------------------------------------------------------
def process_proof_folder(proof_folder: str, logger) -> tuple[str, dict]:
    """
    Process a single 'Proof' folder.
    Returns a tuple of (folder_path, result_dict).
    """
    logger.info(f"🔍 Correcting proofs in: {proof_folder}")
    try:
        result = correct_all_trials_in_folder(proof_folder, logger)
        return proof_folder, {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Failed correction in {proof_folder}: {e}")
        return proof_folder, {"success": False, "error": str(e)}

def parallel_correct_all(current_directory: str, logger, max_workers: int = 8) -> dict:
    """
    Traverse subfolders in `current_directory`, find valid 'Proof' folders,
    and apply correction in parallel using threads.
    """
    second_correction_results = {}
    proof_folders = []

    # Step 1: Collect all proof folder paths
    for subfolder in os.listdir(current_directory):
        subfolder_path = os.path.join(current_directory, subfolder)
        proof_folder = os.path.join(subfolder_path, "Proof")
        if os.path.isdir(proof_folder):
            proof_folders.append(proof_folder)

    # Step 2: Run corrections in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {
            executor.submit(process_proof_folder, pf, logger): pf
            for pf in proof_folders
        }
        for future in concurrent.futures.as_completed(future_to_folder):
            folder, result = future.result()
            second_correction_results[folder] = result

    return second_correction_results

def process_all_lean_files_parallel(
    target_directory: Union[str, Path], 
    root: str, 
    example: str, 
    max_workers: int = 8,
    logger: logging.Logger = logger_cur
) -> List[Tuple[str, bool]]:
    """
    Process all Lean files in parallel with comprehensive logging and error handling.
    
    Args:
        target_directory: Directory containing Lean files to process
        root: Root directory for results
        example: Path to example Lean file
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of (file_path, success) tuples
    """
    target_directory = Path(target_directory)
    logger.info(f"🚀 Starting parallel Lean file processing")
    logger.info(f"📁 Target directory: {target_directory}")
    logger.info(f"📄 Example file: {example}")
    logger.info(f"👥 Max workers: {max_workers}")
    
    start_time = time.time()
    
    # Find all Lean files
    lean_files = []
    for subdir in target_directory.iterdir():
        if subdir.is_dir():
            candidates = list(subdir.rglob("*.lean"))
            if candidates:
                lean_files.append(random.choice(candidates))
    
    if not lean_files:
        logger.warning("⚠️  No .lean files found in target directory")
        return []
    
    logger.info(f"🔍 Found {len(lean_files)} Lean files to process")
    
    def process_single_file(file_path: Path, logger: logging.Logger = logger_cur) -> Tuple[str, bool]:
        """Process a single Lean file and return result."""
        # Extract problem ID from file path
        problem_id = file_path.stem
        
        file_start = time.time()
        
        try:
            logger.info(f"🎯 Processing file: {file_path}")
            logger.info(f"📂 Output root: {root}")
            logger.info(f"📄 Example file: {example}")
            
            process_all_problems_proof(str(file_path), root, example, logger)
            
            processing_time = time.time() - file_start
            logger.info(f"✅ Processing completed successfully in {processing_time:.2f}s")
            return str(file_path), True
            
        except Exception as e:
            processing_time = time.time() - file_start
            logger.error(f"❌ Processing failed after {processing_time:.2f}s")
            logger.error(f"💥 Error details: {str(e)}")
            return str(file_path), False
    
    # Parallel processing with progress tracking
    results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_path, logger): file_path 
            for file_path in lean_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                progress = (completed / len(lean_files)) * 100
                logger.info(f"📊 Progress: {completed}/{len(lean_files)} ({progress:.1f}%)")
                
            except Exception as e:
                logger.error(f"💥 Unexpected error processing {file_path}: {e}")
                results.append((str(file_path), False))
    
    # Final summary
    total_time = time.time() - start_time
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    logger.info("=" * 60)
    logger.info("🏁 PARALLEL PROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"📊 Total files: {len(lean_files)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success rate: {(successful/len(lean_files)*100):.1f}%")
    logger.info(f"⏱️  Total time: {total_time:.2f}s")
    logger.info(f"⏱️  Average time per file: {total_time/len(lean_files):.2f}s")
    logger.info("=" * 60)
    
    return results


def extract_fixed_files(source_dir: Union[str, Path], target_root_dir: Union[str, Path], logger: logging.Logger = logger_cur) -> int:
    """
    Extract all files ending with '_fixed.lean' from source directory and copy to target directory.
    
    Recursively searches for all files ending with `_fixed.lean` in source_dir and copies them
    to target_root_dir while preserving the relative directory structure.
    
    Args:
        source_dir: Source directory to search for fixed files
        target_root_dir: Target directory for copying files
        
    Returns:
        Number of files copied
    """
    source_dir = Path(source_dir).resolve()
    target_root_dir = Path(target_root_dir).resolve()
    
    logger.info(f"🔄 Starting file extraction")
    logger.info(f"📂 Source directory: {source_dir}")
    logger.info(f"📂 Target directory: {target_root_dir}")
    
    start_time = time.time()
    
    if not source_dir.exists():
        error_msg = f"Source directory not found: {source_dir}"
        logger.error(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # Find all fixed files
    logger.info(f"🔍 Searching for *_fixed.lean files...")
    fixed_files = list(source_dir.rglob("*_fixed.lean"))
    
    if not fixed_files:
        logger.warning(f"⚠️  No *_fixed.lean files found in {source_dir}")
        return 0
    
    logger.info(f"📄 Found {len(fixed_files)} fixed files to copy")
    
    count = 0
    for file in fixed_files:
        try:
            # Get relative path from source directory
            relative_path = file.relative_to(source_dir)
            # Construct target path
            dst_path = target_root_dir / relative_path
            
            # Create target directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file with metadata
            shutil.copy2(file, dst_path)
            logger.debug(f"✅ Copied: {file} → {dst_path}")
            count += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to copy {file}: {e}")
    
    extraction_time = time.time() - start_time
    logger.info(f"🎉 Finished: {count} files copied to '{target_root_dir}' in {extraction_time:.2f}s")
    
    return count

def collect_best_proof_trials(base_dir: Path, logger: logging.Logger = logger_cur) -> list:
    target_dir = base_dir / "ProofTrialFinal"
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Created/Using directory: {target_dir}")

    all_files = list(base_dir.glob("ProofTrial*.lean"))
    file_map = {}  # idx -> list of matching files

    for file in all_files:
        match = re.match(r"ProofTrial(\d+)(?:_fix(\d+)|_fixed)?\.lean", file.name)
        if match:
            idx = int(match.group(1))
            fix_version = match.group(2)
            file_map.setdefault(idx, []).append((file, fix_version))

    selected_files = []

    for idx, files in sorted(file_map.items()):
        fixed_file = None
        max_fix_file = None
        max_fix_number = -1
        original_file = None

        for file, fix_version in files:
            if "_fixed" in file.name:
                fixed_file = file
            elif fix_version is not None:
                fix_number = int(fix_version)
                if fix_number > max_fix_number:
                    max_fix_number = fix_number
                    max_fix_file = file
            else:
                original_file = file

        chosen = fixed_file or max_fix_file or original_file
        if chosen:
            dest = target_dir / chosen.name
            shutil.copy(chosen, dest)
            selected_files.append(dest)
            logger.info(f"✅ Selected for idx {idx}: {chosen.name}")

    logger.info(f"📦 Total selected files: {len(selected_files)}")
    return selected_files

# ---------------------------------------------------------------------------
# Main Pipeline Functions
# ---------------------------------------------------------------------------

class LeanPipeline:
    """
    Main pipeline orchestrator for Lean code generation and error correction.
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: logging.Logger = logger_cur):
        """Initialize pipeline with configuration."""
        self.config = {**PIPELINE_CONFIG, **(config or {})}
        self.logger = logger
        
        # Set up shared fix logger
        fix_log_dir = os.path.join(self.config["default_results_root"], "fix_logs")
        os.makedirs(fix_log_dir, exist_ok=True)
        self.fix_logger = setup_logger(
            name="FixLogger",
            log_file=os.path.join(fix_log_dir, "fixlog.log"),  # Fixed file path
            console=True
        )

        self.logger.info(f"🎯 Initializing Lean Pipeline")
        self.logger.info(f"⚙️  Configuration: {self.config}")
        self.logger.info(f"🔧 Fix logs will be saved to: {self.fix_logger.handlers[0].baseFilename}")

    def _process_fixed_files_only(self, fixed_files: List[Path], logger: logging.Logger = logger_cur) -> List[Tuple[str, bool]]:
        """
        Process only the fixed files from step 5 using harmless fixing.
        
        Args:
            fixed_files: List of *_fixed.lean files to process
            
        Returns:
            List of (file_path, success) tuples
        """
        from SITA.Code.src.automatic_fixer import LeanErrorCorrector
        
        self.logger.info(f"🚀 Starting harmless fixing for {len(fixed_files)} fixed files")
        start_time = time.time()
        
        results = []
        
        def process_single_fixed_file(file_path: Path) -> Tuple[str, bool]:
            """Process a single fixed file with harmless fixing."""
            file_start = time.time()
            
            try:
                self.logger.info(f"🎯 Processing fixed file: {file_path}")
                
                # First check if the file already compiles
                self.logger.info(f"🔍 Pre-checking compilation status...")
                check_start = time.time()
                
                try:
                    from interact import compile_file, parse_lean_output_with_context
                    _, info, _ = compile_file(str(file_path))
                    _, errors = parse_lean_output_with_context(info)
                    check_time = time.time() - check_start
                    
                    if not errors:
                        total_time = time.time() - file_start
                        self.logger.info(f"✅ Fixed file already compiles without errors ({check_time:.2f}s), skipping")
                        return (str(file_path), True)
                    else:
                        self.logger.info(f"⚠️  Found {len(errors)} errors ({check_time:.2f}s), proceeding with harmless fixing...")
                        
                except Exception as e:
                    check_time = time.time() - check_start
                    self.logger.warning(f"⚠️  Pre-compilation check failed ({check_time:.2f}s): {e}, proceeding...")

                # Apply harmless fixing
                corrector = LeanErrorCorrector(str(file_path), self.fix_logger)
                success = corrector.fix_errors()
                
                process_time = time.time() - file_start
                if success:
                    self.logger.info(f"✅ Harmless fixing completed in {process_time:.2f}s")
                else:
                    self.logger.warning(f"⚠️  Harmless fixing failed after {process_time:.2f}s")
                
                return (str(file_path), success)
                
            except Exception as e:
                process_time = time.time() - file_start
                self.logger.error(f"❌ Exception during harmless fixing ({process_time:.2f}s): {e}")
                return (str(file_path), False)
        
        # Process files in parallel
        max_workers = min(self.config["max_workers"], len(fixed_files))
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_fixed_file, file_path): file_path 
                for file_path in fixed_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    progress = (completed / len(fixed_files)) * 100
                    file_name = Path(result[0]).name
                    success_status = "✅ SUCCESS" if result[1] else "❌ FAILED"
                    self.logger.info(f"📊 Progress: {completed}/{len(fixed_files)} ({progress:.1f}%) - {success_status}: {file_name}")
                    
                except Exception as e:
                    self.logger.error(f"💥 Unexpected error processing {file_path}: {e}")
                    results.append((str(file_path), False))
        
        # Summary
        total_time = time.time() - start_time
        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful
        
        self.logger.info("=" * 60)
        self.logger.info("🏁 HARMLESS FIXING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Fixed files processed: {len(fixed_files)}")
        self.logger.info(f"✅ Successful: {successful}")
        self.logger.info(f"❌ Failed: {failed}")
        self.logger.info(f"📈 Success rate: {(successful/len(fixed_files)*100):.1f}%")
        self.logger.info(f"⏱️  Total time: {total_time:.2f}s")
        self.logger.info("=" * 60)
        
        return results
    
    def run_complete_pipeline(
        self,
        problem_file: Optional[str] = None,
        lean_root: Optional[str] = None,
        root: Optional[str] = None,
        example: Optional[str] = None,
        structure: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Execute the complete Lean generation and error correction pipeline.
        
        Args:
            problem_file: Path to JSONL file with problems
            lean_root: Root directory for generated Lean files
            root: Root directory for results
            example: Path to example Lean file
            structure: Path to structure template file
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        # Use defaults if not provided
        problem_file = problem_file or self.config["default_problem_file"]
        lean_root = lean_root or self.config["default_lean_root"]
        root = root or self.config["default_results_root"]
        example = example or self.config["default_example"]
        structure = structure or self.config["default_structure"]
        
        self.logger.info("🚀 Starting Complete Lean Pipeline")
        self.logger.info("=" * 80)
        pipeline_start = time.time()
        
        results = {
            "start_time": pipeline_start,
            "steps": {},
            "total_files_processed": 0,
            "success": False
        }
        
        try:
            # Step 1: Generate initial Lean code
            self.logger.info("📝 STEP 1: Generating initial Lean code")
            step_start = time.time()
            
            process_all_problems(
                problem_file=problem_file,
                lean_root=lean_root,
                root=root,
                example=example,
                structure=structure,
                logger=self.logger
            )
            
            step_time = time.time() - step_start
            results["steps"]["generation"] = {"time": step_time, "success": True}
            self.logger.info(f"✅ Step 1 completed in {step_time:.2f}s")
            
            # Step 2: First error correction pass
            self.logger.info("🔧 STEP 2: First error correction pass")
            step_start = time.time()
            
            correction_results = correct_all_trials_in_folder(lean_root, self.fix_logger)
            
            step_time = time.time() - step_start
            results["steps"]["first_correction"] = {
                "time": step_time, 
                "success": True,
                "results": correction_results
            }
            self.logger.info(f"✅ Step 2 completed in {step_time:.2f}s")
            
            # Step 3: Extract fixed files
            self.logger.info("📋 STEP 3: Extracting fixed files")
            step_start = time.time()
            
            lean_root_path = Path(lean_root)
            target_directory = lean_root_path.with_name(lean_root_path.name + "_fixed")
            copied_count = extract_fixed_files(lean_root, target_directory)
            
            step_time = time.time() - step_start
            results["steps"]["extraction"] = {
                "time": step_time, 
                "success": True,
                "files_copied": copied_count
            }
            self.logger.info(f"✅ Step 3 completed in {step_time:.2f}s")
            
            # Step 4: Generate proofs for fixed files
            self.logger.info("🎯 STEP 4: Generating proofs for fixed files")
            step_start = time.time()
            
            proof_results = process_all_lean_files_parallel(
                target_directory, root, example, self.config["max_workers"], self.logger
            )
            
            step_time = time.time() - step_start
            results["steps"]["proof_generation"] = {
                "time": step_time, 
                "success": True,
                "results": proof_results
            }
            self.logger.info(f"✅ Step 4 completed in {step_time:.2f}s")
            
            # Step 5: Second error correction pass
            self.logger.info("🔧 STEP 5: Second error correction pass")
            step_start = time.time()

            current_directory = target_directory

            second_correction_results = parallel_correct_all(current_directory, self.logger, max_workers=8)

            step_time = time.time() - step_start
            results["steps"]["second_correction"] = {
                "time": step_time,
                "success": all(v["success"] for v in second_correction_results.values()),
                "results": second_correction_results
            }
            self.logger.info(f"✅ Step 5 completed in {step_time:.2f}s")
            
            # Step 6: Final harmless fixing (only on step 5 results from subfolders)
            self.logger.info("🛠️  STEP 6: Final harmless fixing (per subfolder)")
            step_start = time.time()

            all_fixed_files = []

            for subdir in sorted(target_directory.iterdir()):
                if subdir.is_dir():
                    proof_dir = subdir / "Proof"
                    if proof_dir.exists() and proof_dir.is_dir():
                        self.logger.info(f"📂 Processing: {proof_dir}")
                        fixed_files = collect_best_proof_trials(proof_dir, self.logger)
                        self.logger.info(f"   🔍 Found {len(fixed_files)} files in {proof_dir.name}")
                        all_fixed_files.extend(fixed_files)
                    else:
                        self.logger.warning(f"⚠️ Skipping {subdir.name}: no 'Proof' subdirectory")

            self.logger.info(f"🔍 Total fixed files collected from all subfolders: {len(all_fixed_files)}")
            def harmless_wrapper(f: Path):
                try:
                    return self._process_fixed_files_only([f], self.logger)
                except Exception as e:
                    self.logger.error(f"❌ Harmless fixing failed for {f}: {e}")
                    return []

            if all_fixed_files:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(harmless_wrapper, f) for f in all_fixed_files]
                    harmless_results_nested = [f.result() for f in concurrent.futures.as_completed(futures)]
                    harmless_results = [item for sublist in harmless_results_nested for item in sublist]
            else:
                self.logger.warning("⚠️  No fixed files found in any subfolder, skipping harmless fixing")
                harmless_results = []

            # Step Summary
            step_time = time.time() - step_start
            results["steps"]["harmless_fixing"] = {
                "time": step_time,
                "success": True,
                "results": harmless_results,
                "files_processed": len(all_fixed_files)
            }

            self.logger.info(f"✅ Step 6 completed in {step_time:.2f}s")

            # Pipeline completion
            total_time = time.time() - pipeline_start
            results["total_time"] = total_time
            results["success"] = True

            # Final summary
            self.logger.info("=" * 80)
            self.logger.info("🏆 COMPLETE PIPELINE FINISHED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Total pipeline time: {total_time:.2f}s")
            self.logger.info(f"📊 Pipeline steps completed: {len(results['steps'])}")
            for step_name, step_data in results["steps"].items():
                self.logger.info(f"   • {step_name}: {step_data['time']:.2f}s")
            self.logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            total_time = time.time() - pipeline_start
            results["total_time"] = total_time
            results["error"] = str(e)
            results["success"] = False
            
            self.logger.error("💥 Pipeline failed with error:")
            self.logger.error(f"❌ {e}")
            self.logger.error(f"⏱️  Failed after {total_time:.2f}s")
            
            raise


# ---------------------------------------------------------------------------
# Command-line Interface
# ---------------------------------------------------------------------------

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Lean Code Generation and Error Correction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --run-pipeline                        # Run complete pipeline with defaults
  %(prog)s --problem-file data/problems.jsonl    # Use custom problem file
  %(prog)s --lean-root /path/to/lean --max-workers 16  # Custom settings
  %(prog)s --extract-only --source /path/source --target /path/target  # Extract only
        """
    )
    
    # Pipeline options
    parser.add_argument(
        "--run-pipeline", 
        action="store_true",
        help="Run the complete pipeline"
    )
    
    parser.add_argument(
        "--problem-file",
        type=str,
        default=PIPELINE_CONFIG["default_problem_file"],
        help=f"Path to JSONL problem file (default: {PIPELINE_CONFIG['default_problem_file']})"
    )
    
    parser.add_argument(
        "--lean-root",
        type=str, 
        default=PIPELINE_CONFIG["default_lean_root"],
        help=f"Root directory for Lean files (default: {PIPELINE_CONFIG['default_lean_root']})"
    )
    
    parser.add_argument(
        "--root",
        type=str,
        default=PIPELINE_CONFIG["default_results_root"], 
        help=f"Root directory for results (default: {PIPELINE_CONFIG['default_results_root']})"
    )
    
    parser.add_argument(
        "--example",
        type=str,
        default=PIPELINE_CONFIG["default_example"],
        help=f"Path to example Lean file (default: {PIPELINE_CONFIG['default_example']})"
    )
    
    parser.add_argument(
        "--structure", 
        type=str,
        default=PIPELINE_CONFIG["default_structure"],
        help=f"Path to structure template file (default: {PIPELINE_CONFIG['default_structure']})"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=PIPELINE_CONFIG["max_workers"],
        help=f"Maximum number of parallel workers (default: {PIPELINE_CONFIG['max_workers']})"
    )
    
    # Individual operation options
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract fixed files (requires --source and --target)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        help="Source directory for file extraction"
    )
    
    parser.add_argument(
        "--target", 
        type=str,
        help="Target directory for file extraction"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Enable quiet mode (errors only)"
    )
    
    parser.add_argument(
        "--log-model-output",
        action="store_true",
        help="Log model outputs to console (default: file only)"
    )
    
    return parser


def main() -> int:
    """Main function with command-line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()
    logger = logger_cur
    # Setup logging level based on arguments
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    try:
        if args.extract_only:
            # Handle extraction-only mode
            if not args.source or not args.target:
                logger.error("❌ --extract-only requires both --source and --target arguments")
                return 1
            
            logger.info(f"🔄 Running extraction-only mode")
            copied_count = extract_fixed_files(args.source, args.target)
            logger.info(f"✅ Extraction completed: {copied_count} files copied")
            return 0
            
        elif args.run_pipeline:
            # Handle complete pipeline mode
            config = {
                "max_workers": args.max_workers,
                "log_model_output": args.log_model_output
            }
            pipeline = LeanPipeline(config, logger)
            
            results = pipeline.run_complete_pipeline(
                problem_file=args.problem_file,
                lean_root=args.lean_root,
                root=args.root,
                example=args.example,
                structure=args.structure
            )
            
            if results["success"]:
                logger.info("🎉 Pipeline completed successfully!")
                return 0
            else:
                logger.error("💥 Pipeline failed!")
                return 1
        else:
            # No mode specified, show help
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        logger.warning("🛑 Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return 1


# ---------------------------------------------------------------------------
# Legacy Mode (for backward compatibility)
# ---------------------------------------------------------------------------

def run_legacy_pipeline(logger: logging.Logger = logger_cur):
    """Run the original pipeline for backward compatibility."""
    
    # Original hardcoded values
    problem_file = "./data/problem/problem_test.jsonl"
    lean_root = "lean/Optlib/Autoformalization/R1"
    root = "./results/R1"
    example = "./lean/Optlib/Autoformalization/Example/GD_example.lean"
    structure = "./lean/Optlib/Autoformalization/Template/GD_template.lean"
    
    pipeline = LeanPipeline(config=PIPELINE_CONFIG, logger=logger)
    results = pipeline.run_complete_pipeline(
        problem_file=problem_file,
        lean_root=lean_root,
        root=root,
        example=example,
        structure=structure
    )
    
    return results


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        exit_code = main()
        sys.exit(exit_code)
    else:
        # Legacy mode when run without arguments
        try:
            logger: logging.Logger = logger_cur
            results = run_legacy_pipeline(logger)
            if results["success"]:
                logger.info("🎉 Legacy pipeline completed successfully!")
            else:
                logger.error("💥 Legacy pipeline failed!")
                sys.exit(1)
        except Exception as e:
            logger.error(f"💥 Legacy pipeline error: {e}")
            sys.exit(1)
