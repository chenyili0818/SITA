from openai import OpenAI
import asyncio
from typing import List, Tuple, Dict, Any
import json 
import os
import re 
import time
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from interact import compile_file, parse_lean_output_with_context
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

CONFIG: Dict[str, Any] = {
    "max_workers": 16,
    "initial_generations": 1,  # Number of parallel API calls per problem
    "proof_generations": 1,
    "model_name": "deepseek-chat",
    "temperature": 0.7
}

with open('./data/api/config.json', 'r', encoding='utf-8') as f:
    external_config = json.load(f)

CONFIG.update({
    "api_key": external_config.get("api_key"),
    "base_url": external_config.get("base_url")
})

def slugify(text: str) -> str:
    """Convert *text* to a safe path fragment (alnum + underscore)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    result = slug or "unnamed"
    return result

def load_problem_from_jsonl(jsonl_path: os.PathLike | str) -> list[dict]:
    """Load problems from JSONL file with progress tracking."""
    print(f"📂 Loading problems from: {jsonl_path}")
    data_list = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line.strip():  # Skip empty lines
                    try:
                        data_list.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping malformed JSON on line {line_num}: {e}")
        
        return data_list
    except Exception as e:
        print(f"❌ Error loading JSONL file {jsonl_path}: {e}")
        return []

def make_problem_description(candidate_filestr: str, problem_jsonl: str) -> Tuple[str, str, str]:
    parts = candidate_filestr.split(os.sep)
    try:
        sita_idx = parts.index("Autoformalization")
    except ValueError:
        raise ValueError("Path must contain 'Autoformalization'")

    normalized_path = slugify(candidate_filestr)
    data = load_problem_from_jsonl(problem_jsonl)

    # Build mapping: (problem_slug, class_slug) -> entry
    slug_to_entry = {
        (slugify(entry.get("problem_name", "")), slugify(entry.get("class", ""))): entry
        for entry in data
    }

    # Try to match one where both slugified problem_name and class exist in the path
    for (pname_slug, class_slug), entry in slug_to_entry.items():
        if pname_slug in normalized_path and class_slug in normalized_path:
            stmt = entry.get("problem_statement", "").strip()
            problem_description = f"""
            Problem name: {pname_slug}
            Problem statement: {stmt}
            Algorithm Class: {class_slug}
            """
            return pname_slug, class_slug, problem_description

    raise ValueError("No matching problem_name and class pair found")

def read_file_content(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not find：{path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
client: OpenAI = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

def call_deepseek_api(prompt: str) -> str:
    """Call the DeepSeek API with retry logic and detailed logging."""
    print(f"🤖 Calling DeepSeek API with {len(prompt)} character prompt")
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=1000,
            temperature=CONFIG["temperature"],
        )
        
        api_time = time.time() - start_time
        content = response.choices[0].message.content
        
        print(f"✅ DeepSeek API call successful ({api_time:.2f}s), "
                     f"received {len(content) if content else 0} characters")
        
        # Log the complete model output to both file and console
        if content:
            print("🤖 Model Response:")
            print("=" * 60)
            print(f"{content}")
            print("=" * 60)
        
        return content or ""
    except Exception as exc:
        api_time = time.time() - start_time
        print(f"❌ DeepSeek API error after {api_time:.2f}s: {exc}")
        return ""

def parse_score(text: str) -> float:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = re.search(r"Score[:：]?\s*(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(1))

    raise ValueError(f"Error when getting the score: {text}")

def score_output(
    model_name: str,
    problem: str,
    candidate: str,
    error_messages: str,
    min_score: float = 0.0,
    max_score: float = 100.0
) -> float:
    """
    Asynchronously ask the given model to rate the candidate answer
    on a scale from min_score to max_score. Returns the numeric score.
    """
    prompt = f""" You are an expert in optimization and formal mathematics using the Lean theorem prover. You are given:

- An optimization problem and a corresponding algorithm designed to solve it.
- A candidate Lean formalization of the problem and algorithm.

Please carefully evaluate the candidate formalization and assign a **single numeric score** on a scale from **{min_score:.0f} to {max_score:.0f}**, based on the following criteria:

### Scoring Criteria:
1. Files with more errors will receive lower scores. The error message from Lean compilation is given to you below in error messages part. 
2. Score decomposition:
  1) Problem Formalization: Does the Lean code fully and accurately formalize the given optimization problem? (20')
  2) Algorithm Correctness: Is the algorithm correctly and rigorously formalized in Lean? (20')
  3) Update Scheme Explicitness: Does the formalization explicitly capture the update scheme (iteration rule) used by the algorithm? (20')
  4) Theoretical Analysis: Does the formalization include proofs or reasoning about properties of the problem (e.g., convexity, Lipschitz continuity) and the algorithm (e.g., convergence, complexity)? (20')
  5) Proof: Is the formalization of the proof complete and nonsorry? (20')
3. The score should reflect both syntatic correctness and semantic correctness of the formalization. 
4. The use of sorry to omit essential proofs or definitions will negatively impact the score. Additionally, any lack of clarity or ambiguity in the formalization should result in a score reduction.
5. If you want to score 100, please make sure that the formalization is complete, clear, and rigorous, with no missing definitions or proofs. Scoring 100 needs to be justified by the completeness and correctness of the formalization.

Please provide **only the numeric score** in your response. Be **objective, strict, and rigorous** in your evaluation.

Problem and algorithm:
{problem}

Candidate answer:
{candidate}

Error messages:
{error_messages}
"""

    num = call_deepseek_api(prompt)
    try:
        return parse_score(num)
    except ValueError:
        raise RuntimeError(f"Could not parse score from {model_name}: {num!r}")

def ensemble_score_concurrent(
    model_name: str,
    num: int,
    problem: str,
    candidate: str,
    error_messages: str,
    normalize: bool = True
) -> float:
    """
    Given a model, concurrently request 'num' scores from it and combine them.
    If normalize=True, returns the average score.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(score_output, model_name, problem, candidate, error_messages)
            for _ in range(num)
        ]
        scores = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                score = future.result()
                print(f"Model {i+1} score: {score:.2f}")
                scores.append(score)
            except Exception as e:
                print(f"Error in scoring model {i+1}: {e}")

    if not scores:
        return 0.0

    return sum(scores) / len(scores) if normalize else sum(scores)

def batch_score_evaluation_parallel(
    model_name: str,
    num: int,
    candidate_dir: str,
    problem_jsonl: str,
    output_txt: str,
    normalize: bool = True,
    max_workers: int = 16  
):
    candidate_dir = Path(candidate_dir)
    result_lines = []
    class_scores = defaultdict(list)
    file_scores = []
    lock = threading.Lock()

    all_lean_files = list(candidate_dir.rglob("*.lean"))

    def process_file(candidate_file):
        try:
            pname, pclass, problem_description = make_problem_description(str(candidate_file), problem_jsonl)
            candidate_content = read_file_content(candidate_file)
            _, info, _ = compile_file(str(candidate_file))
            _, errors = parse_lean_output_with_context(info)
            error_count = len(errors)

            if error_count == 0:
                error_messages = "No errors found in Lean file."
            else:
                header = f"{error_count} error(s) found in Lean file:"
                details = "\n".join(
                    f"[Line {e['line']}, Col {e['column']}] {e['message']} -- {e['line_content'].strip()}"
                    for e in errors
                )
                error_messages = f"{header}\n{details}"

            score = ensemble_score_concurrent(
                model_name, num, problem_description, candidate_content, normalize=normalize, error_messages=error_messages
            )

            result_line = f"{pclass:<20} | {pname:<40} | {str(candidate_file):<80} | Score: {score:.4f}"

            with lock:
                result_lines.append(result_line)
                class_scores[pclass].append(score)
                file_scores.append(score)
        except Exception as e:
            with lock:
                result_lines.append(f"ERROR processing {str(candidate_file)}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in all_lean_files}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Scoring Lean files", unit="file"):
            pass  
    result_lines.append("\n=== Average Scores by Class ===")
    for cls, scores in class_scores.items():
        avg_score = sum(scores) / len(scores)
        result_lines.append(f"{cls:<20} | Avg Score: {avg_score:.4f}")

    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w") as f:
        f.write("\n".join(result_lines))

    print(f"✅ Results written to: {output_txt}")

def summarize_scores_from_txt(input_path: str):
    valid_lines = []
    pattern = re.compile(r'^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*Score:\s*([0-9.]+)\s*$')

    with open(input_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                valid_lines.append([
                    match.group(1),
                    match.group(2),
                    match.group(3),
                    float(match.group(4))
                ])

    if not valid_lines:
        print("No valid data lines found.")
        return
    df = pd.DataFrame(valid_lines, columns=["Algorithm", "Problem", "FilePath", "Score"])

    max_scores = df.groupby(["Algorithm", "Problem"])["Score"].max().reset_index(name="MaxScore")

    overall_avg = max_scores["MaxScore"].mean()

    category_avg = max_scores.groupby("Algorithm")["MaxScore"].mean().reset_index(name="CategoryAvgScore")

    with open(input_path, "a") as f:
        f.write("\n\n===== Autoformalization Summary =====\n")
        f.write("Max score for each (Algorithm, Problem):\n")
        f.write(max_scores.to_string(index=False))
        f.write("\n\nAverage of max scores overall: {:.4f}\n".format(overall_avg))
        f.write("\nCategory-wise average of max scores:\n")
        f.write(category_avg.to_string(index=False))
        f.write("\n===== End of Summary =====\n")

    print(f"Analysis complete. Summary appended to: {input_path}")

