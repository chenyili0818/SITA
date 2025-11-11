import os
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def compile_file(filepath, cwd="lean"):
    """
    Compile single Lean file
    Input:
    - filepath (str): path to Lean file
    - cwd (str): working directory, where the toolchian is at
    Return:
    - file name
    - infoview output
    - compile result (success = 0, fail = 1) 
    """
    cwd = Path(cwd).resolve()
    filepath = Path(filepath).resolve()
    r = subprocess.run(["lake", "env", "lean", filepath], cwd=cwd,
                       capture_output=True, text=True, encoding='utf-8')
    stdout = r.stdout
    return os.path.basename(filepath), stdout, r.returncode

def parse_lean_output_with_context_aux(lean_output):
    """
    Parse Lean output and attach file‐context to each warning/error exactly once.
    """
    warning_re = re.compile(r"(.+):(\d+):(\d+): warning: (.+)")
    error_re   = re.compile(r"(.+):(\d+):(\d+): error: (.+)")

    warnings, errors = [], []
    current = None
    is_warning = False

    def flush():
        nonlocal current
        if not current:
            return
        (warnings if is_warning else errors).append(current)
        current = None

    # 1) Scan lines, accumulate multi‐line messages
    for line in lean_output.splitlines():
        w = warning_re.match(line)
        e = error_re.match(line)
        if w or e:
            # flush the previous one
            flush()

            fp, ln, col, msg = (w or e).groups()
            current = {
                "filepath": fp.strip(),
                "line":     int(ln),
                "column":   int(col),
                "message":  msg.strip(),
                # placeholders for enrichment
                "line_content": "",
                "char_at_column": "",
            }
            is_warning = bool(w)
        elif current:
            # continuation line
            current["message"] += "\n" + line.strip()

    # 2) at EOF, flush the very last record
    flush()

    # 3) read file lines once
    file_paths = {r["filepath"] for r in warnings + errors}
    contents = {}
    for f in file_paths:
        try:
            contents[f] = open(f, encoding="utf-8").read().splitlines()
        except FileNotFoundError:
            contents[f] = None

    # 4) enrich each record
    for rec in warnings + errors:
        lines = contents.get(rec["filepath"])
        if lines is None:
            rec["line_content"] = f"[couldn't read file {rec['filepath']}]"
            rec["char_at_column"] = ""
            continue
        idx = rec["line"] - 1
        if not (0 <= idx < len(lines)):
            rec["line_content"] = f"[couldn't read line {rec['line']}]"
            rec["char_at_column"] = ""
            continue
        ln_txt = lines[idx]
        rec["line_content"] = ln_txt
        if 0 <= rec["column"] < len(ln_txt):
            rec["char_at_column"] = ln_txt[rec["column"]]
        else:
            rec["char_at_column"] = ""

    return warnings, errors

def process_item(item):
  new_item = []
  for key in item:
      current_record = {
                "line": key["line"],
                "column": key["column"],
                "message": key["message"],
                "line_content": key["line_content"],
                "char_at_column": key["char_at_column"],
            }
      new_item.append(current_record)
  return new_item

def parse_lean_output_with_context(info):
    warnings, errors = parse_lean_output_with_context_aux(info)
    return process_item(warnings), process_item(errors)

