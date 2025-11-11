import sys
import re
from interact import compile_file, parse_lean_output_with_context

"""
Script to process a Lean source file:
- For each simple error (unexpected token 'sorry'), comment out the line.
- For each fields missing: 'xxx' error, keep the line and insert xxx := sorry on the next line (one tab indent).
- For 'xxx is not a field of structure yyy', comment out the line and its indented block (unless next line is same indent); do NOT insert sorry.
- For let-related errors with :=, replace RHS with sorry and comment the let-block.
- For other errors, comment out the line or RHS and insert sorry, then comment out the subsequent block with the same indentation.
"""

SIMPLE_MSG = "unexpected token 'sorry'; expected command"
MISSING_FIELD_RE = re.compile(r"^fields missing: ((?:'[^']+',?\s*)+)$")
FIELD_NOT_IN_STRUCT_RE = re.compile(r"^'[^']+' is not a field of structure '[^']+'$")
NO_GOALS_RE = re.compile(r"no goals") 
UNSOLVED_GOALS_RE = re.compile(r"unsolved goals") 

def process_file(path: str, error_info: list[dict]) -> None:
    if not error_info:
        return
    
    first_error = error_info[0]
    first_line = first_error['line']
    msg = first_error['message']
    col = first_error.get('column', None)

    is_simple = msg == SIMPLE_MSG
    is_missing_field_match = MISSING_FIELD_RE.match(msg)
    is_field_not_in_struct = FIELD_NOT_IN_STRUCT_RE.match(msg)
    is_no_goals = NO_GOALS_RE.search(msg) is not None
    is_unsolved_goals = UNSOLVED_GOALS_RE.search(msg) is not None
    is_block_error = not is_simple and not is_missing_field_match and not is_field_not_in_struct and not is_no_goals and not is_unsolved_goals

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    n = len(lines)

    regions_to_comment = set()
    sorry_insertions = []

    if is_unsolved_goals:
        print(f"[FIX_SORRY] Processing unsolved goals error at line {first_line}")
        
        error_line = lines[first_line - 1] 
        error_line_indent = error_line[:len(error_line) - len(error_line.lstrip(' \t'))]
        
        if 'by' in error_line:
            print(f"[FIX_SORRY] Error line contains 'by', using empty line strategy")
            
            empty_line_found = False
            target_line_idx = None
            last_non_empty_indent = ""
            
            for i in range(first_line, len(lines)):
                line = lines[i]
                if line.strip() == "":  
                    target_line_idx = i
                    empty_line_found = True
                    break
                else:
                    last_non_empty_indent = line[:len(line) - len(line.lstrip(' \t'))]
            
            if empty_line_found and target_line_idx is not None:
                sorry_insertions.append((target_line_idx, last_non_empty_indent))
                print(f"[FIX_SORRY] Will insert sorry at empty line {target_line_idx + 1} with indent '{last_non_empty_indent}'")
            else:
                print(f"[FIX_SORRY] No empty line found after unsolved goals error, skipping sorry insertion")
        else:
            print(f"[FIX_SORRY] Error line does not contain 'by', using same-indent or empty line strategy")
            
            current_indent_len = len(error_line_indent)
            target_line_idx = None
            target_type = None  
            
            for i in range(first_line, len(lines)):
                line = lines[i]
                if line.strip() == "": 
                    target_line_idx = i
                    target_type = "empty_line"
                    break
                else:
                    line_indent = line[:len(line) - len(line.lstrip(' \t'))]
                    if len(line_indent) == current_indent_len:
                        target_line_idx = i  
                        target_type = "same_indent"
                        break
            
            if target_line_idx is not None:
                if target_type == "empty_line":
                    additional_indent = error_line_indent + "  "  
                    sorry_insertions.append((target_line_idx, additional_indent))
                    print(f"[FIX_SORRY] Will insert sorry at empty line {target_line_idx + 1} with additional indent '{additional_indent}'")
                elif target_type == "same_indent":
                    additional_indent = error_line_indent + "  "  
                    sorry_insertions.append((target_line_idx, additional_indent))
                    print(f"[FIX_SORRY] Will insert sorry before same-indent line {target_line_idx + 1} with additional indent '{additional_indent}'")
            else:
                print(f"[FIX_SORRY] No same-indent line or empty line found, skipping sorry insertion")

    elif is_block_error or is_field_not_in_struct or is_no_goals:
        raw = lines[first_line - 1]
        indent = raw[:len(raw) - len(raw.lstrip(' \t'))]
        indent_len = len(indent)
        regions_to_comment.add(first_line)
        i = first_line + 1

        if is_no_goals:
            print(f"[FIX_SORRY] Processing no goals error at line {first_line}, indent_len={indent_len}")
            while i <= n:
                raw_i = lines[i - 1]
                if not raw_i.strip():  
                    regions_to_comment.add(i)
                    print(f"[FIX_SORRY] Adding empty line {i} to comment regions")
                    i += 1
                    continue
                indent_i = raw_i[:len(raw_i) - len(raw_i.lstrip(' \t'))]
                if len(indent_i) >= indent_len:
                    regions_to_comment.add(i)
                    print(f"[FIX_SORRY] Adding line {i} to comment regions (indent={len(indent_i)}): {raw_i.strip()}")
                    i += 1
                elif len(indent_i) < indent_len:
                    print(f"[FIX_SORRY] Stopping at line {i} (indent={len(indent_i)} < {indent_len}): {raw_i.strip()}")
                    break
                else:
                    i += 1

        elif is_block_error and 'let' in raw:
            while i <= n:
                raw_i = lines[i - 1]
                if not raw_i.strip():
                    regions_to_comment.add(i)
                    i += 1
                    continue
                indent_i = raw_i[:len(raw_i) - len(raw_i.lstrip(' \t'))]
                if len(indent_i) > indent_len:
                    regions_to_comment.add(i)
                    i += 1
                else:
                    break

        elif is_field_not_in_struct:
            if i <= n:
                next_line = lines[i - 1]
                if next_line.strip():  
                    indent_next = next_line[:len(next_line) - len(next_line.lstrip(' \t'))]
                    if len(indent_next) == indent_len:
                        pass
                    elif len(indent_next) > indent_len:
                        while i <= n:
                            raw_i = lines[i - 1]
                            if not raw_i.strip():
                                regions_to_comment.add(i)
                                i += 1
                                continue
                            indent_i = raw_i[:len(raw_i) - len(raw_i.lstrip(' \t'))]
                            if len(indent_i) > indent_len:
                                regions_to_comment.add(i)
                                i += 1
                            else:
                                break
        else:
            while i <= n:
                raw_i = lines[i - 1]
                stripped_i = raw_i.lstrip()
                if not stripped_i.strip():
                    regions_to_comment.add(i)
                    i += 1
                    continue
                indent_i = raw_i[:len(raw_i) - len(stripped_i)]
                if len(indent_i) > indent_len or indent_i.startswith(indent):
                    regions_to_comment.add(i)
                    i += 1
                else:
                    break

    output_lines = []
    for idx, line in enumerate(lines, start=1):
        indent = line[:len(line) - len(line.lstrip(' \t'))]
        stripped = line.lstrip()

        for insert_idx, insert_indent in sorry_insertions:
            if idx - 1 == insert_idx: 
                output_lines.append(f"{insert_indent}sorry\n")
                print(f"[FIX_SORRY] Inserted sorry at line {idx} with indent '{insert_indent}'")

        if idx == first_line:
            if is_simple:
                output_lines.append(f"{indent}-- {stripped}" if stripped.strip() else line)
            elif is_missing_field_match:
                fields_str = is_missing_field_match.group(1)
                fields = [f.strip("' ") for f in fields_str.split(",") if f.strip()]
                output_lines.append(line)
                for f in fields:
                    output_lines.append(f"{indent}\t{f} := sorry\n")
            elif is_no_goals:
                output_lines.append(f"{indent}-- {stripped}" if stripped.strip() else line)
            elif is_unsolved_goals:
                output_lines.append(line)
            elif is_block_error:
                pos = line.find(':=') if line else -1
                if pos != -1 and col is not None and col > pos + 2:
                    before = line[:pos + 2]
                    after = line[pos + 2:]
                    output_lines.append(f"{before} sorry --{after}")
                else:
                    output_lines.append(f"{indent}-- {stripped}")
                    output_lines.append(f"{indent}sorry\n")
            elif is_field_not_in_struct:
                output_lines.append(f"{indent}-- {stripped}")
        elif idx in regions_to_comment:
            output_lines.append(f"{indent}-- {stripped}" if stripped.strip() else line)
        else:
            output_lines.append(line)

    filtered_lines = [l for l in output_lines if not re.match(r"^\s*--\s*sorry\s*$", l)]

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

def auto_process_file(path: str) -> bool:
    """
    Process one error per round. Stop if compiled or exceeds max rounds.
    """
    with open(path, 'r', encoding='utf-8') as f:
        original = f.read()

    round = 1
    max_rounds = 10
    while round <= max_rounds:
        print(f"\n[Round {round}] Compiling...")
        _, info, tag = compile_file(path)
        _, errors = parse_lean_output_with_context(info)

        if tag == 0 or not errors:
            print("✅ All errors have been processed and replaced with 'sorry'.")
            return True

        print(f"🚧 Found {len(errors)} errors. Processing the first one at line {errors[0]['line']}...")
        process_file(path, errors)
        round += 1

    with open(path, 'w', encoding='utf-8') as f:
        f.write(original)
    print("❌ Max rounds reached. Errors remain. Original content restored.")
    return False
