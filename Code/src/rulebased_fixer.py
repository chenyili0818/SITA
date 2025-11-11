import re
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def replace_section_with_noncomputable(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith('section') and not stripped.startswith('noncomputable'):
        indent = line[:len(line) - len(stripped)]
        return indent + 'noncomputable section' + line[len(indent) + len('section'):]
    return line

def process_variable_line(line: str) -> str:
    """
    If the line starts with 'variable' (possibly indented), replace each outermost
    parenthesis group '( ... )' with '{ ... }', but keep already braced groups '{ ... }'
    and do not modify inner parentheses inside the group.
    """
    match = re.match(r'^(\s*variable)(.*)$', line)
    if not match:
        return line

    prefix, rest = match.group(1), match.group(2)
    result = []
    i = 0
    while i < len(rest):
        ch = rest[i]
        if ch == '(':
            # Try to find matching ')'
            depth = 1
            j = i + 1
            while j < len(rest) and depth > 0:
                if rest[j] == '(':
                    depth += 1
                elif rest[j] == ')':
                    depth -= 1
                j += 1
            if depth == 0:
                content = rest[i+1:j-1]
                result.append('{'+content+'}')
                i = j
            else:
                result.append(ch)
                i += 1
        elif ch == '{':
            # Already a braced group, copy as is until matching '}'
            depth = 1
            j = i + 1
            while j < len(rest) and depth > 0:
                if rest[j] == '{':
                    depth += 1
                elif rest[j] == '}':
                    depth -= 1
                j += 1
            result.append(rest[i:j])
            i = j
        else:
            result.append(ch)
            i += 1

    return prefix + ''.join(result)

def correct_notation_line(line: str) -> str:
    stripped = line.strip()
    target_2 = 'local notation "‖" x "‖₂" => @Norm.norm _ (PiLp.instNorm 2 fun _ ↦ ℝ) x'
    target_1 = 'local notation "‖" x "‖₁" => @Norm.norm _ (PiLp.instNorm 1 fun _ ↦ ℝ) x'
    target_op = 'local notation "|‖" A "|‖" => ‖(Matrix.toEuclideanLin ≪≫ₗ LinearMap.toContinuousLinearMap) A‖₊'

    if stripped.startswith('local notation "‖" x "‖₂"'):
        return target_2 + '\n' if stripped != target_2 else line
    elif stripped.startswith('local notation "‖" x "‖₁"'):
        return target_1 + '\n' if stripped != target_1 else line
    elif stripped.startswith('local notation "|‖" A "|‖"'):
        return target_op + '\n' if stripped != target_op else line
    return line

def clean_empty_variable_blocks(line: str) -> str:
    """
    Remove any { ... } or ( ... ) group in variable lines that does not contain a colon ':'.
    Groups with a colon are considered typed and are kept. Nested parentheses inside
    typed groups (e.g. {lam : EuclideanSpace ℝ (Fin d)}) are handled correctly.
    """
    if not line.lstrip().startswith("variable"):
        return line

    result = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] in ['{', '(']:
            open_sym = line[i]
            close_sym = '}' if open_sym == '{' else ')'
            depth = 1
            i += 1
            group_content = []
            while i < n and depth > 0:
                if line[i] == open_sym:
                    depth += 1
                elif line[i] == close_sym:
                    depth -= 1
                group_content.append(line[i])
                i += 1
            inner = ''.join(group_content[:-1]) if depth == 0 else ''.join(group_content)
            full_group = open_sym + inner + close_sym
            if ':' in inner:
                result.append(full_group)
        else:
            result.append(line[i])
            i += 1
    return ''.join(result)

def process_lean_file(file_path: str, inplace: bool = True):
    """
    Process a Lean file:
    - Remove old import statements
    - Insert standard import lines based on file name
    - Insert unified open declaration after import
    - Replace 'section' with 'noncomputable section'
    - Replace ( ... ) with { ... } in 'variable' lines
    - Remove bracket groups without type annotations
    - Correct L1 and L2 norm local notation
    - Replace λ with lam, ⊙ with •
    - Remove meaningless lines like 'variable', 'open'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    file_basename = str(file_path)
    # print(file_basename)

    # Step 1: Remove all import and open lines
    content_lines = []
    for line in original_lines:
        stripped = line.lstrip()
        if stripped.startswith('import') or stripped.startswith('open'):
            continue
        content_lines.append(line)

    # Step 2: Add imports
    imports_to_add = ['import Optlib.Autoformalization.Lemmas']
    if 'proximal_gradient' in file_basename:
        imports_to_add.append('import Optlib.Autoformalization.Template.PGD_template')
    if 'BCD' in file_basename:
        imports_to_add.append('import Optlib.Autoformalization.Template.BCD_template')
    if 'gradient_descent' in file_basename:
        imports_to_add.append('import Optlib.Autoformalization.Template.GD_template')
    if 'Nesterov' in file_basename:
        imports_to_add.append('import Optlib.Autoformalization.Template.NesterovFirst_template')
    if 'ADMM' in file_basename:
        imports_to_add.append('import Optlib.Autoformalization.Template.ADMM_template')

    processed_lines = [imp + '\n' for imp in imports_to_add]
    processed_lines.append('\n')  # add empty line
    processed_lines.append('open Set Real Matrix Finset Filter Bornology BigOperators Topology Classical\n')

    # Step 3: Process remaining lines
    new_lines = []
    keywords_to_strip = {'variable', 'open'}
    for line in content_lines:
        # Replace unicode symbols
        line = line.replace('λ', 'lam')
        line = line.replace('⊙', '•')

        # Skip trivial lines
        if line.strip() in keywords_to_strip:
            continue

        line = replace_section_with_noncomputable(line)
        line = process_variable_line(line)
        line = clean_empty_variable_blocks(line)
        line = correct_notation_line(line)

        new_lines.append(line.rstrip() + '\n')

    processed_lines.extend(new_lines)

    # Output path
    output_path = file_path if inplace else (
        file_path[:-5] + '_modified.lean' if file_path.endswith('.lean') else file_path + '_modified'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

    # print(f"File written to: {output_path}")
