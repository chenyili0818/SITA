from __future__ import annotations

import datetime
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from openai import OpenAI
from sentence_transformers import SentenceTransformer  
from sklearn.metrics.pairwise import cosine_similarity 

from interact import compile_file, parse_lean_output_with_context
from SITA.Code.src.rulebased_fixer import process_lean_file


# ---------------------------------------------------------------------------
# Configuration and Global Setup
# ---------------------------------------------------------------------------

# Application Configuration
CONFIG = {
    "max_attempts": 3,  
    "final_max_attempts": 2, 
    "model_name": "deepseek-reasoner",  
    "temperature": 0.7,  
    "error_db": "./data/bugfix/error_knowledge.json",   
    "theorem_db_dir": "./data/theorem_database/Optlib"  
}

with open('./data/api/config.json', 'r', encoding='utf-8') as f:
    external_config = json.load(f)

CONFIG.update({
    "api_key": external_config.get("api_key"),
    "base_url": external_config.get("base_url")
})

# Initialize OpenAI client
client: OpenAI = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

# Initialize BERT model globally (load once)
CODE_MODEL = SentenceTransformer('../models/all-mpnet-base-v2-local')


def setup_logging(log_level=logging.INFO, log_file_path=None) -> logging.Logger:
    """Setup comprehensive logging with both file and console output."""
    logger = logging.getLogger("LeanErrorCorrector")
    logger.setLevel(log_level)
    
    # If logger already has handlers (from shared_logger), return it
    if logger.handlers:
        return logger
  
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12s] [%(levelname)-8s] %(name)s: %(message)s"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s"
    )
    
    # File handler (detailed)
    if log_file_path:
        # Use provided log file path
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    else:
        # Fallback to default location if no path provided
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        default_log_path = f"./results/logs/fixlog_{timestamp_str}.log"
        os.makedirs("./results/logs", exist_ok=True)
        file_handler = logging.FileHandler(default_log_path, encoding="utf-8")

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simplified)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info("🚀 Lean Error Correction Pipeline Started")
    logger.info("=" * 80)
    logger.info(f"📝 Log file: {file_handler.baseFilename}")
    logger.info(f"⚙️  Configuration: {CONFIG}")
    
    return logger


# ---------------------------------------------------------------------------
# Main Error Correction Class
# ---------------------------------------------------------------------------

class LeanErrorCorrector:
    """
    Advanced Lean error correction system with intelligent error classification,
    knowledge base management, and comprehensive logging.
    """
    
    def __init__(self, lean_file: str, logger : logging.Logger = None):
        """Initialize the error corrector with a Lean file."""
        self.lean_file = Path(lean_file)
        # Use provided logger or create new one
        self.logger = logger if logger else setup_logging()
        # Initialize file operations
        self.logger.info(f"🎯 Initializing error corrector for: {self.lean_file}")
        start_time = time.time()
        
        self.original_code = self._read_file()
        self.current_code = self.original_code
        self.error_knowledge = self._load_error_knowledge()
        # Precompute knowledge base embeddings
        self._preprocess_knowledge()
        self.theorem_db = self._load_theorem_database()
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ Initialization completed in {init_time:.2f}s")
        self.logger.info(f"📏 Original code: {len(self.original_code)} characters")
        self.logger.info(f"📚 Loaded {len(self.theorem_db)} theorems from database")
        self.logger.info(f"🧠 Knowledge base contains {len(self.error_knowledge)} error types")
    
    def _load_theorem_database(self) -> Dict[str, Dict]:
        """
        Load and merge all theorem databases from the configured directory.
        
        Returns:
            Dictionary mapping theorem names to their full information,
            combining all theorems from all JSON files in the theorem database directory.
        """
        self.logger.debug("📚 Loading theorem database...")
        db_dir = Path(CONFIG["theorem_db_dir"])
        
        if not db_dir.exists():
            self.logger.warning(f"⚠️ Theorem database directory not found: {db_dir}")
            return {}
        
        theorem_map = {}
        loaded_files = 0
        
        # Process all JSON files in the theorem database directory
        for db_file in db_dir.glob("*.json"):
            try:
                with open(db_file, 'r', encoding='utf-8') as f:
                    theorems = json.load(f)
                    
                    # Add theorems to the global map
                    for thm in theorems:
                        name = thm.get("theorem_name")
                        if name:
                            if name in theorem_map:
                                self.logger.debug(f"⚠️ Duplicate theorem name: {name}")
                                
                            theorem_map[name] = thm
                    
                    loaded_files += 1
                    self.logger.debug(f"Loaded {len(theorems)} theorems from {db_file.name}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load {db_file.name}: {e}")
                continue
        
        if theorem_map:
            self.logger.info(f"✅ Loaded {len(theorem_map)} theorems from {loaded_files} files")
        else:
            self.logger.warning("⚠️ No theorems loaded from database")
        
        return theorem_map
    
    def _read_file(self) -> str:
        """Read Lean file content with error handling and logging."""
        try:
            self.logger.debug(f"📖 Reading file: {self.lean_file}")
            with open(self.lean_file, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.debug(f"✅ Successfully read {len(content)} characters")
            return content
        except FileNotFoundError:
            self.logger.error(f"❌ File not found: {self.lean_file}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error reading file {self.lean_file}: {e}")
            raise
    
    def _write_file(self, file_path: Path | str, content: str):
        """Write content to file with logging."""
        file_path = Path(file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.current_code = content
            self.logger.debug(f"💾 Written {len(content)} characters to {file_path}")
        except Exception as e:
            self.logger.error(f"❌ Error writing to file {file_path}: {e}")
            raise
    
    # ---------------------------------------------------------------------------
    # Error Knowledge Management
    # ---------------------------------------------------------------------------
    
    def _load_error_knowledge(self) -> List[Dict]:
        """Load error knowledge database with new structure."""
        knowledge_file = Path(CONFIG["error_db"])
        
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    knowledge = json.load(f)
                self.logger.debug(f"Loaded error knowledge from {knowledge_file}")
                return knowledge
            except Exception as e:
                self.logger.error(f"Failed to load error knowledge: {e}")
                return []
        else:
            self.logger.info("Creating new error knowledge database")
            return []

    def _save_error_knowledge(self, new_knowledge: List[Dict] = None):
        """
        Update error knowledge base by merging new knowledge with existing one.
        
        Args:
            new_knowledge: New error knowledge entries to be merged.
                        If None, uses the current self.error_knowledge.
        """
        if new_knowledge is None:
            # new_knowledge = self.error_knowledge
            return
            
        knowledge_file = Path(CONFIG["error_db"])
        knowledge_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            existing_knowledge = []
            # Load existing knowledge if file exists
            if knowledge_file.exists():
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    existing_knowledge = json.load(f)
                self.logger.debug(f"Loaded {len(existing_knowledge)} existing knowledge entries")
            
            # Merge new knowledge with existing
            merged_knowledge = self._merge_knowledge(existing_knowledge, new_knowledge)
            
            # Save merged knowledge
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(merged_knowledge, f, indent=2, ensure_ascii=False)
            self.error_knowledge = merged_knowledge
                
            self.logger.info(f"✅ Updated error knowledge base. Total entries: {len(merged_knowledge)}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update error knowledge: {e}")
            raise
    
    def _merge_knowledge(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """
        Merge new error knowledge entries with existing ones, avoiding duplicates.
        
        Args:
            existing: List of existing error knowledge entries
            new: List of new error knowledge entries
            
        Returns:
            Merged list of knowledge entries with duplicates removed
        """
        # Create a dictionary for fast lookup by error signature
        knowledge_map = {}
        
        # Index existing knowledge
        for entry in existing:
            key = self._get_error_signature(entry)
            knowledge_map[key] = entry
        
        # Add or update with new knowledge
        for entry in new:
            key = self._get_error_signature(entry)
            if key in knowledge_map:
                # Merge successful fixes if same error exists
                existing_fixes = knowledge_map[key].get("successful_fixes", [])
                new_fixes = entry.get("successful_fixes", [])
                knowledge_map[key]["successful_fixes"] = list(set(existing_fixes + new_fixes))
            else:
                knowledge_map[key] = entry
        
        return list(knowledge_map.values())
    
    def _preprocess_knowledge(self):
        """Preprocess knowledge base with enhanced normalization."""
        self.processed_kb = []
        error_signatures = []
        
        for entry in self.error_knowledge:
            pattern = self._extract_error_pattern(entry["error_message"])
            error_signature = f"[{entry['error_type']}] {pattern}"
            
            processed = {
                "original": entry,
                "error_type": entry["error_type"],
                "pattern": pattern,
                "signature": error_signature,
                "context_keywords": self._extract_keywords(entry.get("context", "")),
                "fix_suggestion": entry.get("fix_suggestion", ""),
                "successful_fixes": entry.get("successful_fixes", [])
            }
            self.processed_kb.append(processed)
            error_signatures.append(error_signature)
        
        # Compute embeddings for complete signatures
        if error_signatures:
            self.kb_embeddings = CODE_MODEL.encode(error_signatures)
        else:
            self.kb_embeddings = np.array([])

        # Compute embeddings for context keywords
        self.kb_kw_embeddings = CODE_MODEL.encode(
            [', '.join(entry["context_keywords"]) 
            for entry in self.processed_kb]
        ) if self.processed_kb else None

    def _get_error_signature(self, error_entry: Dict) -> str:
        """
        Create a unique signature for an error entry to detect duplicates.
        
        Args:
            error_entry: An error knowledge entry
            
        Returns:
            String signature combining error type and message
        """
        return f"{error_entry.get('error_type','')}|{error_entry.get('error_message','')}"
    
    # ---------------------------------------------------------------------------
    # Context and Function Analysis
    # ---------------------------------------------------------------------------
    
    def _get_context_lines(self, line_num: int, context_size: int = 3) -> str:
        """Get code context around error line with enhanced formatting."""
        lines = self.current_code.split('\n')
        start = max(0, line_num - 1 - context_size)  # Line numbers start from 1, indices from 0
        end = min(len(lines), line_num + context_size)
        
        context_lines = []
        for i in range(start, end):
            line_marker = ">>> " if i == line_num - 1 else "    "
            context_lines.append(f"{i+1:4d}: {line_marker}{lines[i]}")
        
        context = "\n".join(context_lines)
        self.logger.debug(f"📍 Context for line {line_num}:\n{context}")
        return context
    
    def _extract_function_name(self, error_message: str) -> Optional[str]:
        """
        Extract function/theorem name from error message with comprehensive patterns.
        
        Args:
            error_message: Raw error message from Lean
            
        Returns:
            Extracted function/theorem name or None if not found
        """
        # Ordered list of patterns to try (most specific first)
        patterns = [
            # Type mismatch errors
            (r"type mismatch\s+([\w'.]+)\s", "type mismatch"),
            (r"application type mismatch\s+([\w'.]+)", "application type mismatch"),
            
            # Unknown identifier errors
            (r"unknown identifier '([\w'.]+)'", "unknown identifier"),
            (r"identifier '([\w'.]+)' not found", "identifier not found"),
            
            # Function application errors
            (r"function expected at\s+([\w'.]+)", "function expected"),
            (r"invalid apply tactic, failed to unify\s+([\w'.]+)", "apply tactic failed"),
            
            # Theorem/Lemma specific errors
            (r"theorem '([\w'.]+)' not found", "theorem not found"),
            (r"lemma '([\w'.]+)' not found", "lemma not found"),
            (r"failed to synthesize instance for '([\w'.]+)'", "instance synthesis failed"),
            
            # Type class resolution
            (r"failed to infer instance of '([\w'.]+)'", "instance inference failed"),
            
            # Rewrite errors
            (r"rewrite tactic failed, did not find instance of the pattern.*theorem '([\w'.]+)'", "rewrite failed"),
            (r"rewrite tactic failed, did not find instance of the pattern.*lemma '([\w'.]+)'", "rewrite failed"),
            
            # Simplifier errors
            (r"simplify tactic failed to prove.*using '([\w'.]+)'", "simplify failed"),
            
            # General term errors
            (r"term '([\w'.]+)' has type", "term type"),
            (r"expression '([\w'.]+)' has type", "expression type"),
            
            # Equation compiler errors
            (r"equation compiler failed to generate code for '([\w'.]+)'", "equation compiler failed"),
        ]

        for pattern, pattern_name in patterns:
            match = re.search(pattern, error_message)
            if match:
                func_name = match.group(1)
                self.logger.debug(f"🔍 Extracted '{func_name}' from {pattern_name} pattern")
                return func_name
        
        # Special case: tactic failures with quoted names
        if "'" in error_message:
            quoted_names = re.findall(r"'([\w'.]+)'", error_message)
            if quoted_names:
                self.logger.debug(f"🔍 Considering quoted names: {quoted_names}")
                return quoted_names[-1]  # Often the most relevant one
        
        self.logger.debug("🔍 No function/theorem name found in error message")
        return None
    
    # ---------------------------------------------------------------------------
    # Enhanced Error Retrieval with BERT
    # ---------------------------------------------------------------------------
    def _normalize_code_context(self, code: str) -> str:
        """
        Normalize code context by replacing specific patterns with generic placeholders:
        1. Module paths (Mathlib.Algebra.Group) → MODULE.PATH
        2. Namespaced definitions (Namespace.func_name) → NAMESPACE.FUNC
        3. Structure/class names (StructureName) → STRUCTURE
        """
        # Handle multi-level module paths first
        code = re.sub(r'\b([A-Z][a-zA-Z0-9_]*(\.[a-zA-Z0-9_]+)+)\b', 'MODULE.PATH', code)
        
        # Handle definition statements (def/lemma etc.)
        code = re.sub(
            r'\b(def|lemma|theorem|structure|class|instance)\s+([A-Z][a-zA-Z0-9_]*\.)?[a-z][a-zA-Z0-9_]*\b',
            r'\1 DEFNAME',
            code
        )
        
        # Handle namespaced function calls
        code = re.sub(r'\b([A-Z][a-zA-Z0-9_]*)\.([a-z][a-zA-Z0-9_]*)\b', 'NAMESPACE.FUNC', code)
        
        # Handle remaining structure names (exclude already processed cases)
        code = re.sub(r'\b([A-Z][a-zA-Z0-9_]*)\b(?!\.)', 'STRUCTURE', code)
        
        return code

    def _extract_error_pattern(self, error_msg: str) -> str:
        """Extract key error pattern with structure preservation."""
        # Normalize names in error message
        normalized = self._normalize_code_context(error_msg)
        
        # Basic normalization
        cleaned = re.sub(r'\s+', ' ', normalized.strip())
        cleaned = cleaned.replace('→', '->').replace('⇒', '=>')
        
        # Preserve type variables but normalize numbering
        cleaned = re.sub(r'\?m\.\d+', '?VAR', cleaned)
        
        # Normalize numeric constants
        cleaned = re.sub(r'(?<!\w)[\-+]?\d+\.?\d*(?!\w)', 'NUM', cleaned)
        
        return ' '.join(cleaned.split())

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract context keywords with name normalization."""
        # Normalize code context first
        normalized = self._normalize_code_context(text)
        
        # Exclude placeholder terms
        excluded_terms = {
            'MODULE.PATH', 'NAMESPACE.FUNC', 'STRUCTURE', 'DEFNAME',
            'the', 'and', 'or', 'not', 'is', 'in', 'of', 'let', 'fun'
        }
        
        # Extract identifiers (including lowercase)
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', normalized)
        
        # Extract math types and structures
        types = re.findall(
            r'\b(?:EuclideanSpace|Fin|Matrix|Vec|HSub|HMul|HAdd|HDiv|HPow|NNNorm|Norm|Seminormed|LinearMap|Continuous|Differentiable)\b', 
            normalized
        )
        
        # Extract math operators
        operators = re.findall(
            r'[+\-*/^=<>≤≥‖⊢∀∃⇒→⮕↦]|::|->|=>|\^|\*ᵥ|⬝|ᵀ|ᵥ', 
            normalized
        )
        
        # Extract Lean keywords and tactics
        lean_keys = re.findall(
            r'\b(?:theorem|lemma|def|structure|class|instance|variable|example|have|show|assume|suffices|by|done|calc|rw|apply|exact|unfold|simp|intro)\b', 
            normalized
        )
        
        # Combine and filter keywords
        keywords = list(set(identifiers + types + operators + lean_keys))
        return [kw for kw in keywords 
                if len(kw) > 1 and kw.lower() not in excluded_terms]

    def _retrieve_similar_error(self, error: Dict) -> List[Dict]:
        """Retrieve top 3 similar errors with enhanced matching."""
        error_message = error.get("message", "")
        error_type = self._classify_error_type(error_message)
        context = error.get("full_context", "")
        
        if not self.processed_kb:
            return []
        
        # Build current error's signature
        current_pattern = self._extract_error_pattern(error_message)
        current_signature = f"[{error_type}] {current_pattern}" if error_type else current_pattern
        current_embedding = CODE_MODEL.encode([current_signature])
        
        # Compute semantic similarity
        semantic_sims = cosine_similarity(current_embedding, self.kb_embeddings)[0]
        
        # Compute context keyword similarity (Jaccard index)
        current_keywords = set(self._extract_keywords(context))
        keyword_sims = []
        
        # Turn the list of keywords into a sentence (comma-separated)
        current_kw_text = ', '.join(current_keywords)
        
        # Compute embeddings for keywords
        current_embedding = CODE_MODEL.encode([current_kw_text])
        kb_embeddings = self.kb_kw_embeddings
        
        # Compute cosine similarity for keyword embeddings
        keyword_sims = cosine_similarity(current_embedding, kb_embeddings)[0]
        
        # Type matching enhancement
        for i, entry in enumerate(self.processed_kb):
            if error_type and entry["error_type"] == error_type:
                keyword_sims[i] = min(keyword_sims[i] + 0.2, 1.0)
        
        # Combined score with dynamic weighting
        combined_scores = []
        for sem, key in zip(semantic_sims, keyword_sims):
            weight = 0.7 if sem > 0.7 else 0.5  # Favor semantics for high scores
            combined_scores.append(weight * sem + (1 - weight) * key)
        
        # Get top 3 matches
        top_indices = np.argsort(combined_scores)[-3:][::-1]
        results = []
        
        for idx in top_indices:
            match = self.processed_kb[idx]
            results.append({
                "match_rank": len(results) + 1,
                "score": float(combined_scores[idx]),
                # error info
                "original_error": {
                    "message": error_message,
                    "type": error_type,
                    "context": context
                },
                # match info
                "matched_error": {
                    "type": match["error_type"],
                    "message": match["original"]["error_message"],
                    "fix_suggestion": match["fix_suggestion"],
                    "successful_fixes": match["successful_fixes"],
                    "context_similarity": float(keyword_sims[idx]),
                    "pattern_similarity": float(semantic_sims[idx])
                }
            })
        
        return results

    # ---------------------------------------------------------------------------
    # Fuzzy Theorem Matching
    # ---------------------------------------------------------------------------

    def _normalize_name(self, name: str) -> str:
        """Normalize names for fuzzy matching."""
        if not name:
            return ""
        # Remove all non-alphanumeric chars and convert to lowercase
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def _fuzzy_match_theorem(self, theorem_name: str, top_n: int = 3) -> List[Dict]:
        """Find top_n most similar theorems using fuzzy matching."""
        normalized_input = self._normalize_name(theorem_name)
        if not normalized_input:
            return []

        results = []
        for thm_name, thm_info in self.theorem_db.items():
            normalized_thm = self._normalize_name(thm_name)
            # Simple ratio of common characters
            common = sum(1 for c in normalized_input if c in normalized_thm)
            similarity = common / max(len(normalized_input), len(normalized_thm))
            results.append((similarity, thm_name, thm_info))
        
        # Sort by similarity and return top_n
        results.sort(reverse=True, key=lambda x: x[0])
        return [{
            "name": name,
            "info": info,
            "similarity": sim
        } for sim, name, info in results[:top_n]]

    def _retrieve_theorem_info(self, theorem_name: str) -> Optional[List[Dict]]:
        """Retrieve theorem info with fuzzy matching."""
        self.logger.debug(f"Retrieving theorem info: {theorem_name}")
        matches = self._fuzzy_match_theorem(theorem_name)
        if matches:
            self.logger.debug(f"Found {len(matches)} similar theorems")
            return matches
        self.logger.debug(f"Theorem not found: {theorem_name}")
        return None
    
    # ---------------------------------------------------------------------------
    # Language Model Interaction
    # ---------------------------------------------------------------------------
    
    def _call_large_model(self, prompt: str, error_info: dict) -> str:
        """Call language model API with enhanced error handling and logging."""
        self.logger.debug(f"🤖 Calling {CONFIG['model_name']} API...")
        start_time = time.time()
        
        try:
            # Add explicit English requirement
            enhanced_prompt = (
                "IMPORTANT: Output MUST use ONLY English characters and Lean4 syntax. "
                "Do NOT use any Chinese characters.\n\n" + prompt
            )
            
            response = client.chat.completions.create(
                model=CONFIG["model_name"],
                messages=[{"role": "user", "content": enhanced_prompt}],
                stream=False,
                # max_tokens=16000,
                temperature=CONFIG["temperature"],
                top_p=0.9,
                frequency_penalty=0.2,
                response_format={"type": "text"},
            )
            
            content = response.choices[0].message.content.strip()
            api_time = time.time() - start_time
            
            self.logger.info(f"✅ API call successful ({api_time:.2f}s), "
                           f"received {len(content)} characters")
            if content:
                self.logger.info("🤖 Model Response:")
                self.logger.info("=" * 60)
                self.logger.info(f"{content}")
                self.logger.info("=" * 60)
            
            return content
            
        except Exception as e:
            api_time = time.time() - start_time
            self.logger.error(f"❌ API call failed after {api_time:.2f}s: {e}")
            return ""
    
    def _extract_full_code(self, response: str) -> str:
        """Extract complete code from language model response."""
        self.logger.debug(f"🔧 Extracting code from {len(response)} character response")
        
        # Clean code block markers
        original_length = len(response)
        pattern = r"```(?:lean4|lean)?\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            extracted = match.group(1).strip()
            self.logger.debug(f"🔧 Extracted {len(extracted)} characters "
                         f"(removed {original_length - len(extracted)} formatting chars)")
            return extracted
        else:
            self.logger.warning("⚠️ No code block found in response. Returning original stripped response.")
            return response.strip()
    
    # ---------------------------------------------------------------------------
    # Enhanced Error Detection with Context and Error Compilation
    # ---------------------------------------------------------------------------
    
    def _identify_code_block(self, line_num: int) -> Tuple[str, str]:
        """
        Identify the code block type and name containing the given line number.
        Returns (block_type, block_name)
        """
        lines = self.current_code.split('\n')
        block_types = ['import', 'section', 'open', 'local notation', 
                      'variable', 'def', 'class', 'instance', 'lemma', 'theorem']
        
        if line_num < 1 or line_num > len(lines):
            return "", ""
    
        # Scan backwards from the error line
        for i in range(line_num-1, -1, -1):
            line = lines[i].strip()
            
            # Check for block starters
            for block_type in block_types:
                if line.startswith(block_type + ' '):
                    # Extract block name (content after block type until next space or :)
                    remaining = line[len(block_type):].strip()
                    block_name = remaining.split()[0].split(':')[0].strip()
                    return block_type, block_name
                    
        return "", ""

    def _get_errors(self, file_path: Path | str = None) -> list:
        """Get errors with enhanced context information."""
        if file_path is None:
            file_path = self.lean_file
        
        self.logger.debug(f"Compiling file to check errors: {file_path}")
        start_time = time.time()
        
        try:
            _, info, _ = compile_file(str(file_path))
            _, errors = parse_lean_output_with_context(info)
            
            # Enhance errors with code block context
            for error in errors:
                line_num = error.get('line', 1)
                if line_num < 1:
                  error['code_block_type'] = "invalid"
                  error['code_block_name'] = "invalid"
                  error['full_context'] = "invalid"
                  continue
                block_type, block_name = self._identify_code_block(line_num)
                error['code_block_type'] = block_type
                error['code_block_name'] = block_name
                error['full_context'] = f"{block_type} {block_name}" if block_type else ""
            
            compile_time = time.time() - start_time
            error_count = len(errors) if errors else 0
            
            if error_count == 0:
                self.logger.debug(f"Compilation successful ({compile_time:.2f}s)")
            else:
                self.logger.debug(f"Found {error_count} errors ({compile_time:.2f}s)")
                
            return errors
            
        except Exception as e:
            compile_time = time.time() - start_time
            self.logger.error(f"Compilation failed ({compile_time:.2f}s): {e}")
            return []

    
    # ---------------------------------------------------------------------------
    # Prompt Building Functions
    # ---------------------------------------------------------------------------
    
    def _build_enhanced_prompt(self, errors: List[Dict]) -> str:
        """Build enhanced prompt (handling multiple errors with RAG results)"""
        error_sections = []
        
        # List of tactics that may involve theorem applications
        theorem_tactics = ["apply", "exact", "refine", "rw", "convert", "have", "suffices", "by"]

        for error in errors:
            # 1. Retrieve similar error
            similar_errors = self._retrieve_similar_error(error)
            
            # 2. Check if it's a theorem application error
            theorem_info = None
            error_msg = error.get("message", "").lower()
            if any(tactic in error_msg for tactic in theorem_tactics):
                func_name = self._extract_function_name(error["message"])
                if func_name:
                    theorem_info = self._retrieve_theorem_info(func_name)
            
            # Build prompt section for single error
            error_section = f"""
            [Error {len(error_sections)+1}/{len(errors)}]
            File: {self.lean_file}
            Line: {error['line']}
            Error: {error['message']}
            
            [Context]
            {self._get_context_lines(error['line'])}
            
            [Full Block Context]
            {error['full_context']}
            """

            # Add similar errors info if found
            if similar_errors:
                error_section += f"""
                [Top 3 Similar Error Solutions]
                ** Error 1 **
                Similarity : {similar_errors[0]["score"]}
                Type: {similar_errors[0]["matched_error"]["type"]}
                Message: {similar_errors[0]["matched_error"]["message"]}
                Fix Suggestion: {similar_errors[0]["matched_error"]["fix_suggestion"]}
                Example Solutions:
                {similar_errors[0]["matched_error"]["successful_fixes"]}
                ** Error 2 **
                Similarity : {similar_errors[1]["score"]}
                Type: {similar_errors[1]["matched_error"]["type"]}
                Message: {similar_errors[1]["matched_error"]["message"]}
                Fix Suggestion: {similar_errors[1]["matched_error"]["fix_suggestion"]}
                Example Solutions:
                {similar_errors[1]["matched_error"]["successful_fixes"]}
                ** Error 3 **
                Similarity : {similar_errors[2]["score"]}
                Type: {similar_errors[2]["matched_error"]["type"]}
                Message: {similar_errors[2]["matched_error"]["message"]}
                Fix Suggestion: {similar_errors[2]["matched_error"]["fix_suggestion"]}
                Example Solutions:
                {similar_errors[2]["matched_error"]["successful_fixes"]}
                """
            
            # Add theorem info if found
            if theorem_info:
                  error_section += "\n[Theorem Details]\n"
                  for thm in theorem_info:
                      name = thm.get("name", "Unknown")
                      statement = thm["info"].get("statement", "No statement")
                      description = thm["info"].get("description", "No description")
                      error_section += f"""
              Name: {name}
              Statement: {statement}
              Description: {description}
              """
            
            error_sections.append(error_section)
        
        prompt = f"""[Task] As an expert proficient in Lean, your task is to fix the Lean code with the error information Lean offers. Given the following code and list of compiler errors, return a fully fixed version. For definitions, please carefully fix the errors in the code, and for theorems, you can add `sorry` to fix the code if you cannot prove it.
        
        [Full Current Code]
        ```lean
        {self.current_code}
        ```
        The error message gives as:
        """
        prompt += "\n".join(error_sections)
        
        # Add general fix requirements
        prompt += """
        [Fix Requirements]
        1. Fix all above errors, output complete Lean4 code
        2. Return the entire file content, not just fixes.
        3. Wrap complete code in ```lean ```
        4. Don't fix errors individually, provide a unified solution file.
        5. Output ONLY the complete Lean4 code WITHOUT any explanations.
        """
        
        self.logger.debug(f"📝 Built multi-error enhanced prompt: {len(errors)} errors, {len(prompt)} chars")
        return prompt
    
    def _build_harmless_prompt(self, errors: List[Dict]) -> str:
        """Build prompt for creating harmless version with sorry statements."""
        self.logger.debug(f"📝 Building harmless prompt for {len(errors)} errors")
        
        context_lines = []
        for e in errors:
            context_lines.append(
                f"Line {e['line']} Column {e['column']}: {e['message']} "
                f"(Line content: {e.get('line_content', 'N/A')})"
            )
        context = "\n".join(context_lines)
        
        prompt = f"""
        [Task] As an expert proficient in Lean, your task is to sanitize a Lean 4 code containing errors by adding sorry at appropriate locations based on error reports, thereby producing a harmless version of the erroneous Lean 4 code that includes several sorry statements but no error messages.
        
        Here are some guidelines to follow:
        1. For a continuous proof segment: If an intermediate tactic fails, all subsequent tactics in that segment should be removed, and sorry should be added after the last correct tactic.
        2. For have tactics or multi-part proofs: If an error occurs in the proof of a have statement or within one part of a multi-part proof, only mark the affected part with sorry, leaving the rest intact.
        
        [Requirements] 
        1. Just place sorry do not correct the error
        2. Return the entire file content, not just fixes.
        3. Output ONLY the complete Lean4 code WITHOUT any explanations.
        
        Here is error message:
        {context}

        [Full Current Code]
        ```lean
        {self.current_code}
        ```
        """
        
        self.logger.debug(f"📝 Built harmless prompt with {len(prompt)} characters")
        return prompt

    # ---------------------------------------------------------------------------
    # Main Error Correction Functions
    # ---------------------------------------------------------------------------

    def correct_errors(self) -> bool:
        """
        Main error correction function with comprehensive logging and progress tracking.
        Attempts to fix errors by merging them and using rule-based fixes.
        
        Returns:
            bool: True if all errors were fixed successfully, False otherwise
        """
        self.logger.info("🚀 Starting error correction process")
        self.logger.info(f"📁 Processing file: {self.lean_file}")
        
        attempt_count = 0
        start_time = time.time()
        
        # Store previous code version for potential rollback
        previous_code = self.current_code
        
        # Session logging
        session_log = {
            "start_time": datetime.datetime.now().isoformat(),
            "original_size": len(self.original_code),
            "initial_errors": len(self._get_errors()),
            "attempts": []
        }
        
        self.logger.info(f"📊 Initial state: {session_log['original_size']} chars, "
                        f"{session_log['initial_errors']} errors")

        while attempt_count < CONFIG["max_attempts"]:
            attempt_start = time.time()
            attempt_count += 1
            
            self.logger.info(f"🔄 Attempt {attempt_count}/{CONFIG['max_attempts']}")
            
            # Store current code before attempting fixes
            current_file_path = self.lean_file
            errors = self._get_errors(file_path=current_file_path)
            
            if not errors:
                total_time = time.time() - start_time
                self.logger.info("🎉 All errors fixed successfully!")
                
                # Save final correct code
                fixed_file_path = self.lean_file.parent / (self.lean_file.stem + '_fixed.lean')
                self._write_file(fixed_file_path, self.current_code)
                self.logger.info(f"Final fixed code saved to: {fixed_file_path}")
                return True

            self.logger.info(f"⚠️  Found {len(errors)} errors, attempting to fix...")
            
            # Log error details
            for i, error in enumerate(errors[:5]):  # Show first 5 errors
                self.logger.debug(f"   Error {i+1}: Line {error.get('line', '?')} - "
                            f"{error.get('message', 'No message')[:100]}")
            if len(errors) > 5:
                self.logger.debug(f"   ... and {len(errors) - 5} more errors")

            # Build prompt and call model
            enhanced_prompt = self._build_enhanced_prompt(errors)
            suggestion = self._call_large_model(enhanced_prompt, {"line": "multiple", "message": "merged_errors"})

            # Record attempt
            attempt_info = {
                "attempt": attempt_count,
                "errors_count": len(errors),
                "prompt_length": len(enhanced_prompt),
                "response_length": len(suggestion) if suggestion else 0,
                "timestamp": datetime.datetime.now().isoformat()
            }

            if suggestion:
                suggested_code = self._extract_full_code(suggestion)
                if suggested_code and suggested_code != self.current_code:
                    self.logger.info("📦 Received code suggestion from model")
                    
                    # Store current state before applying changes
                    backup_code = self.current_code
                    
                    # Write suggested fix
                    fix_file_path = self.lean_file.parent / (self.lean_file.stem + f'_fix{attempt_count}.lean')
                    current_file_path = fix_file_path
                    self._write_file(fix_file_path, suggested_code)
                    
                    # Update current code state
                    self.current_code = suggested_code
                    
                    # Process the file
                    process_lean_file(fix_file_path)
                    
                    self.logger.info(f"💾 Suggested code saved to: {fix_file_path}")
                    # Check if fix resolved errors
                    new_errors = self._get_errors(file_path=current_file_path)

                    with open(fix_file_path, 'r', encoding='utf-8') as f:
                        suggested_code = f.read()
                    self.current_code = suggested_code
                    # Record successful fix with proper parameters
                    try :
                        self._record_successful_fix(backup_code, errors, suggested_code, new_errors)
                    except Exception as e:
                        self.logger.error(f"❌ Failed to record successful fix: {e}")
                    if not new_errors:
                        self.logger.info("✅ Model fix resolved all errors!")

                        # Save final fixed version
                        fixed_file_path = self.lean_file.parent / (self.lean_file.stem + '_fixed.lean')
                        self._write_file(fixed_file_path, suggested_code)
                        self.logger.info(f"💾 Final fixed code saved to: {fixed_file_path}")
                        
                        total_time = time.time() - start_time
                        self._log_session_summary(session_log, True, total_time)
                        return True
                    else:
                        self.logger.warning(f"❌ Model fix did not resolve errors "
                                        f"({len(new_errors)} errors remain)")
                        
                        # Rollback to previous code version if new errors are worse
                        if len(new_errors) >= len(errors):
                            self.logger.info("🔄 Rolling back to previous code version")
                            self.current_code = backup_code
                            self._write_file(current_file_path, backup_code)
                        
                        attempt_info["new_errors_count"] = len(new_errors)
                else:
                    self.logger.warning("⚠️  No valid code found in model response")
                    attempt_info["status"] = "invalid_response"
            else:
                self.logger.error("❌ No response received from model")
                attempt_info["status"] = "no_response"

            attempt_time = time.time() - attempt_start
            attempt_info["duration"] = attempt_time
            session_log["attempts"].append(attempt_info)
            
            self.logger.info(f"⏱️  Attempt {attempt_count} completed in {attempt_time:.2f}s")

        # Max attempts reached - restore original code if not successful
        if self.current_code != previous_code:
            self.logger.info("🔄 Restoring original code after failed attempts")
            self.current_code = previous_code
            self._write_file(self.lean_file, previous_code)
        
        total_time = time.time() - start_time
        self.logger.error(f"🛑 Reached maximum attempts ({CONFIG['max_attempts']}). "
                        f"Errors remain unresolved.")
        self._log_session_summary(session_log, False, total_time)
        return False

    
    def fix_errors(self) -> bool:
        """
        Fix errors by making the code harmless (adding sorry statements).
        More conservative approach that focuses on compilation rather than correctness.
        """
        self.logger.info("🚀 Starting harmless error fixing process")
        self.logger.info(f"📁 Processing file: {self.lean_file}")
        
        attempt_count = 0
        start_time = time.time()
        
        # Session logging
        session_log = {
            "start_time": datetime.datetime.now().isoformat(),
            "original_size": len(self.original_code),
            "initial_errors": len(self._get_errors()),
            "attempts": [],
            "mode": "harmless_fix"
        }
        
        self.logger.info(f"📊 Initial state: {session_log['original_size']} chars, "
                        f"{session_log['initial_errors']} errors")

        while attempt_count < CONFIG["final_max_attempts"]:
            attempt_start = time.time()
            attempt_count += 1
            
            self.logger.info(f"🔄 Attempt {attempt_count}/{CONFIG['final_max_attempts']} (harmless mode)")
            
            current_file_path = self.lean_file
            errors = self._get_errors(file_path=current_file_path)
            
            if not errors:
                total_time = time.time() - start_time
                self.logger.info("🎉 All errors resolved (harmless version)!")
                
                # Save final harmless code
                fixed_file_path = self.lean_file.parent / (self.lean_file.stem + '_harmless.lean')
                self._write_file(fixed_file_path, self.current_code)
                self.logger.info(f"💾 Final harmless code saved to: {fixed_file_path}")

                # Log session summary
                self._log_session_summary(session_log, True, total_time)
                return True

            self.logger.info(f"⚠️  Found {len(errors)} errors, making harmless...")

            # Build harmless prompt and call model
            harmless_prompt = self._build_harmless_prompt(errors)
            suggestion = self._call_large_model(harmless_prompt, {"line": "multiple", "message": "harmless_fix"})

            # Record attempt
            attempt_info = {
                "attempt": attempt_count,
                "errors_count": len(errors),
                "prompt_length": len(harmless_prompt),
                "response_length": len(suggestion) if suggestion else 0,
                "timestamp": datetime.datetime.now().isoformat(),
                "mode": "harmless"
            }

            if suggestion:
                suggested_code = self._extract_full_code(suggestion)
                if suggested_code and suggested_code != self.current_code:
                    self.logger.info("📦 Received harmless code suggestion from model")
                    # Store current state before applying changes
                    backup_code = self.current_code
                    
                    # Write suggested fix
                    fix_file_path = self.lean_file.parent / (self.lean_file.stem + f'_fix{attempt_count}.lean')
                    current_file_path = fix_file_path
                    self._write_file(fix_file_path, suggested_code)
                    
                    # Update current code state
                    self.current_code = suggested_code
                    
                    # Process the file
                    process_lean_file(fix_file_path)
                    with open(fix_file_path, 'r', encoding='utf-8') as f:
                        suggested_code = f.read()
                    self.current_code = suggested_code
                    self.logger.info(f"💾 Suggested harmless code saved to: {fix_file_path}")

                    # Check if fix resolved errors
                    new_errors = self._get_errors(file_path=current_file_path)
                    if not new_errors:
                        self.logger.info("✅ Harmless fix resolved all compilation errors!")
                        
                        # Save final harmless version
                        fixed_file_path = self.lean_file.parent / (self.lean_file.stem + '_fixed.lean')
                        self._write_file(fixed_file_path, suggested_code)
                        process_lean_file(fixed_file_path)
                        self.logger.info(f"💾 Final harmless code saved to: {fixed_file_path}")
                        
                        total_time = time.time() - start_time
                        self._log_session_summary(session_log, True, total_time)
                        return True
                    else:
                        self.logger.warning(f"❌ Harmless fix did not resolve errors "
                                          f"({len(new_errors)} errors remain)")
                        attempt_info["new_errors_count"] = len(new_errors)
                else:
                    self.logger.warning("⚠️  No valid harmless code found in model response")
                    attempt_info["status"] = "invalid_response"
            else:
                self.logger.error("❌ No response received from model")
                attempt_info["status"] = "no_response"

            attempt_time = time.time() - attempt_start
            attempt_info["duration"] = attempt_time
            session_log["attempts"].append(attempt_info)
            
            self.logger.info(f"⏱️  Attempt {attempt_count} completed in {attempt_time:.2f}s")

        # Max attempts reached
        total_time = time.time() - start_time
        self.logger.error(f"🛑 Reached maximum attempts ({CONFIG['final_max_attempts']}). "
                         f"Errors remain unresolved.")
        self._log_session_summary(session_log, False, total_time)
        return False
    
    # ---------------------------------------------------------------------------
    # Utility and Logging Functions
    # ---------------------------------------------------------------------------
    
    def _log_session_summary(self, session_log: Dict, success: bool, total_time: float):
        """Log comprehensive session summary."""
        summary = {
            **session_log,
            "end_time": datetime.datetime.now().isoformat(),
            "total_duration": total_time,
            "success": success,
            # "final_errors": len(self._get_errors()),
            "total_attempts": len(session_log["attempts"])
        }

        # Get the log file path from the logger's file handler
        log_file_path = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename
                break
        
        # Write detailed session log
        try:
            if log_file_path:
                session_file = Path(log_file_path).with_suffix('.session.json')
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                self.logger.debug(f"📋 Session summary saved to: {session_file}")
            else:
                self.logger.warning("⚠️ No log file handler found - skipping session file save")
        except Exception as e:
            self.logger.error(f"❌ Failed to save session summary: {e}")
        
        # Log summary to main log
        self.logger.info("=" * 50)
        self.logger.info("📊 SESSION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"🎯 File: {self.lean_file}")
        self.logger.info(f"⏱️  Total time: {total_time:.2f}s")
        self.logger.info(f"🔄 Attempts: {len(session_log['attempts'])}")
        self.logger.info(f"📊 Initial errors: {session_log['initial_errors']}")
        # self.logger.info(f"📊 Final errors: {summary['final_errors']}")
        self.logger.info(f"✅ Success: {success}")
        self.logger.info("=" * 50)

    def _record_successful_fix(self, original_code: str, original_errors: List[Dict], fixed_code: str, fixed_errors: List[Dict]):
        """
        Record successful fixes by comparing fixed and original code blocks.
        Enhanced version that tracks which specific code blocks were fixed.
        """
        if re.search(r'\bsorry\b', fixed_code):
            return False
        # Identify fixed blocks (had errors before, no errors now)
        fixed_blocks = self._get_error_context_snippet(
            original_code, original_errors, fixed_code, fixed_errors)
        if not fixed_blocks:
            self.logger.debug("No complete code blocks were fixed")
            return False
        new_knowledge = []
        # Create new knowledge entries for each fixed block
        for block in fixed_blocks:
            # Generate error type classification using LLM
            error_type = self._classify_error_type(block['original_error']
            )
            new_entry = {
            "error_type": error_type,
            "error_message": block['original_error'],
            "context": block['original_block'], 
            "successful_fixes": [block['fixed_block']],  
            "fix_suggestion": self._generate_fix_explanation(
                block['original_block'],  
                block['fixed_block']     
              )
            }
            new_knowledge.append(new_entry)
            self.logger.info(
                f"Recorded fix for {error_type} - "
                f"{block['original_error'][:100]}..."
            )
        self._save_error_knowledge(new_knowledge)
        self.logger.info(f"💾 Saved {len(fixed_blocks)} new fixes to knowledge base")

    def _classify_error_type(self, error: str) -> str:
        """
        Classify error type using LLM by analyzing the error message.
        
        Returns:
            Classified error type (e.g., "Type mismatch")
        """
        error_message = error.strip()
        valid_types = [
            "Type mismatch",
            "Unknown identifier",
            "Incorrect syntax", 
            "Missing definition",
            "Theorem application error",
            "Type class instance",
            "Scope/variable issue",
            "Pattern matching failure",
            "Termination proof",
            "Other"
        ]
        
        prompt = f"""Analyze this Lean 4 error and classify its type:

        Error: {error_message}

        Please classify this error into one of these common Lean 4 error types:
        {', '.join(valid_types[:-1])}

        Return ONLY the most specific error type name from the list above, nothing else."""

        try:
            response = self._call_large_model(prompt, {"line": "classification", "message": error_message})
            response = response.strip()
            
            valid_types_lower = [t.lower() for t in valid_types]
            response_lower = response.lower()

            # Try exact match first
            for err_type, err_lower in zip(valid_types, valid_types_lower):
                if err_lower in response_lower:
                    self.logger.debug(f"Classified error as: {err_type} (exact match)")
                    return err_type
                    
            # Try fuzzy matching (remove punctuation)
            clean_response = re.sub(r'[^\w\s]', '', response_lower)
            for err_type, err_lower in zip(valid_types, valid_types_lower):
                if re.search(r'\b' + re.escape(err_lower) + r'\b', clean_response):
                    self.logger.debug(f"Classified error as: {err_type} (fuzzy match)")
                    return err_type
                    
            # Try keyword matching
            keyword_mapping = {
                r'type mismatch|types do not match': 'Type mismatch',
                r'unknown identifier|unknown constant': 'Unknown identifier',
                r'syntax error|unexpected token': 'Incorrect syntax',
                r'missing definition|no such definition': 'Missing definition',
                r'theorem|lemma': 'Theorem application error',
                r'type class|failed to synthesize': 'Type class instance',
                r'scope|variable not found': 'Scope/variable issue',
                r'pattern matching': 'Pattern matching failure',
                r'termination': 'Termination proof'
            }
            
            for pattern, err_type in keyword_mapping.items():
                if re.search(pattern, response_lower):
                    self.logger.debug(f"Classified error as: {err_type} (keyword match)")
                    return err_type
                    
            self.logger.debug(f"Could not classify error, response: {response}")
            return "unclassified"
            
        except Exception as e:
            self.logger.error(f"Error type classification failed: {e}")
            return "unclassified"

    def _generate_fix_explanation(self, original: str, fixed: str) -> str:
        """
        Generate professional explanation of the fix using LLM analysis.
        
        Args:
            original: Original code snippet with error
            fixed: Fixed version of the code
            
        Returns:
            Detailed explanation of what was fixed and why
        """
        prompt = f"""As a Lean 4 expert, analyze these code changes and explain the fix professionally:

        Original Code (with error):
        ```lean""" + original + """
        ```
        Fixed Code:
        ```""" + fixed + """
        ```
        Please provide a concise but detailed explanation that:

        Identifies the root cause of the error
        Explains what specifically was changed
        Describes why the fix works
        Uses appropriate Lean terminology
        Is under 200 words
        Format your response as:

        Error Type: <type>

        Root Cause: <cause>

        Fix Description: <description>

        Why It Works: <explanation>"""

        try:
            response = self._call_large_model(prompt, {"line": "explanation", "message": "code fix"})

            # Post-process the response
            explanation = response.strip()
            if not explanation:
                raise ValueError("Empty response from model")
                
            # Ensure the explanation isn't too long
            if len(explanation) > 1000:
                explanation = explanation[:1000] + "... [truncated]"
                
            self.logger.debug(f"Generated fix explanation: {explanation[:200]}...")
            return explanation
        
        except Exception as e:
            self.logger.error(f"Failed to generate fix explanation: {e}")
            # Fallback to simple diff
            return self._generate_simple_diff_explanation(original, fixed)

    def _generate_simple_diff_explanation(self, original: str, fixed: str) -> str:
        """Fallback simple diff-based explanation"""
        original_lines = original.split('\n')
        fixed_lines = fixed.split('\n')

        explanation = []
        for i, (orig, fix) in enumerate(zip(original_lines, fixed_lines)):
            if orig != fix:
                explanation.append(
                    f"Line {i+1}: Changed '{orig.strip()}' to '{fix.strip()}'"
                )

        if not explanation:
            return "Exact changes unclear - please review the full fixed snippet"

        return "Changes made:\n" + "\n".join(explanation)

    def _get_error_context_snippet(
            self, 
            original_code: str, 
            original_errors: List[Dict],
            fixed_code: str, 
            fixed_errors: List[Dict]
        ) -> List[Dict]:
        """
        Identifies fixed code blocks by comparing error locations before and after fixes.
        
        Args:
            original_code: Complete code before fixes
            original_errors: List of error dicts from original code
            fixed_code: Complete code after fixes
            fixed_errors: List of error dicts from fixed code
            
        Returns:
            List of dicts containing:
            - original_block: Code block from original code with error
            - original_error: Error message from original code
            - fixed_block: Corresponding fixed code block
            - block_info: Type and name of the code block
        """
        
        # Split codes into lines for processing
        original_lines = original_code.split('\n')
        fixed_lines = fixed_code.split('\n')
        # Extract error line numbers from fixed code for quick lookup
        fixed_error_lines = {e.get('line', -1) for e in fixed_errors}
        
        results = []
        
        for error in original_errors:
            line_num = error.get('line', 1)
            # 1. Find the code block boundaries in original code
            start, end = self._find_code_block_boundaries(line_num)
            if start >= end:
                continue  # Skip invalid blocks
            # 2. Extract the original block and its header
            original_block = '\n'.join(original_lines[start:end])
            block_header = original_lines[start].strip()
            # Parse block type and name
            block_type, block_name = "", ""
            for t in ['theorem', 'lemma', 'def', 'instance', 'class', 'structure', 'variable']:
                if block_header.startswith(t + ' '):
                    block_type = t
                    block_name = block_header[len(t):].strip().split()[0]
                    break
            if not block_type:  # Not a supported block type
                continue
                
            # 3. Find matching block in fixed code using fuzzy matching
            normalized_name = self._normalize_name(block_name)
            for i in range(len(fixed_lines)):
                line = fixed_lines[i].strip()
                if line.startswith(block_type + ' '):
                    # Extract name from fixed code
                    fixed_name = line[len(block_type):].strip().split()[0]
                    if self._normalize_name(fixed_name) == normalized_name:
                        # Found matching block, get its boundaries
                        fixed_start, fixed_end = self._find_code_block_boundaries(i+1)  # +1 for 1-based line num
                        if fixed_start >= fixed_end:
                            continue
                        # 4. Check if this block has errors in fixed code
                        has_errors = any(
                            fixed_start < e_line <= fixed_end 
                            for e_line in fixed_error_lines
                        )
                        
                        if not has_errors:
                            # This block was fixed successfully
                            fixed_block = '\n'.join(fixed_lines[fixed_start:fixed_end])
                            results.append({
                                'original_block': original_block,
                                'original_error': error.get('message', ''),
                                'fixed_block': fixed_block,
                                'block_info': f"{block_type} {block_name}"
                            })
                        break
            
        return results
    
    def _find_code_block_boundaries(self, line_num: int) -> Tuple[int, int]:
        """
        Find the start and end line numbers of the code block containing the given line.
        Returns (start_line, end_line)
        """
        lines = self.current_code.split('\n')
        block_types = ['theorem', 'lemma', 'def', 'instance', 'class', 'structure', 'variable']
        
        # Initialize boundaries
        block_start = 0
        block_end = len(lines)
        
        # Find start of block (look upwards for block starters)
        for i in range(line_num-1, -1, -1):
            stripped = lines[i].strip()
            if any(stripped.startswith(t + ' ') or stripped.startswith(t + '\n') 
                  for t in block_types):
                block_start = i
                break
        
        # Find end of block (look for next block starter or EOF)
        for i in range(line_num, len(lines)):
            stripped = lines[i].strip()
            # Check for next block start or end of proof
            if (any(stripped.startswith(t + ' ') for t in block_types) or
                stripped == 'end' or 
                stripped.startswith('#')):
                block_end = i
                break
                
        return block_start, block_end

# ---------------------------------------------------------------------------
# Batch Processing Functions
# ---------------------------------------------------------------------------

def correct_all_trials_in_folder(folder_path: str, logger: logging.Logger = None, max_workers: int = 8) -> List[Tuple[str, bool]]:
    """
    Process all Lean files in a folder with comprehensive logging and progress tracking.
    Uses the correct_errors method for each file.
    """
    folder_path = Path(folder_path)
    # Use provided logger or create new one
    logger = logger if logger else setup_logging()
    logger.info(f"🚀 Starting batch correction in folder: {folder_path}")
    start_time = time.time()
    
    # Find all Lean files
    lean_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.lean'):
                lean_files.append(Path(root) / file)

    logger.info(f"🔍 Found {len(lean_files)} Lean files to process")
    
    if not lean_files:
        logger.warning("⚠️  No Lean files found in the specified folder")
        return []

    def process_file(file_path: Path, logger: logging.Logger) -> Tuple[str, bool]:
        """Process a single file and return results."""
        file_start = time.time()
        
        try:
            logger.info(f"🎯 Processing: {file_path}")
            corrector = LeanErrorCorrector(str(file_path), logger)
            success = corrector.correct_errors()
            
            process_time = time.time() - file_start
            if success:
                logger.info(f"✅ Correction completed successfully in {process_time:.2f}s")
            else:
                logger.warning(f"⚠️  Correction failed after {process_time:.2f}s")
            
            return (str(file_path), success)
            
        except Exception as e:
            process_time = time.time() - file_start
            logger.error(f"❌ Exception during processing ({process_time:.2f}s): {e}")
            return (str(file_path), False)

    # Parallel processing with progress tracking
    results = []
    completed = 0
    
    logger.info(f"🚀 Starting parallel processing with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_file, path, logger): path for path in lean_files}
        
        # Process completed tasks
        for future in as_completed(future_to_path):
            path, success = future.result()
            completed += 1
            
            progress = (completed / len(lean_files)) * 100
            logger.info(f"📊 Progress: {completed}/{len(lean_files)} ({progress:.1f}%) - "
                       f"{'✅ SUCCESS' if success else '❌ FAILED'}: {Path(path).name}")
            
            results.append((path, success))

    # Final summary
    total_time = time.time() - start_time
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    logger.info("=" * 80)
    logger.info("🏁 BATCH CORRECTION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"📁 Folder: {folder_path}")
    logger.info(f"📊 Total files: {len(lean_files)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success rate: {(successful/len(lean_files)*100):.1f}%")
    logger.info(f"⏱️  Total time: {total_time:.2f}s")
    logger.info(f"⏱️  Average time per file: {total_time/len(lean_files):.2f}s")
    logger.info("=" * 80)
    
    return results


def fix_all_trials_in_folder(folder_path: str, logger: logging.Logger = None, max_workers: int = 8) -> List[Tuple[str, bool]]:
    """
    Process all Lean files in a folder using the harmless fix method.
    Uses the fix_errors method for each file.
    """
    folder_path = Path(folder_path)
    # Use provided logger or create new one
    logger = logger if logger else setup_logging()
    logger.info(f"🚀 Starting batch harmless fixing in folder: {folder_path}")
    start_time = time.time()
    
    # Find all Lean files
    lean_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.lean'):
                lean_files.append(Path(root) / file)

    logger.info(f"🔍 Found {len(lean_files)} Lean files to process (harmless mode)")
    
    if not lean_files:
        logger.warning("⚠️  No Lean files found in the specified folder")
        return []

    def process_file(file_path: Path, logger: logging.Logger) -> Tuple[str, bool]:
        """Process a single file with harmless fixes."""
        file_start = time.time()
        
        try:
            logger.info(f"🎯 Processing (harmless): {file_path}")
            
            # First, check if the file already compiles without errors
            logger.info(f"🔍 Pre-checking compilation status...")
            check_start = time.time()
            
            try:
                # Use the compile_file function directly to check errors
                _, info, _ = compile_file(str(file_path))
                _, errors = parse_lean_output_with_context(info)
                check_time = time.time() - check_start
                
                if not errors:
                    # File already compiles successfully, no need to process
                    total_time = time.time() - file_start
                    logger.info(f"✅ File already compiles without errors ({check_time:.2f}s), skipping model processing")
                    logger.info(f"✅ Pre-check completed successfully in {total_time:.2f}s")
                    return (str(file_path), True)
                else:
                    # File has errors, proceed with harmless fixing
                    logger.info(f"⚠️  Found {len(errors)} errors ({check_time:.2f}s), proceeding with harmless fixing...")
                    
            except Exception as e:
                check_time = time.time() - check_start
                logger.warning(f"⚠️  Pre-compilation check failed ({check_time:.2f}s): {e}, proceeding with harmless fixing...")
            
            # Proceed with normal harmless fixing
            corrector = LeanErrorCorrector(str(file_path), logger)
            success = corrector.fix_errors()
            
            process_time = time.time() - file_start
            if success:
                logger.info(f"✅ Harmless fix completed successfully in {process_time:.2f}s")
            else:
                logger.warning(f"⚠️  Harmless fix failed after {process_time:.2f}s")
            
            return (str(file_path), success)
            
        except Exception as e:
            process_time = time.time() - file_start
            logger.error(f"❌ Exception during processing ({process_time:.2f}s): {e}")
            return (str(file_path), False)

    # Parallel processing
    results = []
    completed = 0
    skipped_count = 0
    processed_count = 0
    
    logger.info(f"🚀 Starting parallel harmless processing with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_file, path, logger): path for path in lean_files}
        
        for future in as_completed(future_to_path):
            path, success = future.result()
            completed += 1
            
            # Count files that were skipped vs actually processed
            # This is a simple heuristic - if it completed very quickly, it was likely skipped
            if success:
                # We can't easily distinguish here, but the individual logs will show
                pass
            
            progress = (completed / len(lean_files)) * 100
            logger.info(f"📊 Progress: {completed}/{len(lean_files)} ({progress:.1f}%) - "
                       f"{'✅ SUCCESS' if success else '❌ FAILED'}: {Path(path).name}")
            
            results.append((path, success))

    # Final summary
    total_time = time.time() - start_time
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    logger.info("=" * 80)
    logger.info("🏁 BATCH HARMLESS FIXING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"📁 Folder: {folder_path}")
    logger.info(f"📊 Total files: {len(lean_files)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success rate: {(successful/len(lean_files)*100):.1f}%")
    logger.info(f"⏱️  Total time: {total_time:.2f}s")
    logger.info(f"⏱️  Average time per file: {total_time/len(lean_files):.2f}s")
    logger.info("=" * 80)
    
    return results
