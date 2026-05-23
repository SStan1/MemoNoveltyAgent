#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-report Checklist Evaluation Script.

Evaluates one report at a time. The report path is specified in the config JSON file.
Results are saved to the configured output directory.

Usage:
    python Evaluation.py
    python Evaluation.py --config_path ./Evaluation.json
    python Evaluation.py --rerun_dimension depth
"""

import os
import sys
import json
import re
import time
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from ragflow_sdk import RAGFlow

# Resolve script directory for relative path computation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIFEST_CACHE = {}


def resolve_path(path):
    """Resolve a path relative to the script directory if it is not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def _safe_output_name(value):
    """Create a filesystem-safe, readable output folder name."""
    value = re.sub(r"[\\/]+", "__", str(value))
    value = re.sub(r"[\x00-\x1f<>:\"|?*]+", "_", value)
    value = re.sub(r"\s+", " ", value).strip(" ._")
    return value or "report"


def _title_from_parent_dir_name(dirname):
    """Recover a readable paper title from dataset folder names like 28_Title_With_Underscores."""
    raw = str(dirname).strip()
    title = re.sub(r"^\d+_", "", raw)
    title = title.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title or raw


def _title_from_report_basename(report_path, paths_cfg):
    """Recover the original paper title from filenames like Title__ReportSource.txt."""
    title = os.path.splitext(os.path.basename(report_path))[0]
    separator = str(paths_cfg.get("paper_title_source_separator") or "__")
    if separator and separator in title:
        title = title.split(separator, 1)[0]
    if title.endswith("_polished"):
        title = title[: -len("_polished")]
    return title.strip()


def _path_value_list(value):
    """Small local list normalizer used before the main config helpers are defined."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _report_name_roots(paths_cfg):
    roots = []
    for key in ("report_name_root", "report_name_roots", "report_dir", "report_dirs", "reports_dir"):
        roots.extend(_path_value_list(paths_cfg.get(key)))
    return roots


def derive_report_output_name(report_path, paths_cfg):
    """
    Derive the output folder name for a report.

    In relative mode, the name includes the report's relative path under the configured
    report root, so repeated basenames such as ACL_DeepReviewer.txt do not collide.
    """
    mode = str(paths_cfg.get("output_name_mode") or paths_cfg.get("paper_name_mode") or "basename").lower()
    if mode in {"relative", "relative_path", "relpath"}:
        abs_report = os.path.abspath(report_path)
        for root in _report_name_roots(paths_cfg):
            abs_root = os.path.abspath(resolve_path(root))
            try:
                if os.path.commonpath([abs_report, abs_root]) != abs_root:
                    continue
            except ValueError:
                continue
            rel_path = os.path.relpath(abs_report, abs_root)
            rel_no_ext = os.path.splitext(rel_path)[0]
            return _safe_output_name(rel_no_ext)

    return _safe_output_name(os.path.splitext(os.path.basename(report_path))[0])


def _load_manifest_map(manifest_path):
    resolved_manifest = os.path.abspath(resolve_path(manifest_path))
    if resolved_manifest in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[resolved_manifest]

    with open(resolved_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dataset_root = os.path.dirname(resolved_manifest)
    path_to_title = {}
    for paper in manifest.get("papers", []):
        title = paper.get("paper_title", "")
        for report_info in paper.get("reports", {}).values():
            dataset_path = report_info.get("dataset_path")
            if not dataset_path:
                continue
            abs_report = os.path.abspath(os.path.join(dataset_root, dataset_path))
            path_to_title[os.path.normcase(abs_report)] = title

    _MANIFEST_CACHE[resolved_manifest] = path_to_title
    return path_to_title


def lookup_paper_title_from_manifest(report_path, paths_cfg):
    manifest_path = (
        paths_cfg.get("report_manifest_path") or
        paths_cfg.get("dataset_manifest_path") or
        paths_cfg.get("manifest_path")
    )
    if not manifest_path:
        return None
    try:
        manifest_map = _load_manifest_map(manifest_path)
    except Exception as e:
        print(f"[WARNING] Could not load report manifest {manifest_path!r}: {e}")
        return None
    return manifest_map.get(os.path.normcase(os.path.abspath(report_path)))


def derive_paper_title(report_path, paths_cfg):
    """Derive the human paper title used in prompts/RAG queries."""
    manifest_title = lookup_paper_title_from_manifest(report_path, paths_cfg)
    if manifest_title:
        return manifest_title

    title_mode = str(paths_cfg.get("paper_title_mode") or "basename").lower()
    if title_mode in {"parent", "parent_dir", "parent_directory"}:
        return _title_from_parent_dir_name(os.path.basename(os.path.dirname(report_path)))
    if title_mode in {"basename_before_source", "basename_source", "basename_without_source"}:
        return _title_from_report_basename(report_path, paths_cfg)

    return os.path.splitext(os.path.basename(report_path))[0]


class ReportEvaluator:
    """Evaluates a single report against the checklist evaluation framework."""

    # Retry configuration
    MAX_RETRIES = 5
    BASE_RETRY_DELAY = 2  # seconds

    def __init__(self, config):
        """
        Initialize the evaluator for a single report.

        Args:
            config: Configuration dictionary. Must contain paths.report_path.
        """
        self.config = config

        # Resolve report path from config
        raw_report_path = self.config['paths']['report_path']
        self.report_path = resolve_path(raw_report_path)

        if not os.path.exists(self.report_path):
            raise FileNotFoundError(f"Report file not found: {self.report_path}")

        # Derive paper name from the report filename (without extension).
        # For consistency datasets, use filenames like
        #   Original Paper Title__ReportSource.txt
        # so the original RAGFlow dataset prefix remains unchanged while output
        # folders stay unique.
        self.paper_name = os.path.splitext(os.path.basename(self.report_path))[0]

        # Load evaluation framework data
        questions_path = resolve_path(self.config['paths']['questions_json'])
        with open(questions_path, 'r', encoding='utf-8') as f:
            self.evaluation_data = json.load(f)

        self.output_base_dir = resolve_path(self.config['paths']['output_base_dir'])
        self.dimensions = self.evaluation_data['evaluation_framework']['dimensions']

        # Rerun dimension switch (None / "" / "none" means disabled).
        # Supports both legacy rerun_dimension="depth" and
        # rerun_dimensions=["fluency", "depth"].
        self.rerun_dimensions = []
        raw_rerun_dimensions = self.config.get("rerun_dimensions", None)
        if raw_rerun_dimensions is None:
            raw_rerun_dimensions = self.config.get("rerun_dimension", None)

        if raw_rerun_dimensions is not None:
            if isinstance(raw_rerun_dimensions, (list, tuple, set)):
                requested_dimensions = raw_rerun_dimensions
            else:
                requested_dimensions = str(raw_rerun_dimensions).split(",")

            lower_map = {k.lower(): k for k in self.dimensions.keys()}
            for rd in requested_dimensions:
                rd = str(rd).strip()
                if rd == "" or rd.lower() == "none":
                    continue
                matched = lower_map.get(rd.lower(), None)
                if matched is None:
                    print(f"[WARNING] rerun dimension '{rd}' not found in dimension list. "
                          f"Ignoring it.")
                    continue
                if matched not in self.rerun_dimensions:
                    self.rerun_dimensions.append(matched)

        self.rerun_dimension = (
            self.rerun_dimensions[0] if len(self.rerun_dimensions) == 1 else None
        )

        # Initialize OpenAI API client
        self.openai_client = openai.OpenAI(
            api_key=self.config['api_credentials']['openai_api_key'],
            base_url=self.config['api_endpoints']['openai_base_url']
        )

        # Initialize RAGFlow client
        self.rag_object = RAGFlow(
            api_key=self.config['api_credentials']['ragflow_api_key'],
            base_url=self.config['api_endpoints']['ragflow_base_url']
        )

        # Default article template (used when RAG retrieval is not applicable for a group)
        self.default_article = """


### OBJECTIVE:
You are an expert in academic paper analysis. Your objective is to produce a concise and precise 3-section report on the paper '{paper_name}'. You should evaluate the true novelty of the paper '{paper_name}' by comparing it against its most relevant works. Use your tools to research '{paper_name}' and its related papers (you should at least look up its references papers). Your entire analysis must be based strictly on the information you retrieve. Your analysis must be based **exclusively** on information retrieved using the available tools (e.g., Web_Access_search). Do not speculate or invent any details (e.g., metrics, datasets, architectures, baselines) not explicitly present in the retrieved text. If information is insufficient for any section, you must clearly state: "Insufficient information retrieved for [relevant section/point]; analysis limited." Below is the report template I have provided for you. You should follow this template when writing your report.

---

### MAIN SECTIONS (exactly these, numbered 1-3):


---
Below are specific guidelines on what each section should include. You must strictly follow my instructions below when writing each section and must not make any unauthorized changes.

#### **Section 1. Paper Content Summary**
Provide a concise summary of the paper's content in a single, factual paragraph derived only from the retrieved text. Include its main task or objective, the problems it aims to solve, and its key methods or approaches. Briefly explain the meaning of any uncommon abbreviations (e.g., define a rare term like "Hamilton-Jacobi-Bellman (HJB)"; skip common ones like CNN or NLP).

---

#### **Section 2. Point-wise Novelty Analysis**
For each novelty point you identify from the retrieved text, create a numbered subsection with its classification in the header (e.g., **### 2.1. Novelty Point 1: (Classification: Methodological/Algorithmic)**, **### 2.2. Novelty Point 2: (Classification: Theoretical)**, etc.). Within each numbered subsection, structure your analysis into four labeled parts: **a) Claimed Novelty**, **b) Similarities**, **c) Unique Differences**, and **d) Unique Feature Description**.

**a) Claimed Novelty:**  
Identify and list the paper's claimed novelty points. For each point, you must provide a classification (already shown in the subsection header) and a brief explanation of the novelty.

*   **Classification Types (MUST use one):**
    *   **Methodological/Algorithmic:** A new or significantly improved algorithm, model, or technique for solving a computational problem.
    *   **Theoretical:** A new formal theory, mathematical proof, or conceptual framework that deepens understanding.
    *   **System/Infrastructure:** The design and evaluation of a novel software/hardware system or architecture with new capabilities.
    *   **Dataset/Benchmark:** The creation of a new, high-quality dataset or benchmark enabling new research or more rigorous evaluation.
    *   **Empirical/Analytical:** A non-obvious insight derived from rigorous experimentation or large-scale analysis.
    *   **Task/Application:** The formulation of a new computational problem or the novel application of existing techniques to another domain.

**b) Similarities:**  
Based on the retrieved related papers, summarize any comparisons between this novelty point and existing work. Detail similar objectives, structures, or mechanisms, providing specific examples where possible. If no similarities are mentioned, state: "No explicit similarities were identified in the retrieved text."

*   **Example:** "The paper's use of a neural ODE to model temporal evolution in the latent space shares a conceptual foundation with [Existing Paper B, 'Latent ODEs for Irregular Time Series']. Both works leverage the continuous-time modeling capabilities of ODEs to handle asynchronous data. The shared mechanism is the optimization of an ODE solver's backpropagation path via the adjoint method."

**c) Unique Differences:**  
Excluding the similarities mentioned in section (b), summarize what makes this novelty unique (e.g., features, integration pattern, objective), elaborating on specific contrasts derived from the comparison text. This must reflect ONLY what is stated in the retrieved text. If the text does not specify any differences, state: "Unique differences were not specified in the retrieved text." If the comparison text for a point seems incomplete, append your analysis with "..."

*   **Example:** "The primary distinction is its Methodological/Algorithmic novelty. While [Existing Paper B] defines its ODE function over a static latent vector, this paper introduces a novel 'Graph-Informed ODE' function where the derivative at time `t` is conditioned on the graph's adjacency matrix. This structural adaptation of the ODE dynamics to explicitly incorporate relational information is the core unique contribution."

**d) Unique Feature Description:**  
Based on the genuinely unique aspects identified in part (c), answer the relevant questions below to analyze the novelty's significance. Organize your answer into a single paragraph. If part (c) is empty or shows no uniqueness, skip this section. If the retrieved text is insufficient for this analysis, state: "Insufficient detail in retrieved text to conduct a full analysis for this point."

*   **If Methodological/Algorithmic:**
    *   How is the claimed uniqueness technically implemented at a structural or procedural level (e.g., modified layer, new fusion sequence)?
    *   What is the specific claimed benefit (e.g., improved accuracy, efficiency), and how is it substantiated by the evidence presented (e.g., ablation studies, comparative experiments)?

*   **If Theoretical:**
    *   What is the central theoretical claim (e.g., theorem, proof), and what are its primary implications?
    *   How is the claim's rigor established (e.g., formal proof)? Is there enough detail in the text to assess its soundness?

*   **If System/Infrastructure:**
    *   What real-world problem does this system solve, and how does its design uniquely address it?
    *   How is the system's practicality and benefit demonstrated (e.g., performance benchmarks, case studies)?

*   **If Dataset/Benchmark:**
    *   What specific gap in existing resources does this new dataset or benchmark fill (e.g., scale, diversity, annotation quality)?
    *   What new research questions or capabilities does it enable for the community?

*   **If Empirical/Analytical:**
    *   What is the central new insight or finding? Does it confirm, challenge, or refine existing beliefs?
    *   How is the finding's validity supported by the experimental design described in the text?

*   **If Task/Application:**
    *   Why is the newly defined task or novel application important and non-obvious?
    *   How is the task formulated and evaluated? What makes the application of existing methods to this new domain innovative?

---

#### **Section 3. Novelty Summary**

Based on the point-wise novelty analysis in Section 2, write a comprehensive, detailed, and faithful paragraph summarizing the paper's novel contributions. This paragraph should first articulate the paper's overall novelty characteristics (e.g., integration-driven approach, incremental optimization, theoretical breakthrough). Subsequently, **you must critically examine** the novelty points analyzed in Section 2 from both positive and negative perspectives:

*   **Genuine Novelties:** Clearly identify which novelty point(s) constitute **genuine, significant contributions**. For each major contribution, synthesize your analyses from sections 2.c and 2.d to explain **what** the core unique idea is, **how** it is technically implemented or realized, and **what** its claimed significance or impact is (e.g., improved performance, new capability), citing the specific evidence described in the previous text.

*   **Aspects Lacking Originality:** Based on the analysis in Section 2, clearly identify which of the authors' claimed "novelties" lack substantial originality. Your reasoning must be grounded in the similarities summarized in section 2.b. Categorize each non-novel aspect as: Direct Repetition (nearly identical to prior work), Minor Adjustment (trivial modification of existing methods), Simple Combination (straightforward integration without novel synergy), Standard Practice (widely-used conventional approaches), or Other (specify the nature). For each non-novel aspect identified, provide a one-sentence improvement suggestion.

Finally, based on the dialectical analysis above, provide a concluding one-line summary.

**Final One-line Summary:**  
Based on your analysis above, summarize the overall level of novelty, key strengths, and limitations of this paper in a brief sentence. In this sentence, you should follow the rules below to use a score and word to summarize the paper's level of novelty. The specific rating rules are as follows:

**4 - Excellent / Transformative**  
Presents an entirely new question, a highly novel methodology, or a surprising perspective with the potential to open new research avenues or shift the community's understanding of a subject. This work represents a "bold concept" exploring an "under-researched area," embodying true "frontier exploration."

**3 - Good / Significant**  
Achieves clear and substantial progress on an existing problem. This work introduces a well-motivated and non-obvious methodology, or a novel and creative combination of existing ideas. It contributes "novel and useful methodologies" or presents "substantial new ideas."

**2 - Fair / Incremental**  
Effectively but limitedly extends existing work. Contributions may involve minor algorithmic tweaks or direct application of known techniques to a new but similar domain. This work may improve SOTA, but its core ideas are merely "incremental" and lack significant originality.

**1 - Poor / Insufficient**  
Lacks clear original contributions, or presents ideas that are well-known, trivial, or previously published. Claims of novelty are unsubstantiated or incorrect. This work "has been done before."

---

### STRICT FORMAT & RULES:
*   Do not use tables. Format your report using plain text.
*   Maintain an analytical, neutral, and concise tone throughout.
*   I expect you to independently and autonomously evaluate the novelty of the main paper based solely on the information you have retrieved. Therefore, you are prohibited from using OpenReview or any related websites to access existing reviewer evaluations of this article. If relevant content is still retrieved, please disregard all reviewer comments within OpenReview and base your evaluation solely on your own analysis.
*Your output MUST be organized as follows:
## 1. Paper Content Summary
[Single paragraph]

## 2. Point-wise Novelty Analysis
### 2.1. Novelty Point 1: (Classification: [Type])
a) Claimed Novelty:
b) Similarities:
c) Unique Differences:
d) Unique Feature Description:

### 2.2. Novelty Point 2: (Classification: [Type])
...

## 3. Novelty Summary
***
"""

        # Empty response statistics tracking
        self.empty_response_stats = {
            "query_generation": 0,
            "answer_generation": 0,
            "details": []
        }

    def _log(self, message):
        """Helper function for timestamped logging."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def _call_with_retries(self, func, description):
        """Call a function with retry logic on exceptions."""
        last_exc = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                result = func()
                if attempt > 1:
                    self._log(f"[INFO] {description} succeeded on attempt {attempt}.")
                return True, result
            except Exception as e:
                last_exc = e
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY * attempt
                    self._log(f"[WARNING] {description} attempt {attempt}/{self.MAX_RETRIES} "
                              f"failed: {e} -> retrying in {delay}s")
                    time.sleep(delay)
                else:
                    self._log(f"[CRITICAL ERROR] {description} failed after "
                              f"{self.MAX_RETRIES} attempts: {e}")
        return False, last_exc

    def _call_llm_with_retries(self, func, description, validator, fail_type, context):
        """
        Call an LLM function with retry logic, including response content validation.

        Args:
            func: Callable that returns an OpenAI response object.
            validator: Callable(content_str) -> bool (True means acceptable).
            fail_type: "query_generation" or "answer_generation" for stats tracking.
            context: Dict with additional context info (paper, dimension, group, etc.).
        """
        last_content = ""
        last_exc = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = func()
                content = resp.choices[0].message.content if resp and resp.choices else ""
                last_content = content
                if content is None:
                    content = ""
                stripped = (content or "").strip()
                if validator(stripped):
                    if attempt > 1:
                        self._log(f"[INFO] {description} succeeded on attempt {attempt} "
                                  f"(non-empty response).")
                    return True, stripped, resp
                else:
                    if attempt < self.MAX_RETRIES:
                        delay = self.BASE_RETRY_DELAY * attempt
                        self._log(f"[WARNING] {description} attempt {attempt}/{self.MAX_RETRIES} "
                                  f"returned empty/invalid response -> retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        self._log(f"[CRITICAL ERROR] {description} returned empty/invalid "
                                  f"response after {self.MAX_RETRIES} attempts. Giving up.")
            except Exception as e:
                last_exc = e
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY * attempt
                    self._log(f"[WARNING] {description} attempt {attempt}/{self.MAX_RETRIES} "
                              f"exception: {e} -> retrying in {delay}s")
                    time.sleep(delay)
                else:
                    self._log(f"[CRITICAL ERROR] {description} failed after "
                              f"{self.MAX_RETRIES} attempts: {e}")
                    break

        # Record failure statistics
        self.empty_response_stats[fail_type] += 1
        detail = {
            "type": fail_type,
            "attempts": self.MAX_RETRIES,
            "content_length": len(last_content.strip()) if last_content else 0
        }
        detail.update(context or {})
        self.empty_response_stats["details"].append(detail)

        if last_exc:
            return False, f"[EMPTY_OR_ERROR_AFTER_{self.MAX_RETRIES}] last_exception={last_exc}", None
        else:
            return False, f"[EMPTY_AFTER_{self.MAX_RETRIES}] content='{last_content or ''}'", None

    def extract_report_content(self, file_path):
        """
        Extract the main report content from a report file.
        Attempts to find content between 'AI RESPONSE' and 'End of Report' markers.
        Falls back to full file content if markers are not found.

        Args:
            file_path: Path to the report file.

        Returns:
            Extracted report content as a string.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Report file not found: {file_path}")

        self._log(f"[DEBUG-EXTRACT] Starting extraction for file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except UnicodeDecodeError as e:
            self._log(f"[ERROR-EXTRACT] Encoding issue: {e}. Trying fallback to utf-8-sig.")
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                raw_content = f.read()

        raw_length = len(raw_content)
        if raw_length == 0:
            self._log("[WARNING-EXTRACT] Raw content is empty!")

        extracted = None

        # Try to find the AI RESPONSE block
        start_pattern = r'AI RESPONSE:\s*\n=+\s*\n'
        start_match = re.search(start_pattern, raw_content)

        if start_match:
            start_pos = start_match.end()
            end_pattern = r'\n=+\s*\nEnd of Report\s*\n=+'
            end_match = re.search(end_pattern, raw_content[start_pos:], flags=re.IGNORECASE)
            if end_match:
                end_pos = start_pos + end_match.start()
                extracted = raw_content[start_pos:end_pos].strip()
            else:
                self._log("[WARNING-EXTRACT] Found AI RESPONSE but no End of Report marker. "
                          "Using content to end of file.")
                extracted = raw_content[start_pos:].strip()
        else:
            # Try alternative marker
            marker_match = re.search(r'\*\*REPORT CONTENT:\*\*(.*)', raw_content, re.DOTALL)
            if marker_match:
                extracted = marker_match.group(1).strip()
            else:
                self._log("[INFO-EXTRACT] No AI RESPONSE block or **REPORT CONTENT:** marker found. "
                          "Using full file content.")
                extracted = raw_content.strip()

        # Remove trailing End of Report marker if present
        extracted = re.sub(r'(?:^|\n)End of Report\s*$', '', extracted or '',
                           flags=re.IGNORECASE).strip()

        final_length = len(extracted)
        self._log(f"[DEBUG-EXTRACT] Final extracted content length: {final_length}")

        if final_length == 0:
            self._log("[CRITICAL-EXTRACT] Final extracted content is empty!")

        return extracted

    def infer_dataset_name(self, paper_name, db_type="ALL"):
        """
        Infer the RAGFlow dataset name from the paper name and database type.

        Args:
            paper_name: Name of the paper.
            db_type: "Main" or "ALL" to determine dataset suffix.

        Returns:
            Inferred dataset name string.
        """
        suffix = paper_name[:19] if len(paper_name) > 0 else "Unnamed"
        base = f"_{suffix}"
        if db_type == "Main":
            return f"{base}_main_only_dataset"
        return f"{base}_dataset"

    def generate_rag_queries(self, group_name, questions, dimension_name,
                             dimension_description, dimension_conditions,
                             report_content, paper_name):
        """
        Generate RAG queries using the LLM based on the report content and evaluation criteria.

        Returns:
            Tuple of (queries_response_text, query_generation_prompt).
        """
        max_length = 40000
        truncated_content = (report_content[:max_length] + "..."
                             if len(report_content) > max_length else report_content)
        formatted_questions = "\n".join([f"- {q}" for q in questions])

        query_generation_prompt = f"""
You are a report evaluation expert. Based on the report content and evaluation criteria provided, generate exactly 6 query statements for the RAG (retrieval-augmented generation) system. These queries will retrieve relevant information from the database to assist in evaluating the report.

**EVALUATION DIMENSION:** {dimension_name}

**EVALUATION CRITERIA:**
{dimension_description}

**EVALUATION CONDITIONS:**
{dimension_conditions}

**FILTERED QUESTIONS TO EVALUATE:**
{formatted_questions}

**REPORT CONTENT:**
{truncated_content}

---

**HOW TO GENERATE QUERIES:**

**Step 1: Understand what needs to be verified**
Read the evaluation questions and identify what aspects of the report need to be checked against the database.

**Step 2: Locate relevant content in the report**
Find the specific sections, claims, methods, or statements in the report that relate to these evaluation questions.

**Step 3: Extract verifiable technical content**
From those sections, extract specific technical claims, novelty points, methods, baselines, or comparisons that need verification.

**Step 4: Convert to searchable terms**
Transform the extracted content into concrete search keywords that can retrieve relevant papers from the database.

---

**CONCRETE EXAMPLE:**

If the question asks: "Are the claimed unique differences actually unique?"
- Find the "Unique Differences" section in the report
- Identify what is claimed as unique (e.g., "graph-informed ODE dynamics conditioned on adjacency matrix")
- Extract key technical terms: "graph ODE adjacency matrix dynamics"
- Create query: "graph ODE adjacency matrix conditioning"
- **Purpose:** Retrieved papers will show if similar methods exist, verifying the uniqueness claim

If the question asks: "Does the report accurately describe baselines?"
- Find where baselines are mentioned (e.g., "Latent ODEs for Irregular Time Series uses static latent vectors")
- Extract baseline name and key features: "Latent ODEs static latent vector"
- Create query: "Latent ODEs Irregular Time Series static vector"
- **Purpose:** Retrieved papers will contain actual baseline descriptions for comparison

---

**QUERY REQUIREMENTS:**

DO:
- Extract specific technical terms from the report that relate to evaluation questions
- Use exact method names, algorithm names, novelty points mentioned in the report
- Use exact dataset names, benchmark names, baseline model names from the report
- Focus on claims that need verification (uniqueness, accuracy, completeness)
- Combine terms that describe verifiable technical concepts

AVOID:
- Meta-language: "related work", "comparison", "analysis", "the report", "this paper", "evaluation"
- Question words: "how", "what", "why", "whether"
- Pronouns: "this", "it", "their", "current"
- Generic phrases: "methods for", "approaches to", "techniques in"
- Abstract terms: "novel", "improved", "better"
- Do not generate any unnecessary content beyond the six required queries, such as explanations or reasons

---

**QUERY STRUCTURE:**

- **First 4 queries:** Extract 2-4 core technical keywords/phrases directly from the report
  - Examples: method names, algorithm names, benchmark names, technical concepts, novelty claims
  - Format: "keyword1 keyword2 keyword3"

- **Last 2 queries:** Create descriptive phrases about specific technical aspects mentioned in the report
  - Examples: "CVQA benchmark evaluation metrics and performance results", "graph neural ODE adjacency matrix temporal dynamics"
  - Format: Combine 4-6 specific terms into a technical phrase

---

**OUTPUT FORMAT (exactly 6 numbered lines, no other text):**
1. keyword1 keyword2 keyword3
2. keyword1 keyword2
3. keyword1 keyword2 keyword3 keyword4
4. keyword1 keyword2 keyword3 keyword4
5. specific technical aspect description from the report
6. another specific technical aspect description from the report
"""

        self._log("      [DEBUG] Attempting to generate RAG queries via OpenAI API...")

        def _openai_query():
            return self.openai_client.chat.completions.create(
                model=self.config['models']['query_generation_model'],
                messages=[{"role": "user", "content": query_generation_prompt}],
                temperature=0.0
            )

        # Validator: non-empty with minimal length
        def _validator(s):
            return bool(s) and len(s) > 5

        success, content_or_err, resp_obj = self._call_llm_with_retries(
            _openai_query,
            "OpenAI RAG query generation",
            _validator,
            "query_generation",
            {"paper": paper_name, "dimension": dimension_name, "group": group_name}
        )

        if success:
            self._log("      [DEBUG] Successfully generated RAG queries (non-empty).")
            return content_or_err.strip(), query_generation_prompt
        else:
            self._log(f"      [ERROR] Failed to generate RAG queries. Details: {content_or_err}")
            return "", query_generation_prompt

    def extract_queries_from_response(self, queries_response):
        """
        Extract individual queries from the LLM-generated query response.

        Args:
            queries_response: Raw LLM response text containing numbered queries.

        Returns:
            List of query strings (up to 6).
        """
        if not queries_response:
            return []
        query_pattern = r'\d+\.\s*(.+?)(?=\n\d+\.|$)'
        queries = re.findall(query_pattern, queries_response, re.DOTALL)
        cleaned_queries = [q.strip().replace('\n', ' ') for q in queries if q.strip()]
        return cleaned_queries[:6]

    def retrieve_from_rag(self, query, dataset_name):
        """
        Retrieve relevant chunks from RAGFlow for a given query and dataset.

        Args:
            query: The search query string.
            dataset_name: Name of the RAGFlow dataset to search.

        Returns:
            List of chunk records with content and source metadata.
        """
        if not self.rag_object:
            raise ValueError("RAG client is not initialized.")

        self._log(f"        [DEBUG] Retrieving from RAGFlow dataset: '{dataset_name}'")

        def _do_retrieve():
            datasets = self.rag_object.list_datasets(name=dataset_name)
            if not datasets:
                raise RuntimeError(f"RAGFlow dataset not found: {dataset_name}")
            dataset = datasets[0]
            self._log(f"        [DEBUG] Found dataset ID: {dataset.id}. "
                      f"Retrieving for query: '{query[:50]}...'")

            chunks_per_query = self.config["rag"].get('chunks_per_query', 10)

            results = list(self.rag_object.retrieve(
                question=query,
                dataset_ids=[dataset.id],
                document_ids=None,
                page=self.config["rag"]['page'],
                page_size=chunks_per_query,
                similarity_threshold=self.config["rag"]['similarity_threshold'],
                vector_similarity_weight=self.config["rag"]['vector_similarity_weight'],
                top_k=self.config["rag"]['top_k'],
                rerank_id=self.config["rag"]['rerank_id'],
                keyword=self.config["rag"]['keyword']
            ))
            chunk_records = []
            for rank, res in enumerate(results, 1):
                content = getattr(res, "content", "") or ""

                metadata = getattr(res, "metadata", None)
                if metadata is None:
                    metadata = {}
                elif not isinstance(metadata, dict):
                    metadata = getattr(metadata, "__dict__", {}) or {}

                source_parts = []
                for attr in (
                    "document_name", "doc_name", "document_id", "doc_id",
                    "dataset_name", "page", "page_number", "position"
                ):
                    value = getattr(res, attr, None)
                    if value is None and metadata:
                        value = metadata.get(attr)
                    if value not in (None, ""):
                        source_parts.append(f"{attr}={value}")

                source = "; ".join(source_parts) if source_parts else f"rank={rank}"
                chunk_records.append({
                    "source": source,
                    "rank": rank,
                    "content": content
                })
            return chunk_records

        success, results_or_exc = self._call_with_retries(
            _do_retrieve, f"RAGFlow retrieval (dataset={dataset_name})")
        if success:
            self._log(f"        [DEBUG] RAGFlow retrieval successful, "
                      f"found {len(results_or_exc)} chunks.")
            return results_or_exc
        else:
            self._log(f"        [ERROR] RAGFlow retrieval failed: {results_or_exc}")
            return []

    def conduct_individual_rag_retrieval(self, queries, dataset_name):
        """
        Execute RAG retrieval for each query individually.

        Args:
            queries: List of query strings.
            dataset_name: Name of the RAGFlow dataset.

        Returns:
            Dict mapping each query to its list of retrieved chunk records.
        """
        rag_results = {}
        for i, query in enumerate(queries, 1):
            self._log(f"      - Processing RAG query {i}/{len(queries)}...")
            rag_results[query] = self.retrieve_from_rag(query, dataset_name)
            time.sleep(0.3)
        return rag_results

    def combine_rag_results(self, rag_results):
        """
        Combine and deduplicate RAG retrieval results from multiple queries.

        Args:
            rag_results: Dict mapping queries to lists of chunk records.

        Returns:
            Combined and deduplicated string of all chunk contents.
        """
        total_chunks_before = 0
        all_chunks = {}

        for query, chunks in rag_results.items():
            for chunk in chunks:
                if chunk:
                    if isinstance(chunk, dict):
                        content = chunk.get("content", "")
                        source = chunk.get("source", "unknown")
                        rank = chunk.get("rank", "")
                    else:
                        content = str(chunk)
                        source = "unknown"
                        rank = ""

                    if not content:
                        continue

                    key = (source, content)
                    all_chunks[key] = {
                        "source": source,
                        "rank": rank,
                        "content": content
                    }
                    total_chunks_before += 1

        unique_chunks = list(all_chunks.values())
        total_chunks_after = len(unique_chunks)
        duplicates_removed = total_chunks_before - total_chunks_after
        self._log(f"      [DEDUP] Before: {total_chunks_before} chunks, "
                  f"After: {total_chunks_after}, Duplicates removed: {duplicates_removed}")

        # Sort for deterministic output
        unique_list = sorted(unique_chunks, key=lambda item: (item["source"], item["content"]))
        combined = "".join([
            f"Source: {item['source']}\nContent: {item['content']}\n\n"
            for item in unique_list
        ])
        return combined

    def get_rag_knowledge(self, paper_name, group_name, questions, dimension_name,
                          dimension_description, dimension_conditions,
                          report_content, db_type):
        """
        Retrieve RAG knowledge for evaluating a specific question group.

        Returns:
            Tuple of (combined_knowledge_string, rag_debug_info_dict).
        """
        dataset_name = self.infer_dataset_name(paper_name, db_type)
        self._log(f"      - Inferred dataset name: {dataset_name}")

        queries_response, query_generation_prompt = self.generate_rag_queries(
            group_name, questions, dimension_name, dimension_description,
            dimension_conditions, report_content, paper_name
        )
        queries = self.extract_queries_from_response(queries_response)

        if not queries:
            self._log("      - [WARNING] Could not extract any RAG queries from the API response.")
            return "", {}

        self._log(f"      - Extracted {len(queries)} queries. Starting retrieval process.")
        rag_results = self.conduct_individual_rag_retrieval(queries, dataset_name)
        combined_knowledge = self.combine_rag_results(rag_results)

        rag_debug_info = {
            "dataset_name": dataset_name,
            "db_type": db_type,
            "query_generation_prompt": query_generation_prompt,
            "queries_response": queries_response,
            "extracted_queries": queries,
            "rag_results": rag_results,
            "combined_knowledge": combined_knowledge
        }
        return combined_knowledge, rag_debug_info

    def answer_questions(self, article, summary, questions, dimension,
                         definition, conditions, paper_name, group_name):
        """
        Use the LLM to answer evaluation questions based on the report and database content.

        Returns:
            Tuple of (answers_text, prompt_text, success_bool).
        """
        sum_length = len(summary) if summary else 0
        if sum_length == 0:
            self._log("[WARNING-ANSWER] Incoming summary (report content) is empty!")

        # Build the questions string outside the f-string for compatibility
        questions_formatted = '\n'.join([f"Q{i+1}: {q}" for i, q in enumerate(questions)])

        if "Main" in group_name:
            focus_note = (
                "This question group focuses on the parts of the report that summarize or describe the main paper. "
                "When judging completeness or faithfulness, prioritize whether those report statements match the "
                "main-paper evidence in the database."
            )
        elif "ALL" in group_name:
            focus_note = (
                "This question group focuses on the parts of the report that compare the main paper with retrieved "
                "or cited related works. When judging completeness or faithfulness, prioritize whether the report's "
                "similarity, difference, originality, and prior-work descriptions are supported by the related-work "
                "evidence in the database."
            )
        else:
            focus_note = (
                "This question group should be judged primarily from the report itself and the provided task/template "
                "context. In these groups, phrases such as 'provided text', 'retrieved works', or 'available evidence' "
                "refer to the evidence presented, cited, or summarized inside the report unless an external database "
                "is explicitly provided."
            )

        prompt = f"""
<Task Overview>
Your task is to read a provided report and a database that contains information on related papers, then answer 'yes'
or 'no' to specific questions. These questions will relate to a particular
dimension of the report.
<dimension Definition>
{dimension}- {definition}
<dimension Conditions>
{conditions}
<Instructions>
1. Read these instructions thoroughly.
2. Carefully read both the Report and the database.
3. Understand the given questions and the definition of the <dimension>.
4. Respond to each question with 'yes' or 'no'. Base your answers on a clear
rationale.
5. Follow the specified format for your answers.
<Evaluation Focus Note>
The report may contain different kinds of content: some statements summarize the main paper, while other statements
compare it with retrieved or cited related works. The exact naming of a group is less important than its evaluation
focus. Use the following focus for this question group:
{focus_note}
<Equivalent Naming Note>
When evaluating structure or section-specific questions, judge flexibly according to the actual function and meaning
of the section/subsection. If two titles express nearly the same function but use different wording, treat them as
equivalent and do not answer the checklist question as "No" merely because of the wording difference. For example, if
a question asks whether "Claimed Novelty" exists and the report uses "Claimed Innovation", treat that as satisfying
the requested section title. The following titles should be treated as equivalent:
- "Comparative Analysis", "Point-wise Novelty Analysis", and "Point-wise Comparative Analysis" are equivalent titles
  for the second main section.
- "Innovation Point" and "Novelty Point" are equivalent titles for each point-level subsection.
- "Claimed Innovation" and "Claimed Novelty" are equivalent subsection titles.
- "Details of Unique Difference", "Details of Unique Differences", and "Unique Feature Description" are equivalent
  subsection titles.
- "Innovation Summary" and "Novelty Summary" are equivalent titles for the final main summary section.
- "Genuine Innovations" and "Genuine Novelties" are equivalent subsection titles.
- "Aspects Lacking Originality", "Aspects Lacking Substantial Originality", and "Aspects lacking substantial
  originality" are equivalent subsection titles.
- "References" and markdown reference-link definitions are acceptable citation-support sections and should not be
  treated as extra analytical sections.
<Answer Format>
Q1: [Your Answer]
Q2: [Your Answer]
...
# Database #
{article}
# Report #
{summary}
# Questions #
{questions_formatted}
# Response #
Provide your answers to the given questions, following the specified Answer Format.
"""
        self._log("      [DEBUG] Attempting to answer questions via OpenAI API...")

        def _openai_answer():
            return self.openai_client.chat.completions.create(
                model=self.config['models']['evaluation_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

        # Validator: must contain one yes/no answer for every question in the group
        def _validator(s):
            if not s:
                return False
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            answered = {}
            pattern = re.compile(r'^Q(\d+)\s*:\s*(yes|no)\b', re.IGNORECASE)
            for line in lines:
                match = pattern.match(line)
                if match:
                    answered[int(match.group(1))] = match.group(2).lower()
            expected = set(range(1, len(questions) + 1))
            return expected.issubset(set(answered.keys()))

        success, content_or_err, resp_obj = self._call_llm_with_retries(
            _openai_answer,
            "OpenAI question answering",
            _validator,
            "answer_generation",
            {"paper": paper_name, "dimension": dimension, "group": group_name}
        )

        if success:
            self._log("      [DEBUG] Successfully answered questions (non-empty with yes/no).")
            return content_or_err.strip(), prompt, True
        else:
            self._log(f"      [ERROR] Failed to answer questions. Details: {content_or_err}")
            return content_or_err, prompt, False

    def evaluate_dimension(self, dimension_name, report_content):
        """
        Evaluate a single dimension for the loaded report.

        Args:
            dimension_name: Name of the dimension to evaluate.
            report_content: Pre-extracted report content string.

        Returns:
            Tuple of (dimension_results_dict, dimension_rag_debug_dict, success_bool).
        """
        dimension = self.dimensions[dimension_name]

        rc_length = len(report_content) if report_content else 0
        if rc_length == 0:
            self._log("[WARNING-EVAL_DIM] Report content is empty!")

        dimension_results = {}
        dimension_rag_debug = {}
        dimension_failed = False

        for group_name, questions in dimension['filtered_questions'].items():
            self._log(f"    - Processing group: {group_name}")
            rag_debug_info = {}
            dimension_conditions = dimension.get('conditions', '')

            if 'Main' in group_name or 'ALL' in group_name:
                db_type = "Main" if 'Main' in group_name else "ALL"
                article, rag_debug_info = self.get_rag_knowledge(
                    self.paper_name, group_name, questions, dimension_name,
                    dimension['description'], dimension_conditions,
                    report_content, db_type
                )
            else:
                article = self.default_article

            answers, answer_prompt, ok = self.answer_questions(
                article, report_content, questions, dimension_name,
                dimension['description'], dimension_conditions,
                self.paper_name, group_name
            )

            if not ok:
                dimension_failed = True
                self._log(f"[WARNING] Group '{group_name}' in dimension '{dimension_name}' "
                          f"produced empty/invalid answers. Marking dimension as unsaved.")

            dimension_results[group_name] = {
                "answers": answers,
                "answer_prompt": answer_prompt,
                "valid": ok
            }

            if rag_debug_info:
                dimension_rag_debug[group_name] = rag_debug_info

        return dimension_results, dimension_rag_debug, not dimension_failed

    def evaluate_report(self):
        """Main entry point: evaluate the loaded report across all dimensions."""
        self._log(f"Starting evaluation for paper: {self.paper_name}")
        self._log(f"Report path: {self.report_path}")

        # Create output directory for this paper
        paper_output_dir = os.path.join(self.output_base_dir, self.paper_name)
        os.makedirs(paper_output_dir, exist_ok=True)

        results_path = os.path.join(paper_output_dir, "evaluation_results.json")
        debug_path = os.path.join(paper_output_dir, "rag_debug_details.json")

        # Load existing results if present (for incremental evaluation)
        all_paper_evaluations = {}
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    all_paper_evaluations = json.load(f)
            except json.JSONDecodeError:
                self._log(f"  [WARNING] Results file {results_path} is corrupted. Starting fresh.")
                all_paper_evaluations = {}

        all_paper_rag_debug = {}
        if os.path.exists(debug_path):
            try:
                with open(debug_path, 'r', encoding='utf-8') as f:
                    all_paper_rag_debug = json.load(f)
            except json.JSONDecodeError:
                all_paper_rag_debug = {}

        # Extract report content once for all dimensions
        report_content = self.extract_report_content(self.report_path)
        if not report_content:
            self._log("[CRITICAL] Report content is empty. Aborting evaluation.")
            return

        unsaved_dimensions = []

        # Determine which dimensions to run
        if self.rerun_dimensions:
            dims_to_run = self.rerun_dimensions
            self._log(f"[RERUN MODE] Only re-running dimensions: "
                      f"{', '.join(self.rerun_dimensions)}")
        else:
            dims_to_run = list(self.dimensions.keys())
            self._log("[DEFAULT MODE] Evaluating all dimensions.")

        for dimension_name in dims_to_run:
            # Skip already-evaluated dimensions in default mode
            if (not self.rerun_dimensions) and (dimension_name in all_paper_evaluations):
                self._log(f"  Dimension '{dimension_name}' already evaluated. Skipping.")
                continue

            # In rerun mode, remove old results for this dimension before re-evaluation
            if self.rerun_dimensions and dimension_name in all_paper_evaluations:
                self._log(f"  [RERUN] Overwriting dimension '{dimension_name}' "
                          f"for paper '{self.paper_name}'.")
                all_paper_evaluations.pop(dimension_name, None)
                all_paper_rag_debug.pop(dimension_name, None)

            self._log(f"  Evaluating dimension: {dimension_name} (Paper: {self.paper_name})")
            try:
                dimension_results, dimension_rag_debug, dimension_ok = \
                    self.evaluate_dimension(dimension_name, report_content)
            except Exception as e:
                self._log(f"  [CRITICAL ERROR] evaluate_dimension exception for "
                          f"'{dimension_name}': {e}")
                self._log("  Skipping this dimension and continuing with the next.")
                continue

            if dimension_ok:
                all_paper_evaluations[dimension_name] = dimension_results
                if dimension_rag_debug:
                    all_paper_rag_debug[dimension_name] = dimension_rag_debug

                # Save results incrementally after each successful dimension
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(all_paper_evaluations, f, ensure_ascii=False, indent=4)
                self._log(f"  > Saved results for dimension '{dimension_name}' to: {results_path}")

                if all_paper_rag_debug:
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        json.dump(all_paper_rag_debug, f, ensure_ascii=False, indent=4)
            else:
                unsaved_dimensions.append(dimension_name)
                self._log(f"  [WARNING] Dimension '{dimension_name}' had empty response groups. "
                          f"Not saved. Will retry on next run.")

        if unsaved_dimensions:
            self._log(f"[SUMMARY] Unsaved dimensions for paper '{self.paper_name}': "
                      f"{unsaved_dimensions}")

        self._log(f"Finished evaluation for paper: {self.paper_name}")
        self._log(f"Results saved to: {paper_output_dir}")
        print("-" * 40)
        self._print_empty_stats()

    def _print_empty_stats(self):
        """Print statistics about empty/invalid LLM responses."""
        total = (self.empty_response_stats["query_generation"] +
                 self.empty_response_stats["answer_generation"])
        if total == 0:
            self._log("[EMPTY_STATS] No empty responses encountered.")
            return
        self._log("[EMPTY_STATS] Empty/invalid response statistics:")
        self._log(f"  - Query Generation failures: "
                  f"{self.empty_response_stats['query_generation']}")
        self._log(f"  - Answer Generation failures: "
                  f"{self.empty_response_stats['answer_generation']}")
        # Show the last 10 failure details
        for d in self.empty_response_stats["details"][-10:]:
            self._log(f"    * {d}")


def load_config(path):
    """Load the configuration from a JSON file."""
    resolved = resolve_path(path)
    try:
        with open(resolved, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[FATAL ERROR] Configuration file not found at: {resolved}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[FATAL ERROR] Could not parse configuration file. "
              f"Check for JSON syntax errors in: {resolved}")
        sys.exit(1)


def _normalize_list(value):
    """Normalize config values that may be a string, list, tuple, or empty."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    text = str(value).strip()
    return [text] if text else []


def collect_report_paths(config, cli_report_paths=None, cli_report_dirs=None):
    """
    Collect report files from config and CLI inputs.

    Supported inputs:
    - paths.report_path: a single report file
    - paths.report_paths: a list of report files
    - paths.report_dir / paths.report_dirs / paths.reports_dir: one or more directories
    - CLI --report_path and --report_dir, each repeatable
    """
    paths_cfg = config.get("paths", {})
    candidate_files = []
    candidate_dirs = []

    candidate_files.extend(_normalize_list(paths_cfg.get("report_path")))
    candidate_files.extend(_normalize_list(paths_cfg.get("report_paths")))
    candidate_files.extend(_normalize_list(cli_report_paths))

    candidate_dirs.extend(_normalize_list(paths_cfg.get("report_dir")))
    candidate_dirs.extend(_normalize_list(paths_cfg.get("report_dirs")))
    candidate_dirs.extend(_normalize_list(paths_cfg.get("reports_dir")))
    candidate_dirs.extend(_normalize_list(cli_report_dirs))

    extensions = paths_cfg.get("report_extensions", [".txt", ".md", ".markdown"])
    extensions = {str(ext).lower() for ext in _normalize_list(extensions)}
    if not extensions:
        extensions = {".txt", ".md", ".markdown"}

    recursive = bool(paths_cfg.get("recursive", True))

    collected = []

    for file_path in candidate_files:
        resolved = resolve_path(file_path)
        if os.path.isfile(resolved):
            collected.append(os.path.abspath(resolved))
        elif file_path:
            print(f"[WARNING] Report file not found, skipping: {resolved}")

    for dir_path in candidate_dirs:
        resolved_dir = resolve_path(dir_path)
        if not os.path.isdir(resolved_dir):
            print(f"[WARNING] Report directory not found, skipping: {resolved_dir}")
            continue

        if recursive:
            walker = os.walk(resolved_dir)
            for root, _, files in walker:
                for filename in sorted(files):
                    if os.path.splitext(filename)[1].lower() in extensions:
                        collected.append(os.path.abspath(os.path.join(root, filename)))
        else:
            for filename in sorted(os.listdir(resolved_dir)):
                full_path = os.path.join(resolved_dir, filename)
                if os.path.isfile(full_path) and os.path.splitext(filename)[1].lower() in extensions:
                    collected.append(os.path.abspath(full_path))

    # Deduplicate while preserving input order.
    deduped = []
    seen = set()
    for path in collected:
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(path)

    return deduped


def warn_duplicate_output_names(report_paths, job_config=None):
    """Warn if multiple input files would write to the same current output folder."""
    output_names = {}
    for path in report_paths:
        paper_name = os.path.splitext(os.path.basename(path))[0]
        output_names.setdefault(paper_name, []).append(path)

    duplicates = {name: paths for name, paths in output_names.items() if len(paths) > 1}
    if not duplicates:
        return

    print("[WARNING] Multiple report files share the same basename. Current output logic "
          "uses the basename as the result folder, so these files will share one output folder:")
    for name, paths in duplicates.items():
        print(f"  - {name}:")
        for path in paths:
            print(f"      {path}")


def get_max_workers(config, job=None, cli_max_workers=None):
    """Resolve thread count from CLI, job config, root config, or default."""
    raw_value = None
    if cli_max_workers is not None:
        raw_value = cli_max_workers
    elif job and isinstance(job, dict) and job.get("max_workers") is not None:
        raw_value = job.get("max_workers")
    elif config.get("max_workers") is not None:
        raw_value = config.get("max_workers")
    elif config.get("parallel", {}).get("max_workers") is not None:
        raw_value = config.get("parallel", {}).get("max_workers")
    else:
        raw_value = 50

    try:
        max_workers = int(raw_value)
    except (TypeError, ValueError):
        print(f"[WARNING] Invalid max_workers={raw_value!r}; falling back to 50.")
        max_workers = 50

    return max(1, max_workers)


def build_evaluation_jobs(config, cli_report_paths=None, cli_report_dirs=None,
                          cli_output_base_dir=None, cli_max_workers=None):
    """
    Build one or more evaluation jobs.

    Backward-compatible mode:
    - If no evaluation_jobs are configured, use the root paths config.

    Paired-input/output mode:
    - Configure evaluation_jobs as a list. Each item can define report_path,
      report_paths, report_dir/report_dirs, output_base_dir, recursive,
      report_extensions, output_name_mode, report_manifest_path, and model
      overrides. Each job writes to its own output_base_dir.
    """
    cli_has_inputs = bool(_normalize_list(cli_report_paths) or _normalize_list(cli_report_dirs))
    configured_jobs = config.get("evaluation_jobs") or config.get("paths", {}).get("evaluation_jobs")

    # CLI inputs are treated as one ad-hoc job so command-line usage stays predictable.
    if cli_has_inputs:
        job_config = copy.deepcopy(config)
        if cli_output_base_dir:
            job_config.setdefault("paths", {})["output_base_dir"] = cli_output_base_dir
        report_paths = collect_report_paths(
            job_config,
            cli_report_paths=cli_report_paths,
            cli_report_dirs=cli_report_dirs
        )
        return [{
            "name": "cli",
            "max_workers": get_max_workers(job_config, cli_max_workers=cli_max_workers),
            "config": job_config,
            "report_paths": report_paths,
        }]

    if configured_jobs:
        jobs = []
        for idx, job in enumerate(configured_jobs, 1):
            if not isinstance(job, dict):
                print(f"[WARNING] evaluation_jobs[{idx}] is not an object. Skipping.")
                continue

            job_config = copy.deepcopy(config)
            job_paths = job_config.setdefault("paths", {})

            # Clear root inputs so each job is controlled only by its own paired input config.
            for key in ("report_path", "report_paths", "report_dir", "report_dirs", "reports_dir"):
                job_paths.pop(key, None)

            for key in (
                "report_path", "report_paths", "report_dir", "report_dirs", "reports_dir",
                "output_base_dir", "report_extensions", "recursive",
                "output_name_mode", "paper_name_mode", "paper_title_mode",
                "report_name_root", "report_name_roots",
                "report_manifest_path", "dataset_manifest_path", "manifest_path"
            ):
                if key in job:
                    job_paths[key] = job[key]

            if isinstance(job.get("models"), dict):
                job_config.setdefault("models", {}).update(job["models"])
            if job.get("evaluation_model"):
                job_config.setdefault("models", {})["evaluation_model"] = job["evaluation_model"]
            if job.get("query_generation_model"):
                job_config.setdefault("models", {})["query_generation_model"] = job["query_generation_model"]

            job_name = job.get("name") or f"job_{idx}"
            job_config["job_name"] = job_name
            report_paths = collect_report_paths(job_config)
            jobs.append({
                "name": job_name,
                "max_workers": get_max_workers(config, job=job, cli_max_workers=cli_max_workers),
                "config": job_config,
                "report_paths": report_paths,
            })
        return jobs

    root_config = copy.deepcopy(config)
    if cli_output_base_dir:
        root_config.setdefault("paths", {})["output_base_dir"] = cli_output_base_dir
    report_paths = collect_report_paths(root_config)
    return [{
        "name": "default",
        "max_workers": get_max_workers(root_config, cli_max_workers=cli_max_workers),
        "config": root_config,
        "report_paths": report_paths,
    }]


def evaluate_one_report(job_config, report_path):
    """Evaluate a single report and return a small status record."""
    report_config = copy.deepcopy(job_config)
    report_config.setdefault("paths", {})["report_path"] = report_path

    try:
        evaluator = ReportEvaluator(config=report_config)
        evaluator.evaluate_report()
        return {
            "status": "ok",
            "path": report_path,
            "error": "",
        }
    except FileNotFoundError as e:
        return {
            "status": "failed",
            "path": report_path,
            "error": str(e),
        }
    except KeyError as e:
        return {
            "status": "fatal",
            "path": report_path,
            "error": f"A required key is missing from the config file: {e}",
        }
    except Exception as e:
        import traceback
        return {
            "status": "failed",
            "path": report_path,
            "error": f"{e}\n{traceback.format_exc()}",
        }


# --- Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate one or more reports using the Checklist Evaluation framework. "
                    "Report files can be provided through config or CLI arguments."
    )
    parser.add_argument(
        "--config_path", type=str, default="Evaluation.json",
        help="Path to configuration JSON file. Defaults to ./Evaluation.json "
             "in the same directory as this script."
    )
    parser.add_argument(
        "--rerun_dimension", type=str, default=None,
        help="Name of one or more dimensions to re-run, separated by commas if needed "
             "(overwrites previous results for those dimensions). If not specified, "
             "uses rerun_dimensions/rerun_dimension from config or defaults to running "
             "all missing dimensions. Example: --rerun_dimension fluency,depth"
    )
    parser.add_argument(
        "--report_path", action="append", default=None,
        help="Path to a report file. Can be provided multiple times."
    )
    parser.add_argument(
        "--report_dir", action="append", default=None,
        help="Path to a directory containing report files. Can be provided multiple times."
    )
    parser.add_argument(
        "--output_base_dir", type=str, default=None,
        help="Optional output directory override. Defaults to paths.output_base_dir in config."
    )
    parser.add_argument(
        "--max_workers", type=int, default=None,
        help="Maximum number of reports to evaluate in parallel within each job. Defaults to 50."
    )
    args = parser.parse_args()

    print("--- Starting Checklist Evaluation ---")

    config = load_config(args.config_path)

    # Command-line --rerun_dimension overrides config value if provided
    if args.rerun_dimension is not None:
        config["rerun_dimension"] = args.rerun_dimension
        config.pop("rerun_dimensions", None)
    elif "rerun_dimension" not in config:
        config["rerun_dimension"] = None

    jobs = build_evaluation_jobs(
        config,
        cli_report_paths=args.report_path,
        cli_report_dirs=args.report_dir,
        cli_output_base_dir=args.output_base_dir,
        cli_max_workers=args.max_workers
    )

    total_reports = sum(len(job["report_paths"]) for job in jobs)
    if total_reports == 0:
        print("[FATAL ERROR] No report files found. Provide paths.report_path, "
              "paths.report_paths, paths.report_dir/reports_dir in Evaluation.json, "
              "evaluation_jobs, or use --report_path / --report_dir.")
        sys.exit(1)

    print(f"  Config path      : {resolve_path(args.config_path)}")
    print(f"  Jobs found       : {len(jobs)}")
    print(f"  Reports found    : {total_reports}")
    rerun_display = config.get("rerun_dimensions", config.get("rerun_dimension", "None"))
    print(f"  Rerun dimensions : {rerun_display}")

    failed_reports = []
    processed_reports = 0
    for job_idx, job in enumerate(jobs, 1):
        job_name = job["name"]
        job_config = job["config"]
        report_paths = job["report_paths"]
        output_base_dir = resolve_path(job_config["paths"]["output_base_dir"])

        print("\n" + "#" * 80)
        print(f"[JOB {job_idx}/{len(jobs)}] {job_name}")
        print(f"  Output base dir: {output_base_dir}")
        print(f"  Reports        : {len(report_paths)}")
        print(f"  Max workers    : {job['max_workers']}")
        print(f"  Query model    : {job_config.get('models', {}).get('query_generation_model')}")
        print(f"  Eval model     : {job_config.get('models', {}).get('evaluation_model')}")
        print("#" * 80)
        warn_duplicate_output_names(report_paths, job_config)

        max_workers = min(job["max_workers"], len(report_paths)) if report_paths else 1
        future_to_info = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for report_idx, report_path in enumerate(report_paths, 1):
                processed_reports += 1
                print("\n" + "=" * 80)
                print(f"[SUBMIT REPORT {report_idx}/{len(report_paths)} | "
                      f"GLOBAL {processed_reports}/{total_reports}] {report_path}")
                print("=" * 80)
                future = executor.submit(evaluate_one_report, job_config, report_path)
                future_to_info[future] = (report_idx, report_path)

            for future in as_completed(future_to_info):
                report_idx, report_path = future_to_info[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "status": "failed",
                        "path": report_path,
                        "error": f"Unhandled worker exception: {e}",
                    }

                status = result.get("status")
                if status == "ok":
                    print(f"[DONE REPORT {report_idx}/{len(report_paths)}] {report_path}")
                elif status == "fatal":
                    print(f"[FATAL ERROR] {result.get('error')} (report: {report_path})")
                    sys.exit(1)
                else:
                    print(f"[ERROR] Failed report: {report_path}")
                    print(result.get("error", "Unknown error"))
                    failed_reports.append(report_path)

    if failed_reports:
        print("\n[SUMMARY] Some reports failed:")
        for path in failed_reports:
            print(f"  - {path}")
        sys.exit(1)

    print("--- Evaluation Finished ---")
