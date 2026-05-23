"""
build_initial_memory.py
-----------------------
Reads accepted ICLR PDFs, extracts text, truncates after References/Appendix,
calls LLM for title + summary + innovation extraction, and saves one JSON per paper
as an initial memory entry.

Supports:
  - Accepted-paper filtering via OpenReview JSONL metadata
  - LLM-based title extraction during summary generation
  - Meta-review / reviewer comment extraction from OpenReview metadata
  - Final acceptance decision extraction from OpenReview metadata
  - Year-balanced sampling (equal PDFs per year)
  - Multi-threaded processing (default 128 workers)
  - Resume from previous run (year-aware)

Usage:
    python build_initial_memory.py                  # use default config.json
    python build_initial_memory.py --limit 1000     # override max papers
    python build_initial_memory.py --workers 32     # override thread count
"""

import os
from pathlib import Path
import re
import json
import time
import hashlib
import argparse
import traceback
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import openai


# =========================================================================
# Config
# =========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMORY_SCHEMA_VERSION = 3

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.json"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Accepted-paper filtering from OpenReview JSONL
# =========================================================================

def _normalize_whitespace(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _iter_role_payload_pairs(node):
    """Yield (role, payload) pairs from nested OpenReview review structures."""
    if isinstance(node, list):
        if len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], dict):
            yield node[0], node[1]
            return
        for item in node:
            yield from _iter_role_payload_pairs(item)


def _extract_text_from_payload(payload, candidate_keys):
    """Return the first non-empty text found in payload or payload['value'].""" 
    if not isinstance(payload, dict):
        return None

    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested = payload.get("value")
    if isinstance(nested, dict):
        for key in candidate_keys:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _extract_scores(payload):
    """Extract reviewer scores if present and meaningful."""
    if not isinstance(payload, dict):
        return None

    scores = payload.get("scores")
    if not isinstance(scores, dict):
        nested = payload.get("value")
        if isinstance(nested, dict):
            scores = nested.get("scores")

    if not isinstance(scores, dict):
        return None

    cleaned_scores = {k: v for k, v in scores.items() if v is not None}
    return cleaned_scores or None


def _extract_review_comments(entry):
    """
    Extract OpenReview comments needed by the memory:
      1. PC/AC meta-review comments
      2. Main reviewer reviews only (Reviewer entries with scores)
    """
    meta_review = []
    reviewer_reviews = []
    seen_meta = set()
    seen_reviews = set()

    for role, payload in _iter_role_payload_pairs(entry.get("metareview")):
        comment = _extract_text_from_payload(
            payload, ("comment", "review", "metareview", "decision")
        )
        if not comment:
            continue
        normalized_role = _normalize_whitespace(role) or "MetaReview"
        signature = (normalized_role.lower(), _normalize_whitespace(comment).lower())
        if signature in seen_meta:
            continue
        seen_meta.add(signature)
        meta_review.append({
            "role": normalized_role,
            "comment": comment,
        })

    for role, payload in _iter_role_payload_pairs(entry.get("reviews")):
        role_lower = role.lower()
        if "author" in role_lower:
            continue
        if "review" not in role_lower:
            continue

        scores = _extract_scores(payload)
        if scores is None:
            # Reviewer follow-up questions usually do not contain scores.
            continue

        review_text = _extract_text_from_payload(
            payload, ("review", "comment", "summary_of_the_paper", "summary", "question")
        )
        if not review_text:
            continue

        signature = (_normalize_whitespace(review_text).lower(), json.dumps(scores, sort_keys=True))
        if signature in seen_reviews:
            continue
        seen_reviews.add(signature)

        reviewer_reviews.append({
            "reviewer_index": len(reviewer_reviews) + 1,
            "role": _normalize_whitespace(role) or "Reviewer",
            "scores": scores,
            "review": review_text,
        })

    return {
        "meta_review": meta_review,
        "reviewer_reviews": reviewer_reviews,
    }


def _build_paper_record(entry):
    content = entry.get("content") or {}
    official_title = _normalize_whitespace(content.get("title"))
    final_decision = _normalize_whitespace(entry.get("decision"))
    return {
        "official_title": official_title or None,
        "final_decision": final_decision or None,
        "review_comments": _extract_review_comments(entry),
    }


def load_accepted_papers(jsonl_paths, venue_prefix="ICLR"):
    """
    Read one or more JSONL files and return a set of PDF filenames
    that were accepted (decision starts with "Accept").

    Also prints per-year statistics of accepted vs total papers.

    Args:
        jsonl_paths: list of paths to JSONL files (train.jsonl, test.jsonl)
        venue_prefix: only consider entries whose PDF filename starts with this

    Returns:
        accepted_filenames: set of str (e.g. {"ICLR2017_xxx.pdf", ...})
        paper_records_by_filename: dict mapping PDF filename -> auxiliary metadata
    """
    accepted_filenames = set()
    paper_records_by_filename = {}
    total_by_year = defaultdict(int)
    accepted_by_year = defaultdict(int)
    decision_counts = defaultdict(int)

    for jsonl_path in jsonl_paths:
        if not os.path.exists(jsonl_path):
            print(f"[WARN] JSONL file not found: {jsonl_path}")
            continue

        print(f"[INFO] Loading metadata from: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                pdf_path = entry.get("PDF_path") or ""
                if not pdf_path:
                    continue
                pdf_filename = os.path.basename(pdf_path)

                # Only consider papers matching our venue prefix
                if not pdf_filename.upper().startswith(venue_prefix.upper()):
                    continue

                # Extract year from filename
                year = _extract_year_from_filename(pdf_filename)
                total_by_year[year] += 1

                decision = entry.get("decision", "")
                decision_counts[decision] += 1

                # Check if accepted
                if decision.lower().startswith("accept"):
                    accepted_by_year[year] += 1
                    accepted_filenames.add(pdf_filename)
                    paper_records_by_filename[pdf_filename] = _build_paper_record(entry)

    # Print statistics
    total_all = sum(total_by_year.values())
    accepted_all = len(accepted_filenames)

    print(f"\n[INFO] === {venue_prefix} Paper Statistics from JSONL ===")
    print(f"[INFO] Total {venue_prefix} entries in JSONL: {total_all}")
    print(f"[INFO] Total accepted: {accepted_all}")
    print(f"[INFO] Acceptance rate: {accepted_all/total_all*100:.1f}%" if total_all > 0 else "")

    print(f"\n[INFO] Decision distribution:")
    for decision, count in sorted(decision_counts.items(), key=lambda x: -x[1]):
        marker = " <-- accepted" if decision.lower().startswith("accept") else ""
        print(f"  {count:>6}  {decision}{marker}")

    print(f"\n[INFO] Per-year breakdown (accepted / total):")
    all_years = sorted(set(list(total_by_year.keys()) + list(accepted_by_year.keys())),
                       key=lambda x: (x is None, x))
    for y in all_years:
        label = str(y) if y is not None else "unknown"
        t = total_by_year.get(y, 0)
        a = accepted_by_year.get(y, 0)
        rate = f"{a/t*100:.0f}%" if t > 0 else "N/A"
        print(f"  {label}: {a} / {t} ({rate})")

    print()
    return accepted_filenames, paper_records_by_filename


# =========================================================================
# PDF discovery  —  year-balanced sampling (accepted only)
# =========================================================================

def _extract_year_from_filename(fname):
    """Extract year (int) from filename like ICLR2017_xxx.pdf or ICLR_2024_xxx.pdf."""
    m = re.match(r"[A-Za-z]+[_\-]?(\d{4})", fname)
    if m:
        return int(m.group(1))
    return None


def discover_pdfs(pdf_dir, prefix="ICLR", limit=10,
                  already_processed=None, accepted_filenames=None):
    """
    Recursively find PDF files under *pdf_dir* whose filename starts with *prefix*.
    Returns at most *limit* paths, balanced across years.

    Args:
        pdf_dir: root directory to search
        prefix: filename prefix filter
        limit: target total (already done + new)
        already_processed: set of filenames already processed (for resume)
        accepted_filenames: set of filenames that are accepted papers.
                           If provided, only these PDFs are considered.
    """
    if already_processed is None:
        already_processed = set()

    if not os.path.isdir(pdf_dir):
        print(f"[ERROR] PDF directory does not exist: {pdf_dir}")
        return []

    # Collect all matching PDFs grouped by year
    by_year = defaultdict(list)           # year -> [path, ...] (not yet processed)
    by_year_done = defaultdict(int)       # year -> count already processed
    total_found = 0
    total_accepted = 0
    total_skipped_not_accepted = 0

    for root, dirs, files in os.walk(pdf_dir):
        for fname in files:
            if not (fname.upper().startswith(prefix.upper()) and fname.lower().endswith(".pdf")):
                continue
            total_found += 1

            # Filter: only accepted papers
            if accepted_filenames is not None and fname not in accepted_filenames:
                total_skipped_not_accepted += 1
                continue
            total_accepted += 1

            year = _extract_year_from_filename(fname)
            full_path = os.path.join(root, fname)
            if fname in already_processed:
                by_year_done[year] += 1
            else:
                by_year[year].append(full_path)

    total_done = sum(by_year_done.values())
    print(f"[INFO] Total PDFs with prefix '{prefix}' found: {total_found}")
    if accepted_filenames is not None:
        print(f"[INFO] Accepted papers (with PDF on disk): {total_accepted}")
        print(f"[INFO] Skipped (not accepted): {total_skipped_not_accepted}")
    print(f"[INFO] Already processed: {total_done}")

    # Print per-year distribution
    print(f"[INFO] Year distribution (available / already done / total accepted):")
    all_years = sorted(set(list(by_year.keys()) + list(by_year_done.keys())),
                       key=lambda x: (x is None, x))
    for y in all_years:
        label = str(y) if y is not None else "unknown"
        avail = len(by_year.get(y, []))
        done = by_year_done.get(y, 0)
        print(f"  {label}: {avail} available / {done} done / {avail + done} total")

    if not by_year:
        print("[INFO] No unprocessed PDFs remaining.")
        return []

    # Sort each year's list for deterministic ordering
    for y in by_year:
        by_year[y].sort()

    # Year-balanced sampling, accounting for already-processed counts
    years = sorted([y for y in by_year if y is not None])
    if not years:
        all_flat = []
        for v in by_year.values():
            all_flat.extend(v)
        all_flat.sort()
        need = max(0, limit - total_done)
        result = all_flat[:need]
        print(f"[INFO] No year info, selecting {len(result)} new PDFs")
        return result

    # Target per year = limit / num_years, minus what's already done for that year
    per_year_total_target = max(1, limit // len(years))

    result = []
    year_counts = {}
    shortfall = 0

    for y in years:
        done = by_year_done.get(y, 0)
        available = by_year[y]
        need = max(0, per_year_total_target - done)
        take = min(need, len(available))
        shortfall += max(0, need - len(available))
        result.extend(available[:take])
        year_counts[y] = take

    # Distribute shortfall to years that still have room
    if shortfall > 0:
        for y in sorted(years, key=lambda y: len(by_year[y]), reverse=True):
            if shortfall <= 0:
                break
            available = by_year[y]
            taken = year_counts[y]
            extra = min(shortfall, len(available) - taken)
            if extra > 0:
                result.extend(available[taken:taken + extra])
                year_counts[y] += extra
                shortfall -= extra

    # If overall total (done + new) is still under limit, fill more
    current_total = total_done + len(result)
    if current_total < limit:
        gap = limit - current_total
        for y in sorted(years, key=lambda y: len(by_year[y]), reverse=True):
            if gap <= 0:
                break
            available = by_year[y]
            taken = year_counts[y]
            extra = min(gap, len(available) - taken)
            if extra > 0:
                result.extend(available[taken:taken + extra])
                year_counts[y] += extra
                gap -= extra

    print(f"[INFO] Year-balanced selection ({len(result)} new, "
          f"{total_done} already done, {total_done + len(result)} total, "
          f"limit={limit}):")
    for y in sorted(set(list(year_counts.keys()) + list(by_year_done.keys()))):
        done = by_year_done.get(y, 0)
        new = year_counts.get(y, 0)
        print(f"  {y}: {done} done + {new} new = {done + new}")

    return result


# =========================================================================
# PDF text extraction
# =========================================================================

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text")
        doc.close()
        return full_text
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        return None


# =========================================================================
# Text preprocessing  —  robust truncation after References / Appendix
# =========================================================================

_TRUNCATION_PATTERNS = [
    r"(?m)^\s*(?:\d+\.?\s+)?References\s*$",
    r"(?m)^\s*(?:\d+\.?\s+)?Bibliography\s*$",
    r"(?m)^\s*(?:Appendix|Appendices)(?:\s+[A-Z0-9])?\s*$",
    r"(?m)^\s*(?:\d+\.?\s+)?Supplementary\s+Material\s*$",
    r"(?m)^\s*(?:\d+\.?\s+)?Acknowledge?ments?\s*$",
]


def truncate_after_references(text):
    """
    Remove everything starting from References / Bibliography / Appendix.
    Returns (truncated_text, matched_section_name_or_None).
    """
    earliest_pos = len(text)
    matched_section = None

    for pattern in _TRUNCATION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m and m.start() < earliest_pos:
            earliest_pos = m.start()
            matched_section = pattern

    if earliest_pos < len(text):
        truncated = text[:earliest_pos].rstrip()
        return truncated, matched_section

    return text, None


# =========================================================================
# Metadata extraction from PDF text
# =========================================================================

def extract_title_from_text(text):
    """Heuristic title extraction from the first lines of PDF text."""
    lines = text.split("\n")
    candidate_lines = []

    for line in lines[:30]:
        stripped = line.strip()
        if not stripped:
            if candidate_lines:
                break
            continue

        lower = stripped.lower()
        if any(kw in lower for kw in [
            "published as", "under review", "anonymous",
            "workshop", "proceedings", "arxiv:",
            "preprint", "accepted", "submitted",
        ]):
            if candidate_lines:
                break
            continue

        if candidate_lines and stripped.count(",") >= 2:
            break

        if len(stripped) < 5:
            continue

        candidate_lines.append(stripped)

        if len(candidate_lines) >= 3:
            break

    title = " ".join(candidate_lines).strip()
    return title if title else None


def extract_title_from_filename(pdf_path):
    """Fallback: derive a rough title from the PDF filename."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    base = re.sub(r"^[A-Z]+_?\d{4}_?", "", base)
    title = base.replace("_", " ").strip()
    return title if title else os.path.basename(pdf_path)


def extract_venue_and_year_from_filename(pdf_path):
    """Extract venue and year from filename."""
    base = os.path.basename(pdf_path)
    m = re.match(r"([A-Z]+)[_\-]?(\d{4})", base)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def _parse_json_object(text):
    """Parse a JSON object from raw LLM output, with light fence handling."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    for candidate in (cleaned,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


# =========================================================================
# LLM calling
# =========================================================================

def call_llm(client, model, system_prompt, user_prompt,
             temperature=0.7, max_retries=5, retry_delay=3):
    """Call the OpenAI-compatible API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
            )
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content and content.strip():
                    return content
        except Exception as e:
            print(f"  [ERROR] LLM call failed attempt {attempt + 1}/{max_retries}: {e}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    return None


def call_llm_for_title_and_summary(client, model, summary_prompt_cfg, paper_name_hint,
                                   paper_text, temperature=0.7,
                                   max_retries=5, retry_delay=3):
    """
    Ask the LLM to extract the paper title and summary together.
    Returns (title, summary, raw_response).
    """
    base_user_prompt = summary_prompt_cfg["user_prompt"].format(
        paper_name=paper_name_hint,
        paper_text=paper_text,
    )
    system_prompt = (
        summary_prompt_cfg["system_prompt"].rstrip()
        + "\n\nADDITIONAL OUTPUT REQUIREMENTS:\n"
        + "- Also identify the paper title from the provided paper content.\n"
        + "- Return ONLY a valid JSON object with exactly two string fields: "
        + "\"title\" and \"summary\".\n"
        + "- \"title\" must be the paper's full title as supported by the paper content. "
        + "If the title cannot be determined confidently from the paper content, use the provided paper name hint.\n"
        + "- \"summary\" must fully follow the original summarization instructions above.\n"
        + "- Do not include markdown fences or any explanatory text."
    )
    user_prompt = (
        base_user_prompt
        + "\n\nReturn ONLY a valid JSON object with keys \"title\" and \"summary\". "
        + "The title should be the paper title extracted from the paper content; "
        + "if uncertain, use the paper name hint."
    )

    raw_response = call_llm(
        client,
        model,
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    parsed = _parse_json_object(raw_response)
    if not parsed:
        return None, None, raw_response

    title = _normalize_whitespace(str(parsed.get("title") or ""))
    summary = str(parsed.get("summary") or "").strip()
    return (title or None), (summary or None), raw_response


def call_llm_for_summary(client, model, summary_prompt_cfg, paper_name, paper_text,
                         temperature=0.7, max_retries=5, retry_delay=3):
    """Fallback summary call using the original prompt shape."""
    return call_llm(
        client,
        model,
        summary_prompt_cfg["system_prompt"],
        summary_prompt_cfg["user_prompt"].format(
            paper_name=paper_name,
            paper_text=paper_text,
        ),
        temperature=temperature,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


# =========================================================================
# Core: process a single PDF into a memory entry
# =========================================================================

def process_single_pdf(pdf_path, config, client, paper_record=None, debug_dir=None):
    """
    Full pipeline for one PDF. Thread-safe (no shared mutable state).
    """
    pdf_name = os.path.basename(pdf_path)
    pdf_stem = os.path.splitext(pdf_name)[0]
    tag = f"[{pdf_name}]"

    print(f"\n{'='*80}")
    print(f"{tag} Processing: {pdf_name}")
    print(f"{'='*80}")

    # --- 1. Extract text ---
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return None
    print(f"{tag} Extracted {len(raw_text)} chars")

    if debug_dir:
        _save_debug(debug_dir, f"{pdf_stem}_1_raw.txt", raw_text)

    # --- 2. Truncate ---
    body_text, matched_pattern = truncate_after_references(raw_text)
    removed = len(raw_text) - len(body_text)
    if matched_pattern:
        print(f"{tag} Truncated: {len(body_text)} chars kept, {removed} removed")
    else:
        print(f"{tag} No truncation marker found, keeping full text ({len(body_text)} chars)")

    if debug_dir:
        _save_debug(debug_dir, f"{pdf_stem}_2_truncated.txt", body_text)

    # Print last 200 chars for quick verification
    print(f"{tag} Last 200 chars of truncated text:")
    print(f"  ---begin---")
    for line in body_text[-200:].split("\n"):
        print(f"  | {line}")
    print(f"  ---end---")

    # --- 3. Metadata / title hint ---
    heuristic_title = extract_title_from_text(raw_text)
    filename_title = extract_title_from_filename(pdf_path)
    official_title = None if not paper_record else paper_record.get("official_title")
    paper_name_hint = official_title or heuristic_title or filename_title

    if official_title:
        print(f"{tag} Title hint (OpenReview): {official_title}")
    elif heuristic_title:
        print(f"{tag} Title hint (heuristic): {heuristic_title}")
    else:
        print(f"{tag} Title hint (filename): {filename_title}")

    venue, year = extract_venue_and_year_from_filename(pdf_path)
    print(f"{tag} Venue: {venue}, Year: {year}")

    # --- 4. LLM: Summary ---
    model = config["llm_config"]["model"]
    temperature = config["llm_config"]["temperature"]
    max_retries = config["llm_config"]["max_retries"]
    retry_delay = config["llm_config"]["retry_delay"]
    prompts = config["prompts"]

    print(f"{tag} [STEP 1/2] Extracting title + summary via LLM...")
    title, summary, raw_title_summary = call_llm_for_title_and_summary(
        client,
        model,
        prompts["summary"],
        paper_name_hint=paper_name_hint,
        paper_text=body_text,
        temperature=temperature,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    if debug_dir and raw_title_summary:
        _save_debug(debug_dir, f"{pdf_stem}_3_title_summary_raw.txt", raw_title_summary)

    if not title:
        title = official_title or heuristic_title or filename_title
        print(f"{tag} Title fallback used: {title}")
    else:
        print(f"{tag} Title (LLM): {title}")

    if not summary:
        print(f"{tag} Title+summary JSON parsing failed, retrying summary with original prompt...")
        summary = call_llm_for_summary(
            client,
            model,
            prompts["summary"],
            paper_name=title,
            paper_text=body_text,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    if summary:
        print(f"{tag} Summary OK ({len(summary)} chars)")
        if debug_dir:
            _save_debug(debug_dir, f"{pdf_stem}_4_summary.txt", summary)
    else:
        print(f"{tag} Summary FAILED")

    # --- 5. OpenReview decision / review comments ---
    final_decision = None if not paper_record else paper_record.get("final_decision")
    print(f"{tag} Final decision: {final_decision or 'N/A'}")

    review_comments = {
        "meta_review": [],
        "reviewer_reviews": [],
    }
    if paper_record:
        review_comments = paper_record.get("review_comments") or review_comments
    print(
        f"{tag} Review comments: "
        f"{len(review_comments['meta_review'])} meta-review, "
        f"{len(review_comments['reviewer_reviews'])} reviewer reviews"
    )
    if debug_dir:
        _save_debug(
            debug_dir,
            f"{pdf_stem}_5_final_decision.txt",
            final_decision or "",
        )
        _save_debug(
            debug_dir,
            f"{pdf_stem}_6_review_comments.json",
            json.dumps(review_comments, ensure_ascii=False, indent=2),
        )

    # --- 6. LLM: Innovation points ---
    print(f"{tag} [STEP 2/2] Extracting innovation points via LLM...")
    innovations = call_llm(
        client, model,
        prompts["innovation_extraction"]["system_prompt"],
        prompts["innovation_extraction"]["user_prompt"].format(paper_name=title, paper_text=body_text),
        temperature=temperature, max_retries=max_retries, retry_delay=retry_delay,
    )
    if innovations:
        print(f"{tag} Innovations OK ({len(innovations)} chars)")
        if debug_dir:
            _save_debug(debug_dir, f"{pdf_stem}_7_innovations.txt", innovations)
    else:
        print(f"{tag} Innovations FAILED")

    # --- 7. Assemble memory entry ---
    memory_entry = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "paper_id": hashlib.md5(pdf_name.encode()).hexdigest()[:12],
        "title": title,
        "final_decision": final_decision,
        "summary": summary,
        "innovation_points": innovations,
        "review_comments": review_comments,
        "metadata": {
            "venue": venue,
            "year": year,
            "source_filename": pdf_name,
        },
    }

    return memory_entry


def _save_debug(debug_dir, filename, content):
    """Save a debug/intermediate file."""
    path = os.path.join(debug_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# =========================================================================
# Thread-safe result saving
# =========================================================================

class ResultSaver:
    """Thread-safe incremental result saver."""

    def __init__(self, output_dir, combined_path):
        self._lock = threading.Lock()
        self._output_dir = output_dir
        self._combined_path = combined_path
        self._all_memories = []
        self._processed_filenames = set()

    @staticmethod
    def _is_current_memory_entry(memory):
        if not isinstance(memory, dict):
            return False
        if memory.get("schema_version", 0) < MEMORY_SCHEMA_VERSION:
            return False
        if not isinstance(memory.get("final_decision"), str) or not memory.get("final_decision").strip():
            return False
        review_comments = memory.get("review_comments")
        if not isinstance(review_comments, dict):
            return False
        return (
            isinstance(review_comments.get("meta_review"), list)
            and isinstance(review_comments.get("reviewer_reviews"), list)
        )

    def load_existing(self):
        """Load previous results for resume support."""
        if os.path.exists(self._combined_path):
            try:
                with open(self._combined_path, "r", encoding="utf-8") as f:
                    loaded_memories = json.load(f)

                kept = 0
                skipped = 0
                self._all_memories = []
                self._processed_filenames = set()

                for mem in loaded_memories:
                    if not self._is_current_memory_entry(mem):
                        skipped += 1
                        continue

                    source_filename = ((mem.get("metadata") or {}).get("source_filename"))
                    if not source_filename:
                        skipped += 1
                        continue

                    self._all_memories.append(mem)
                    self._processed_filenames.add(source_filename)
                    kept += 1

                print(f"[RESUME] Loaded {kept} current entries")
                if skipped > 0:
                    print(f"[RESUME] Skipped {skipped} outdated/incomplete entries; they will be rebuilt")
            except Exception as e:
                print(f"[WARN] Failed to load existing results, starting fresh: {e}")
                self._all_memories = []
                self._processed_filenames = set()

    def is_processed(self, filename):
        with self._lock:
            return filename in self._processed_filenames

    def get_processed_filenames(self):
        """Return a copy of the set of already-processed filenames."""
        with self._lock:
            return set(self._processed_filenames)

    def save(self, memory):
        """Save one result (individual JSON + update combined)."""
        with self._lock:
            pdf_name = memory["metadata"]["source_filename"]
            self._all_memories.append(memory)
            self._processed_filenames.add(pdf_name)
            count = len(self._all_memories)

        # Save individual JSON
        individual_path = os.path.join(
            self._output_dir,
            f"{memory['paper_id']}_{pdf_name.replace('.pdf', '')}.json"
        )
        with open(individual_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

        # Update combined file
        with self._lock:
            with open(self._combined_path, "w", encoding="utf-8") as f:
                json.dump(self._all_memories, f, ensure_ascii=False, indent=2)

        print(f"  [SAVED] {pdf_name} (total: {count})")

    @property
    def count(self):
        with self._lock:
            return len(self._all_memories)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Build initial memory from accepted ICLR PDFs")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override max number of PDFs to process")
    parser.add_argument("--workers", type=int, default=128,
                        help="Number of parallel threads (default: 128)")
    args = parser.parse_args()

    config = load_config(args.config)

    # --- Resolve settings ---
    pdf_dir = config["pdf_source"]["pdf_dir"]
    prefix = config["pdf_source"].get("filename_prefix", "ICLR")
    limit = args.limit if args.limit is not None else config["pdf_source"].get("limit", 10)
    workers = args.workers

    # Optional JSONL metadata files for accepted-paper filtering. Leave empty to process all discovered PDFs.
    jsonl_paths = config.get("pdf_source", {}).get("metadata_jsonl_paths", [])

    output_base = str(PROJECT_ROOT)
    output_dir = os.path.normpath(os.path.join(output_base, config.get("output", {}).get("output_dir", "memory_output")))
    os.makedirs(output_dir, exist_ok=True)

    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"[INFO] Debug dir: {debug_dir}")

    combined_path = os.path.join(output_dir, "all_memories.json")

    # --- Step 1: Load accepted paper list from JSONL ---
    accepted_filenames, paper_records_by_filename = load_accepted_papers(
        jsonl_paths,
        venue_prefix=prefix,
    )
    print(f"[INFO] Total accepted {prefix} papers with known PDF: {len(accepted_filenames)}")

    # --- Step 2: Resume support: load existing results ---
    saver = ResultSaver(output_dir, combined_path)
    saver.load_existing()
    already_processed = saver.get_processed_filenames()

    # --- Step 3: Discover PDFs (year-balanced, accepted only, resume-aware) ---
    to_process = discover_pdfs(pdf_dir, prefix=prefix, limit=limit,
                               already_processed=already_processed,
                               accepted_filenames=accepted_filenames)
    if not to_process:
        print("[INFO] Nothing to process. Exiting.")
        return

    print(f"[INFO] {len(already_processed)} already processed, "
          f"{len(to_process)} to process, {workers} workers")

    # --- Step 4: Create LLM client ---
    client = openai.OpenAI(
        api_key=config["api"]["openai_api_key"],
        base_url=config["api"]["openai_base_url"],
        timeout=config["api"].get("openai_timeout", 240.0),
    )

    # --- Step 5: Parallel processing ---
    success_count = 0
    fail_count = 0
    count_lock = threading.Lock()
    start_time = time.time()

    def worker(pdf_path, idx, total):
        nonlocal success_count, fail_count
        try:
            pdf_name = os.path.basename(pdf_path)
            memory = process_single_pdf(
                pdf_path,
                config,
                client,
                paper_record=paper_records_by_filename.get(pdf_name),
                debug_dir=debug_dir,
            )
            if memory:
                saver.save(memory)
                with count_lock:
                    success_count += 1
            else:
                with count_lock:
                    fail_count += 1
        except Exception as e:
            print(f"[ERROR] {os.path.basename(pdf_path)}: {e}")
            traceback.print_exc()
            with count_lock:
                fail_count += 1

        with count_lock:
            done = success_count + fail_count
        print(f"[PROGRESS] {done}/{total} done "
              f"(success={success_count}, fail={fail_count})")

    total = len(to_process)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for idx, pdf_path in enumerate(to_process, 1):
            futures.append(pool.submit(worker, pdf_path, idx, total))

        for future in as_completed(futures):
            pass  # exceptions already handled inside worker

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"DONE")
    print(f"  Already done:    {len(already_processed)}")
    print(f"  Processed:       {len(to_process)}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {fail_count}")
    print(f"  Total entries:   {saver.count}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Debug dir:       {debug_dir}")
    print(f"  Combined file:   {combined_path}")
    print(f"  Workers:         {workers}")
    print(f"  Time elapsed:    {elapsed:.1f}s")
    if success_count > 0:
        print(f"  Avg time/paper:  {elapsed/success_count:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
