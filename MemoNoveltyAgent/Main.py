import os
import sys
import json
import re
import time
import traceback
from pathlib import Path

# Use the same clean_filename as the crawling/database pipeline
# to ensure naming consistency across all stages
from clawer_papers import clean_filename

def apply_env_overrides(config):
    """Fill sensitive/runtime settings from environment variables when present."""
    api = config.setdefault("api", {})
    env_map = {
        "openai_api_key": "OPENAI_API_KEY",
        "openai_base_url": "OPENAI_BASE_URL",
        "openai_model": "OPENAI_MODEL",
        "api_key": "RAGFLOW_API_KEY",
        "base_url": "RAGFLOW_BASE_URL",
        "dataset_name": "RAGFLOW_DATASET_NAME",
        "chat_name": "RAGFLOW_CHAT_NAME",
    }
    defaults = {
        "openai_base_url": "https://api.openai.com/v1",
        "base_url": "http://localhost:9380",
    }
    for key, env_name in env_map.items():
        value = os.environ.get(env_name)
        if value:
            api[key] = value
        elif key in defaults and not api.get(key):
            api[key] = defaults[key]

    paper_name = os.environ.get("NOVELTYAGENT_PAPER_NAME")
    if paper_name:
        config["paper_name"] = paper_name

    memory = config.setdefault("expert_memory", {})
    memory_model = os.environ.get("MEMORY_EMBEDDING_MODEL")
    if memory_model:
        memory["embedding_model_path"] = memory_model
    return config


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return apply_env_overrides(config)

# [REMOVED] sanitize_filename — replaced by clean_filename from clawer_papers
# to unify the cleaning logic with crawling and database construction.

def find_main_pdf_path(paper_folder_path):
    for filename in os.listdir(paper_folder_path):
        if filename.lower().startswith("main_") and filename.lower().endswith(".pdf"):
            return os.path.join(paper_folder_path, filename)
    return None

def derive_report_base_name(main_pdf_path):
    """
    Derive the report base name from the MAIN PDF filename.

    MAIN PDF naming (set by clawer_papers):
        MAIN_<clean_filename(title)[:100]>.pdf

    Dataset naming (set by Create_database_and_parse.upload_pdfs_to_ragflow):
        main_pdf.stem[4:24] + "_main_only_dataset"  /  "_dataset"
        i.e.  "_" + clean_filename(title)[:19] + suffix

    Evaluation.infer_dataset_name reconstructs the dataset name as:
        "_" + paper_name[:19] + "_main_only_dataset"  /  "_dataset"
        where paper_name = report filename without extension

    Therefore, report filename (without extension) must equal
    clean_filename(title)[:100], which is exactly MAIN PDF stem
    with the "MAIN_" prefix stripped.
    """
    main_pdf_stem = os.path.splitext(os.path.basename(main_pdf_path))[0]
    if main_pdf_stem.startswith("MAIN_"):
        return main_pdf_stem[5:]
    return main_pdf_stem

def extract_paper_name(paper_folder_path):
    """
    Extract paper name from a paper folder path (for batch mode).
    Strips the prefix before the first underscore in the folder name.
    E.g. "001_SomePaperTitle" -> "SomePaperTitle"
    """
    base_name = os.path.basename(paper_folder_path)
    underscore_index = base_name.find('_')
    if underscore_index != -1:
        return base_name[underscore_index + 1:]
    return base_name


def get_batch_collection_path(config):
    return (
        config.get('batch_mode', {}).get('batch_collection_path')
        or config.get('paths', {}).get('batch_collection_path')
        or config.get('paths', {}).get('database_dir')
        or ''
    )


def infer_output_tier(config, paper_folder_path):
    if not paper_folder_path:
        return "tier_single"

    collection_path = get_batch_collection_path(config)
    try:
        if collection_path:
            relative = Path(paper_folder_path).resolve().relative_to(Path(collection_path).resolve())
            for part in relative.parts:
                if part.startswith("tier_"):
                    return part
            if relative.parts:
                return relative.parts[0]
    except Exception:
        pass

    for part in Path(paper_folder_path).parts:
        if part.startswith("tier_"):
            return part
    return "tier_unknown"


def get_pipeline_state_dir(config, paper_folder_path, report_base_name, paper_name):
    state_root = config.get('paths', {}).get('result_dir', './result')
    tier = infer_output_tier(config, paper_folder_path)
    state_name = report_base_name or clean_filename(paper_name)
    state_dir = os.path.join(state_root, tier, state_name)
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def get_final_report_path(config, paper_folder_path, report_base_name):
    reports_root = (
        config.get('paths', {}).get('reports_dir')
        or config.get('paths', {}).get('result_dir', './result')
    )
    tier = infer_output_tier(config, paper_folder_path)
    report_dir = os.path.join(reports_root, tier)
    os.makedirs(report_dir, exist_ok=True)
    return os.path.join(report_dir, f"{report_base_name}.txt")


def strip_report_wrapper(report_text):
    text = (report_text or "").strip()
    match = re.search(r'##\s*1\.\s*Paper Content Summary', text)
    if match:
        text = text[match.start():]
    text = re.sub(
        r'\n=+\s*\nEnd of (?:Polished )?Report\s*\n=+\s*$',
        '',
        text,
        flags=re.I
    ).strip()
    return text


def wrap_report_for_evaluation(report_text):
    body = strip_report_wrapper(report_text)
    return (
        "AI RESPONSE:\n"
        + "=" * 100 + "\n"
        + body + "\n"
        + "=" * 100 + "\n"
        + "End of Report\n"
        + "=" * 100 + "\n"
    )


def write_text_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content or "")
    print(f"  [INFO] Saved artifact: {filepath}")


def save_generation_artifacts(config, report_base_name, summary_result=None,
                              innovation_result=None, comparison_data=None,
                              initial_report=None, validated_report=None,
                              polished_report=None, report_generation_trace=None):
    paths = config.get('paths', {})

    if summary_result and paths.get('paper_summary_dir'):
        write_text_file(
            os.path.join(paths['paper_summary_dir'], f"{report_base_name}_content_summary.txt"),
            summary_result
        )

    if innovation_result and paths.get('innovation_points_dir'):
        write_text_file(
            os.path.join(paths['innovation_points_dir'], f"{report_base_name}_innovation_points.txt"),
            innovation_result
        )

    if comparison_data and paths.get('compare_result_dir'):
        comparison_dir = os.path.join(paths['compare_result_dir'], report_base_name)
        os.makedirs(comparison_dir, exist_ok=True)
        for item in sorted(comparison_data, key=lambda x: x.get('point_number', 0)):
            point_number = item.get('point_number', 'NA')
            write_text_file(
                os.path.join(comparison_dir, f"point_{point_number}_comparison.txt"),
                item.get('content', '')
            )
            write_text_file(
                os.path.join(comparison_dir, f"point_{point_number}_rag_chunks.txt"),
                item.get('rag_knowledge', '')
            )
            write_text_file(
                os.path.join(comparison_dir, f"point_{point_number}_expert_memory.txt"),
                item.get('expert_knowledge', '')
            )
            retrieval_trace = {
                'point_number': point_number,
                'innovation_point': item.get('innovation_point', ''),
                'generated_queries': item.get('generated_queries', []),
                'rag_retrieval_trace': item.get('rag_retrieval_trace', {}),
                'expert_knowledge_result': item.get('expert_knowledge_result', {}),
                'comparison_prompt_inputs': item.get('comparison_prompt_inputs', {}),
            }
            write_text_file(
                os.path.join(comparison_dir, f"point_{point_number}_inputs_and_retrieval.json"),
                json.dumps(retrieval_trace, ensure_ascii=False, indent=2, default=str)
            )
        write_text_file(
            os.path.join(comparison_dir, "comparison_data.json"),
            json.dumps(comparison_data, ensure_ascii=False, indent=2, default=str)
        )

    if report_generation_trace and paths.get('compare_result_dir'):
        comparison_dir = os.path.join(paths['compare_result_dir'], report_base_name)
        os.makedirs(comparison_dir, exist_ok=True)
        write_text_file(
            os.path.join(comparison_dir, "report_generation_trace.json"),
            json.dumps(report_generation_trace, ensure_ascii=False, indent=2, default=str)
        )
        write_text_file(
            os.path.join(comparison_dir, "section2_input_for_summary.txt"),
            report_generation_trace.get('section2_input_for_summary', '')
        )
        write_text_file(
            os.path.join(comparison_dir, "expert_memory_bundle_for_summary.txt"),
            report_generation_trace.get('expert_knowledge_bundle_for_summary', '')
        )
        write_text_file(
            os.path.join(comparison_dir, "section3_prompt.txt"),
            report_generation_trace.get('section3_prompt', '')
        )

    if initial_report and paths.get('draft_reports_dir'):
        write_text_file(
            os.path.join(paths['draft_reports_dir'], f"{report_base_name}_draft.txt"),
            strip_report_wrapper(initial_report)
        )

    if validated_report and paths.get('validated_reports_dir'):
        write_text_file(
            os.path.join(paths['validated_reports_dir'], f"{report_base_name}_validated.txt"),
            strip_report_wrapper(validated_report)
        )

    if polished_report and paths.get('polished_reports_dir'):
        write_text_file(
            os.path.join(paths['polished_reports_dir'], f"{report_base_name}_polished.txt"),
            strip_report_wrapper(polished_report)
        )

# ================= Checkpoint utilities for resume support =================

def get_checkpoint_dir(base_result_dir):
    """Create a checkpoints subdirectory inside the result dir for saving intermediate outputs."""
    checkpoint_dir = os.path.join(base_result_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(checkpoint_dir, filename, content):
    """Save intermediate results to the checkpoint directory."""
    filepath = os.path.join(checkpoint_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(content))
    print(f"  [INFO] Checkpoint saved: {filepath}")

def load_checkpoint(checkpoint_dir, filename, as_json=False):
    """Load intermediate results from the checkpoint directory; returns None if missing or empty."""
    filepath = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if as_json:
                data = json.load(f)
                if not data:
                    return None
                return data
            else:
                content = f.read()
                if content and content.strip():
                    return content
                return None
    except Exception as e:
        print(f"  [WARN] Failed to load checkpoint {filepath}: {e}")
        return None


def is_current_comparison_checkpoint(comparison_data):
    if not isinstance(comparison_data, list) or not comparison_data:
        return False
    for item in comparison_data:
        if not isinstance(item, dict):
            return False
        if not item.get('generated_queries'):
            return False
        if 'rag_retrieval_trace' not in item:
            return False
        expert_result = item.get('expert_knowledge_result')
        if isinstance(expert_result, dict) and expert_result:
            if 'selected_parent_aggregates_by_theme_id' not in expert_result:
                return False
        if 'comparison_prompt_inputs' not in item:
            return False
    return True

# ================= Batch mode utilities =================

def get_paper_folders(papers_collection_path):
    """
    Get all paper folders from the papers collection directory, searching recursively.
    A valid paper folder is a directory that contains at least one MAIN_*.pdf file.
    """
    paper_folders = []
    if not os.path.isdir(papers_collection_path):
        print(f"[ERROR] The specified path is not a directory: {papers_collection_path}")
        return paper_folders

    for root, dirs, files in os.walk(papers_collection_path):
        has_main_pdf = any(
            f.lower().startswith('main_') and f.lower().endswith('.pdf')
            for f in files
        )
        if has_main_pdf and root not in paper_folders:
            paper_folders.append(root)

    return sorted(paper_folders)


def should_skip_paper(config, paper_name, paper_folder_path, report_base_name):
    """
    Check if a paper has already been fully processed by looking at the final reports directory.
    Returns True if a final report exists and is valid (> 500 bytes).
    """
    final_path = get_final_report_path(config, paper_folder_path, report_base_name)
    if os.path.isfile(final_path) and os.path.getsize(final_path) > 500:
        print(f"[SKIP] Report already exists for '{paper_name}' ({os.path.getsize(final_path)} bytes)")
        print(f"       Location: {final_path}")
        return True

    return False


def print_section_header(title, char='=', width=100):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")

# ==================================================================

def run_single_paper_pipeline(config, paper_name=None, paper_folder_path=None,
                              enable_validation=True, force_rerun=False):
    """
    Run the full pipeline for a single paper.

    This is the core pipeline function, used by both single-paper mode and batch mode.

    Args:
        config: Configuration dictionary loaded from config.json.
        paper_name: Name of the paper. If None, read from config['paper_name'].
                    In batch mode this is derived from the folder name.
        paper_folder_path: Path to the pre-existing paper folder (batch mode).
                           If None, the paper will be auto-crawled (single mode).
        enable_validation: Whether to run citation validation (step 6).
        force_rerun: If True, ignore existing checkpoints and rerun all steps.

    Returns:
        True on success, False on failure, 'skipped' if already completed.
    """
    # --- Determine paper_name ---
    if paper_name is None:
        paper_name = config.get('paper_name')
    if not paper_name:
        print("[ERROR] 'paper_name' is required (via config or argument)")
        return False

    # Keep stdout/stderr on the normal process stream. The caller already owns
    # complete logs, so this pipeline no longer creates one log file per paper.
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        total_steps = 7 if enable_validation else 6

        print(f"\n{'='*100}")
        print(f"STARTING PIPELINE FOR PAPER: {paper_name}")
        print(f"MODE: {'Batch (pre-existing folder)' if paper_folder_path else 'Single (auto-crawl)'}")
        print(f"VALIDATION: {'Enabled' if enable_validation else 'Disabled'}")
        print(f"{'='*100}\n")

        # 1. Get paper folder
        step = 1
        if paper_folder_path is None:
            # Single-paper mode: auto-crawl
            print(f"\n[STEP {step}/{total_steps}] Checking/Downloading Paper...")
            from clawer_papers import download_paper_if_needed
            paper_folder_path = download_paper_if_needed(config)
            if not paper_folder_path:
                print("[ERROR] Failed to get or download paper.")
                return False
        else:
            # Batch mode: use pre-existing folder
            print(f"\n[STEP {step}/{total_steps}] Using pre-existing paper folder...")
            print(f"  Folder: {paper_folder_path}")
            if not os.path.isdir(paper_folder_path):
                print(f"[ERROR] Paper folder does not exist: {paper_folder_path}")
                return False

        # Get main_pdf_path early since multiple steps need it
        main_pdf_path = find_main_pdf_path(paper_folder_path)
        if not main_pdf_path:
            print(f"[ERROR] MAIN_*.pdf not found in {paper_folder_path}")
            return False

        report_base_name = derive_report_base_name(main_pdf_path)
        print(f"[INFO] Report base name (from MAIN PDF): {report_base_name}")

        paper_result_dir = get_pipeline_state_dir(config, paper_folder_path, report_base_name, paper_name)
        checkpoint_dir = get_checkpoint_dir(paper_result_dir)
        final_path = get_final_report_path(config, paper_folder_path, report_base_name)

        print(f"[INFO] Pipeline state dir: {paper_result_dir}")
        print(f"[INFO] Checkpoint dir: {checkpoint_dir}")
        print(f"[INFO] Final report path: {final_path}")

        # --- Check skip (batch resume support) ---
        if not force_rerun and should_skip_paper(config, paper_name, paper_folder_path, report_base_name):
            return 'skipped'

        # 2. Upload and parse
        step += 1
        print(f"\n[STEP {step}/{total_steps}] Uploading and Parsing PDFs in RAGFlow...")
        from Create_database_and_parse import upload_pdfs_to_ragflow, wait_for_parsing_completion
        main_dataset, all_dataset = upload_pdfs_to_ragflow(config, paper_folder_path)
        if not all_dataset:
            print("[ERROR] Failed to create dataset.")
            return False

        all_parsing_success = wait_for_parsing_completion(all_dataset, config)
        if not all_parsing_success:
            print(f"[ERROR] Dataset parsing failed for all_dataset. Aborting pipeline.")
            return False

        if main_dataset:
            main_parsing_success = wait_for_parsing_completion(main_dataset, config)
            if not main_parsing_success:
                print(f"[WARN] Main-only dataset parsing failed, continuing with all_dataset only.")

        # 3. Extract Summary and Innovation (with checkpoint/resume support)
        step += 1
        print(f"\n[STEP {step}/{total_steps}] Extracting Summary and Innovation Points...")

        if not force_rerun:
            summary_result = load_checkpoint(checkpoint_dir, "summary.txt")
            innovation_result = load_checkpoint(checkpoint_dir, "innovation_points.txt")
        else:
            summary_result = None
            innovation_result = None

        if summary_result:
            print("  [OK] Loaded summary from checkpoint, skipping summary extraction.")
        else:
            from Generate_Mainpaper_summary import get_paper_summary
            summary_result = get_paper_summary(config, paper_name, main_pdf_path)
            if summary_result:
                save_checkpoint(checkpoint_dir, "summary.txt", summary_result)

        if innovation_result:
            print("  [OK] Loaded innovation points from checkpoint, skipping innovation extraction.")
        else:
            from Generate_innovation_points import get_paper_innovation
            innovation_result = get_paper_innovation(config, paper_name, main_pdf_path)
            if innovation_result:
                save_checkpoint(checkpoint_dir, "innovation_points.txt", innovation_result)

        if not summary_result or not innovation_result:
            print("[ERROR] Failed to extract summary or innovation points.")
            return False

        save_generation_artifacts(
            config,
            report_base_name,
            summary_result=summary_result,
            innovation_result=innovation_result,
        )

        # 4. Compare Innovation (with checkpoint/resume support)
        step += 1
        print(f"\n[STEP {step}/{total_steps}] Comparing Innovation Points via RAG...")

        comparison_regenerated = False
        if not force_rerun:
            comparison_data = load_checkpoint(checkpoint_dir, "comparison_data.json", as_json=True)
            if comparison_data and not is_current_comparison_checkpoint(comparison_data):
                print("  [WARN] Existing comparison checkpoint uses an older format; rerunning comparison.")
                comparison_data = None
        else:
            comparison_data = None

        if comparison_data:
            print("  [OK] Loaded comparison data from checkpoint, skipping comparison.")
        else:
            from Compare_innovation_points import compare_paper_innovations
            comparison_data = compare_paper_innovations(
                config, paper_name, all_dataset.name, innovation_result, main_pdf_path
            )
            if comparison_data:
                comparison_regenerated = True
                save_checkpoint(checkpoint_dir, "comparison_data.json", comparison_data)

        if not comparison_data:
            print("[ERROR] Failed to compare innovation points.")
            return False

        save_generation_artifacts(
            config,
            report_base_name,
            comparison_data=comparison_data,
        )

        # 5. Generate report (with checkpoint/resume support)
        step += 1
        print(f"\n[STEP {step}/{total_steps}] Generating Comprehensive Report...")

        initial_report_regenerated = False
        if not force_rerun and not comparison_regenerated:
            initial_report = load_checkpoint(checkpoint_dir, "initial_report.txt")
            report_generation_trace = load_checkpoint(checkpoint_dir, "report_generation_trace.json", as_json=True)
            if initial_report and not report_generation_trace:
                print("  [WARN] Existing report checkpoint lacks generation trace; regenerating report.")
                initial_report = None
        else:
            initial_report = None
            report_generation_trace = None
        point_count = len(comparison_data)

        if initial_report:
            print("  [OK] Loaded initial report from checkpoint, skipping report generation.")
        else:
            from Write_reports import InnovationReportGenerator
            generator = InnovationReportGenerator(config)
            report_content, point_count = generator.generate_comprehensive_report(
                paper_name, summary_result, innovation_result, comparison_data
            )

            if report_content:
                initial_report = report_content
                initial_report_regenerated = True
                report_generation_trace = getattr(generator, "last_report_generation_trace", None)
                save_checkpoint(checkpoint_dir, "initial_report.txt", initial_report)
                if report_generation_trace:
                    save_checkpoint(checkpoint_dir, "report_generation_trace.json", report_generation_trace)

        if not initial_report:
            print("[ERROR] Failed to generate report.")
            return False

        save_generation_artifacts(
            config,
            report_base_name,
            initial_report=initial_report,
            report_generation_trace=report_generation_trace,
        )

        # 6. Validate citations (optional, with checkpoint/resume support)
        report_to_polish = initial_report

        if enable_validation:
            step += 1
            print(f"\n[STEP {step}/{total_steps}] Validating and Correcting Citations...")

            if not force_rerun and not initial_report_regenerated:
                validated_report = load_checkpoint(checkpoint_dir, "validated_report.txt")
            else:
                validated_report = None

            if validated_report:
                print("  [OK] Loaded validated report from checkpoint, skipping citation validation.")
            else:
                try:
                    from Validate_and_correct_citations import CitationValidator
                    validator = CitationValidator(config)
                    validated_report = validator.validate_and_correct_single_report(initial_report, paper_folder_path)
                    if validated_report:
                        save_checkpoint(checkpoint_dir, "validated_report.txt", validated_report)
                except Exception as e:
                    print(f"[ERROR] Citation validation failed: {e}")
                    traceback.print_exc()
                    validated_report = None

            if validated_report:
                report_to_polish = validated_report
                save_generation_artifacts(
                    config,
                    report_base_name,
                    validated_report=validated_report,
                )
            else:
                print("[WARN] Citation validation returned empty, using initial report.")

        # 7 (or 6). Polish report
        step += 1
        print(f"\n[STEP {step}/{total_steps}] Polishing Final Report...")
        from Final_polish import ReportPolisher
        polisher = ReportPolisher(config)
        polished_report = polisher.polish_single_report(report_to_polish, paper_name, point_count)

        save_generation_artifacts(
            config,
            report_base_name,
            polished_report=polished_report,
        )

        # Save final report in the exact tiered layout expected by Checkeval.
        final_report_for_evaluation = wrap_report_for_evaluation(polished_report)
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_report_for_evaluation)

        print(f"\n{'='*100}")
        print(f"[OK] PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Pipeline State Directory: {paper_result_dir}")
        print(f"Final Report Saved To: {final_path}")
        print(f"Checkpoints Saved In: {checkpoint_dir}")
        print(f"{'='*100}\n")

        return True

    except Exception as e:
        print(f"[ERROR] Exception in pipeline for '{paper_name}': {e}")
        traceback.print_exc()
        return False

    finally:
        # Restore stdout/stderr so batch loop can continue printing normally
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# ================= Batch processing =================

def process_batch_papers(config, force_rerun=False, enable_validation=True):
    """
    Process all paper folders in the batch collection directory.

    Expects config['batch_mode']['batch_collection_path'] to point to a directory
    containing paper sub-folders, each with a MAIN_*.pdf file.
    """
    batch_config = config.get('batch_mode', {})
    batch_collection_path = get_batch_collection_path(config)

    if not batch_collection_path or not os.path.isdir(batch_collection_path):
        print(f"[ERROR] Invalid batch_collection_path: {batch_collection_path}")
        return

    paper_folders = get_paper_folders(batch_collection_path)

    if not paper_folders:
        print(f"[ERROR] No paper folders found in {batch_collection_path}")
        return

    total_papers = len(paper_folders)
    successful_papers = 0
    skipped_papers = 0
    failed_papers = []

    print_section_header("BATCH PROCESSING MODE", char='*')
    print(f"Collection Path:   {batch_collection_path}")
    print(f"Total Papers:      {total_papers}")
    print(f"Force Rerun:       {'Yes' if force_rerun else 'No (resume by default)'}")
    print(f"Enable Validation: {'Yes' if enable_validation else 'No'}")
    print("=" * 100)

    start_time = time.time()

    for i, paper_folder in enumerate(paper_folders, 1):
        paper_name = extract_paper_name(paper_folder)

        print_section_header(f"Paper {i}/{total_papers}: {paper_name}", char='-')

        try:
            result = run_single_paper_pipeline(
                config,
                paper_name=paper_name,
                paper_folder_path=paper_folder,
                enable_validation=enable_validation,
                force_rerun=force_rerun,
            )

            if result == 'skipped':
                skipped_papers += 1
            elif result:
                successful_papers += 1
            else:
                failed_papers.append(paper_name)
        except Exception as e:
            print(f"[ERROR] Unhandled exception for '{paper_name}': {e}")
            traceback.print_exc()
            failed_papers.append(paper_name)

        print(f"\n[PROGRESS] {i}/{total_papers} | "
              f"Success: {successful_papers} | Skipped: {skipped_papers} | "
              f"Failed: {len(failed_papers)}")

    elapsed_time = time.time() - start_time

    print_section_header("BATCH PROCESSING SUMMARY", char='*')
    print(f"Total Papers:           {total_papers}")
    print(f"Successfully Processed: {successful_papers}")
    print(f"Skipped (completed):    {skipped_papers}")
    print(f"Failed:                 {len(failed_papers)}")
    print(f"Total Time:             {elapsed_time/60:.2f} minutes")
    if total_papers > 0:
        print(f"Average Time/Paper:     {elapsed_time/total_papers:.2f} seconds")

    if failed_papers:
        print(f"\n[FAILED PAPERS]")
        for idx, name in enumerate(failed_papers, 1):
            print(f"  {idx}. {name}")

    print("=" * 100 + "\n")


# ================= Entry point =================

if __name__ == "__main__":
    config = load_config()

    # All runtime settings come from config.json, single source of truth
    batch_config = config.get('batch_mode', {})
    batch_enabled = batch_config.get('enabled', False)
    batch_collection_path = get_batch_collection_path(config)
    enable_validation = batch_config.get('enable_validation', True)
    force_rerun = batch_config.get('force_rerun', False)

    has_paper_name = bool(config.get('paper_name', '').strip())
    has_batch_path = bool(batch_collection_path) and os.path.isdir(batch_collection_path)

    if batch_enabled and has_batch_path:
        # Batch mode
        process_batch_papers(
            config,
            force_rerun=force_rerun,
            enable_validation=enable_validation,
        )
    elif has_paper_name:
        # Single-paper mode
        result = run_single_paper_pipeline(
            config,
            paper_name=config['paper_name'],
            paper_folder_path=None,
            enable_validation=enable_validation,
            force_rerun=force_rerun,
        )
        if result == 'skipped':
            print("[INFO] Paper already processed, skipped.")
        elif result:
            print("[INFO] Single paper pipeline completed successfully.")
        else:
            print("[ERROR] Single paper pipeline failed.")
    else:
        print("[ERROR] Nothing to do. Either set 'paper_name' in config.json for single-paper mode,")
        print("        or set 'batch_mode.enabled' = true with a valid 'batch_mode.batch_collection_path'")
        print("        for batch processing mode.")
