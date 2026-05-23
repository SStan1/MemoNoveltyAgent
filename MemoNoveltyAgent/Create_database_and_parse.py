import time
import tempfile
from pathlib import Path
import re

import fitz  # PyMuPDF
from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.dataset import DataSet


def chunk_list(items, chunk_size):
    """
    Yield successive chunk_size-sized chunks from items.
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def remove_references_from_pdf(src_pdf, dst_pdf):
    """
    Remove references section from PDF
    :param src_pdf: Source PDF path
    :param dst_pdf: Destination PDF path
    :return: True if successful, False otherwise
    """
    references_patterns = [
        r'^\s*References?\s*$',
        r'^\s*REFERENCES?\s*$',
        r'^\s*Bibliography\s*$',
        r'^\s*BIBLIOGRAPHY\s*$',
        r'^\s*参考文献\s*$',
        r'^\s*引用文献\s*$',
        r'^\s*文献\s*$',
        r'^\s*\[\s*References?\s*\]\s*$',
        r'^\s*参考文献\s*[:：]\s*$',
        r'^\s*\d+\.?\s*References?\s*$',
        r'^\s*[A-Z]\.\s*References?\s*$'
    ]
    
    doc = None
    new_doc = None
    
    try:
        doc = fitz.open(src_pdf)
        total_pages = len(doc)
        
        ref_start_page = None
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                for pattern in references_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        print(f"[INFO] Found references section at page {page_num + 1}: '{line}'")
                        ref_start_page = page_num
                        break
                if ref_start_page is not None:
                    break
            if ref_start_page is not None:
                break
        
        if ref_start_page is None:
            print(f"[WARN] No references section found in {src_pdf.name}")
            doc.close()
            return False
        
        new_doc = fitz.open()
        
        if ref_start_page > 0:
            new_doc.insert_pdf(doc, from_page=0, to_page=ref_start_page-1)
        else:
            print(f"[WARN] References section found at first page in {src_pdf.name}")
            new_doc.close()
            doc.close()
            return False
        
        new_doc.save(dst_pdf)
        
        print(f"[OK] Removed references from {src_pdf.name}: {total_pages} pages -> {len(new_doc)} pages")
        
        new_doc.close()
        doc.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to remove references from {src_pdf}: {e}")
        if new_doc:
            new_doc.close()
        if doc:
            doc.close()
        return False


def simple_pdf_fix_and_remove_references(src: Path, dst: Path) -> bool:
    """
    Remove references and fix PDF structural issues.
    Both steps must succeed for the function to return True.
    :param src: Source PDF path
    :param dst: Destination PDF path
    :return: True if both steps successful, False otherwise
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        if not remove_references_from_pdf(src, tmp_path):
            print(f"[ERROR] Failed to remove references from {src.name}, skipping this PDF")
            if tmp_path.exists():
                tmp_path.unlink()
            return False
        
        try:
            with fitz.open(tmp_path) as doc:
                doc.save(dst, garbage=4, deflate=True, clean=True)
            print(f"[OK] Successfully fixed PDF structure for {src.name}")
        except Exception as e:
            print(f"[ERROR] Failed to fix PDF structure for {src.name}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return False
        
        if tmp_path.exists():
            tmp_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process {src}: {e}")
        return False


def truncate_filename(filename, max_bytes=120):
    """
    Truncate filename to ensure it doesn't exceed max_bytes when encoded in UTF-8.
    Preserves the file extension.
    """
    path_obj = Path(filename)
    stem = path_obj.stem
    suffix = path_obj.suffix
    
    suffix_bytes = len(suffix.encode('utf-8'))
    max_stem_bytes = max_bytes - suffix_bytes
    
    if len(filename.encode('utf-8')) <= max_bytes:
        return filename
    
    truncated_stem = stem
    while len(truncated_stem.encode('utf-8')) > max_stem_bytes:
        truncated_stem = truncated_stem[:-1]
    
    truncated_filename = f"{truncated_stem}{suffix}"
    
    print(f"[WARN] File name truncated (original length: {len(filename.encode('utf-8'))} bytes)")
    print(f"       From: {filename[:60]}...")
    print(f"       To:   {truncated_filename[:60]}...")
    
    return truncated_filename


def create_and_upload_dataset(rag, config, ds_name, first_pdf_name, pdf_files, description_suffix):
    """
    Helper function to create and upload dataset with batched uploads (50 documents per batch).
    """
    print(f"[INFO] Creating new dataset: {ds_name}")
    
    parser_cfg = DataSet.ParserConfig(
        rag=rag,
        res_dict=config['dataset']['parser_config']
    )

    try:
        dataset = rag.create_dataset(
            name=ds_name,
            description=f"Paper Dataset based on {first_pdf_name} ({description_suffix}, naive parser)",
            embedding_model=config['dataset']['embedding_model'],
            permission=config['dataset']['permission'],
            chunk_method=config['dataset']['chunk_method'],
            parser_config=parser_cfg
        )
        print(f"[OK] Dataset created: {dataset.id} - {ds_name}")
    except Exception as e:
        error_text = str(e)
        if "already exists" not in error_text.lower():
            raise

        print(f"[WARN] Dataset '{ds_name}' already exists during creation, trying to reuse it...")
        dataset = find_existing_dataset(rag, ds_name)
        if dataset is None:
            raise

        try:
            existing_docs = dataset.list_documents(page=1, page_size=1)
        except Exception as doc_error:
            print(f"[WARN] Failed to inspect existing dataset documents: {doc_error}")
            existing_docs = None

        if existing_docs:
            print(f"[OK] Reusing existing dataset with documents: {dataset.id} - {ds_name}")
            return dataset

        print(f"[WARN] Existing dataset '{ds_name}' appears empty, will upload documents into it.")

    print(f"[INFO] Preprocessing {len(pdf_files)} PDFs (removing references and fixing structure)...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="ragflow_"))
    documents = []
    
    successful_count = 0
    skipped_count = 0
    
    for pdf in pdf_files:
        fixed_pdf = tmp_dir / f"processed_{pdf.name}"
        if simple_pdf_fix_and_remove_references(pdf, fixed_pdf):
            safe_filename = truncate_filename(pdf.name, max_bytes=120)
            
            documents.append({
                "display_name": safe_filename,
                "name": safe_filename,
                "blob": fixed_pdf.read_bytes()
            })
            successful_count += 1
            print(f"[OK] Successfully processed {pdf.name}")
        else:
            skipped_count += 1
            print(f"[SKIP] Skipping {pdf.name} due to processing failure")

    print(f"[INFO] Processing summary: {successful_count} successful, {skipped_count} skipped")

    if not documents:
        print("[ERROR] No files available for upload after preprocessing")
        try:
            import shutil
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"[WARN] Failed to clean up temporary directory: {e}")
        return None

    print(f"[INFO] Uploading {len(documents)} preprocessed PDFs in batches of 50...")
    for batch_idx, doc_batch in enumerate(chunk_list(documents, 50), start=1):
        print(f"[INFO] Uploading batch {batch_idx} with {len(doc_batch)} documents...")
        dataset.upload_documents(doc_batch)
        print(f"[OK] Batch {batch_idx} uploaded")

    parse_documents(dataset)
    
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(f"[WARN] Failed to clean up temporary directory: {e}")

    return dataset


def upload_pdfs_to_ragflow(config, pdf_path):
    """
    Batch upload PDFs to RAGFlow (naive parser) and trigger parsing.
    Creates two datasets: main_dataset and all_dataset.
    """
    rag = RAGFlow(
        api_key=config['api']['api_key'], 
        base_url=config['api']['base_url']
    )
    
    p = Path(pdf_path)
    pdf_files = ([p] if p.is_file() else [x for x in p.iterdir() if x.suffix.lower() == ".pdf"])
    if not pdf_files:
        print("[ERROR] No PDF files found")
        return None, None

    main_pdf = next((pdf for pdf in pdf_files if pdf.stem.startswith("MAIN")), None)
    if main_pdf:
        main_suffix = main_pdf.stem[4:24] if len(main_pdf.stem) > 4 else "Unnamed"
        main_ds_name = f"{main_suffix}_main_only_dataset"
        all_ds_name = f"{main_suffix}_dataset"
        first_pdf_name = main_pdf.stem
    else:
        print("[WARN] No PDF starting with 'MAIN' found")
        first_pdf_name = "NoMainPDF"
        main_ds_name = "NoMainPDF_main_dataset"
        all_ds_name = "NoMainPDF_dataset"

    print(f"[INFO] Checking for existing datasets...")
    existing_main_dataset = find_existing_dataset(rag, main_ds_name)
    existing_all_dataset = find_existing_dataset(rag, all_ds_name)
    
    main_dataset = None
    if existing_main_dataset:
        print(f"[OK] Found existing main dataset: {existing_main_dataset.id} - {main_ds_name}")
        is_complete, status_summary = check_dataset_parsing_complete(
            existing_main_dataset, 
            check_interval=10,
            max_wait_time=30
        )
        
        if is_complete:
            success_count = status_summary.get("DONE", 0)
            print(f"[OK] Existing main dataset parsing is complete: {success_count} successful")
            main_dataset = existing_main_dataset
        else:
            print(f"[INFO] Existing main dataset parsing is not complete, will wait for completion...")
            main_dataset = existing_main_dataset
    else:
        if main_pdf:
            main_dataset = create_and_upload_dataset(rag, config, main_ds_name, first_pdf_name, [main_pdf], "main PDF only")
        else:
            print("[WARN] No main PDF found, skipping main dataset creation")
    
    all_dataset = None
    if existing_all_dataset:
        print(f"[OK] Found existing all dataset: {existing_all_dataset.id} - {all_ds_name}")
        is_complete, status_summary = check_dataset_parsing_complete(
            existing_all_dataset, 
            check_interval=10,
            max_wait_time=30
        )
        
        if is_complete:
            success_count = status_summary.get("DONE", 0)
            print(f"[OK] Existing all dataset parsing is complete: {success_count} successful")
            all_dataset = existing_all_dataset
        else:
            print(f"[INFO] Existing all dataset parsing is not complete, will wait for completion...")
            all_dataset = existing_all_dataset
    else:
        if main_pdf:
            related_pdfs = [pdf for pdf in pdf_files if pdf != main_pdf]
            if related_pdfs:
                all_dataset = create_and_upload_dataset(rag, config, all_ds_name, first_pdf_name, related_pdfs, "related PDFs only")
            else:
                print("[WARN] No related PDFs found, skipping all dataset creation")
        else:
            all_dataset = create_and_upload_dataset(rag, config, all_ds_name, first_pdf_name, pdf_files, "all PDFs")
    
    return main_dataset, all_dataset


def find_existing_dataset(rag, dataset_name):
    """
    Find existing dataset by name.
    """
    try:
        try:
            datasets = list(rag.list_datasets(name=dataset_name))
            for dataset in datasets:
                if getattr(dataset, "name", None) == dataset_name:
                    return dataset
            if datasets:
                return datasets[0]
        except Exception as name_query_error:
            print(f"[WARN] Name-based dataset lookup failed for '{dataset_name}': {name_query_error}")

        seen_ids = set()
        for page in range(1, 101):
            try:
                datasets = list(rag.list_datasets(page=page, page_size=100))
            except TypeError:
                if page > 1:
                    break
                datasets = list(rag.list_datasets(page_size=100))

            if not datasets:
                break

            new_items = 0
            for dataset in datasets:
                dataset_id = getattr(dataset, "id", None)
                if dataset_id and dataset_id not in seen_ids:
                    seen_ids.add(dataset_id)
                    new_items += 1
                if getattr(dataset, "name", None) == dataset_name:
                    return dataset

            if new_items == 0:
                break
        return None
    except Exception as e:
        print(f"[ERROR] Failed to search for existing dataset: {e}")
        return None


def parse_documents(dataset):
    """
    Trigger parsing of documents in the dataset (up to 200 documents).
    :param dataset: Dataset object
    """
    all_ids = []
    
    for page in [1, 2]:
        try:
            documents = dataset.list_documents(page=page, page_size=100, keywords=".pdf")
            if documents:
                page_ids = [d.id for d in documents]
                all_ids.extend(page_ids)
                print(f"[INFO] Retrieved page {page}: {len(page_ids)} documents")
            else:
                print(f"[INFO] Page {page} is empty")
                break
        except Exception as e:
            print(f"[ERROR] Failed to retrieve documents at page {page}: {e}")
            break
    
    if all_ids:
        print(f"[INFO] Total documents found: {len(all_ids)}")
        dataset.async_parse_documents(all_ids)
        print(f"[OK] Triggered parsing for {len(all_ids)} documents (naive)")
    else:
        print("[WARN] No documents found for parsing")


def check_dataset_parsing_complete(dataset, check_interval, max_wait_time):
    """
    Check if all documents in the dataset have completed parsing (up to 200 documents)
    :param dataset: Dataset object
    :return: tuple (is_complete: bool, status_summary: dict)
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        all_documents = []
        
        for page in [1, 2]:
            try:
                documents = dataset.list_documents(page=page, page_size=100)
                if documents:
                    all_documents.extend(documents)
                else:
                    break
            except Exception as e:
                print(f"[ERROR] Failed to retrieve documents at page {page}: {e}")
                break
        
        if not all_documents:
            print("[WARN] No documents found in dataset")
            return False, {}
        
        status_count = {
            "UNSTART": 0,
            "RUNNING": 0,
            "CANCEL": 0,
            "DONE": 0,
            "FAIL": 0
        }
        
        for doc in all_documents:
            status = doc.run
            if status in status_count:
                status_count[status] += 1
            else:
                status_count.setdefault("UNKNOWN", 0)
                status_count["UNKNOWN"] += 1
        
        total_docs = len(all_documents)
        completed_docs = status_count["DONE"] + status_count["FAIL"]
        
        print(f"[INFO] Parsing progress: {completed_docs}/{total_docs} documents completed")
        print(f"[INFO] Status breakdown: {status_count}")
        
        if completed_docs == total_docs:
            print(f"[OK] All documents parsing completed!")
            print(f"[INFO] Final status: {status_count['DONE']} DONE, {status_count['FAIL']} FAIL")
            return True, status_count
        
        running_docs = status_count["RUNNING"] + status_count["UNSTART"]
        if running_docs > 0:
            print(f"[INFO] Still processing {running_docs} documents, waiting {check_interval}s...")
            time.sleep(check_interval)
        else:
            print(f"[WARN] No running documents but parsing not complete. Status: {status_count}")
            break
    
    elapsed_time = time.time() - start_time
    print(f"[TIMEOUT] Parsing check stopped after {elapsed_time:.1f}s")
    print(f"[INFO] Total documents checked: {len(all_documents) if all_documents else 0}")
    return False, status_count


def wait_for_parsing_completion(dataset, config):
    """
    Convenience function to wait for dataset parsing completion
    """
    check_interval = config['parsing']['check_interval']
    max_wait_time = config['parsing']['max_wait_time']
    
    print(f"[INFO] Waiting for dataset parsing to complete (max wait: {max_wait_time}s)...")
    is_complete, status_summary = check_dataset_parsing_complete(
        dataset, check_interval, max_wait_time
    )
    
    if is_complete:
        success_count = status_summary.get("DONE", 0)
        fail_count = status_summary.get("FAIL", 0)
        print(f"[SUCCESS] Dataset parsing completed: {success_count} successful, {fail_count} failed")
        return True
    else:
        print(f"[FAILED] Dataset parsing did not complete within {max_wait_time}s")
        return False
