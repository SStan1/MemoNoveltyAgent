#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    import arxiv
    ARXIV_LIB_AVAILABLE = True
except ImportError:
    ARXIV_LIB_AVAILABLE = False

# Default value used when config is not provided
DEFAULT_MAX_TOTAL = 199

def clean_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > max_len:
        name = name[:max_len - 3] + "..."
    return name or "untitled"

def _normalize_for_match(s: str) -> str:
    """
    Normalize a string for fuzzy folder name matching.
    Converts colons, underscores, ellipses, etc. into single spaces
    and lowercases so that e.g. 'AgentClinic: a multimodal...' and
    'AgentClinic_ a multimodal...' can match each other.
    """
    s = s.lower().strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', ' ', s)
    s = re.sub(r'[_.\s]+', ' ', s).strip()
    return s


def canonical_title_for_exact_match(s: str) -> str:
    # Strict matching follows Clawer_5Venues.py: only collapse whitespace.
    return re.sub(r"\s+", " ", str(s or "")).strip()


def is_title_match(expected: str, candidate: str) -> bool:
    return canonical_title_for_exact_match(expected) == canonical_title_for_exact_match(candidate)


def filter_by_date(ref: Dict, min_date: datetime, max_date: datetime) -> bool:
    pub_str = ref.get('publicationDate')
    if pub_str:
        try:
            pub_date = datetime.strptime(pub_str, '%Y-%m-%d')
            return min_date <= pub_date <= max_date
        except ValueError:
            pass
    year = ref.get('year')
    if isinstance(year, int) and year > 0:
        ref_date = datetime(year, 1, 1)
        return min_date <= ref_date <= max_date
    return False

def get_existing_refs(paper_dir: Path) -> Dict[str, Path]:
    existing = {}
    for pdf_path in paper_dir.glob("REF_*.pdf"):
        if pdf_path.is_file():
            match = re.match(r"REF_\d+_(.*)\.pdf", pdf_path.name)
            title_part = match.group(1) if match else pdf_path.stem.replace("REF_","").lstrip("0123456789_")
            key = clean_filename(title_part)
            existing[key] = pdf_path
    return existing

def count_existing_pdfs(paper_dir: Path) -> Tuple[int, int, int]:
    """
    Returns:
    - main_count
    - ref_count
    - total_count
    """
    main_count = len([p for p in paper_dir.glob("MAIN_*.pdf") if p.is_file()])
    ref_count = len([p for p in paper_dir.glob("REF_*.pdf") if p.is_file()])
    return main_count, ref_count, main_count + ref_count

def is_paper_database_complete(paper_dir: Path, target_total: int = DEFAULT_MAX_TOTAL) -> bool:
    main_count, ref_count, total_count = count_existing_pdfs(paper_dir)
    print(f"[INFO] Existing database stats: MAIN={main_count}, REF={ref_count}, TOTAL={total_count}, TARGET={target_total}")
    return main_count >= 1 and total_count >= target_total

def find_main_pdf_path(paper_dir: Path) -> Optional[Path]:
    for pdf in paper_dir.glob("MAIN_*.pdf"):
        if pdf.is_file():
            return pdf
    return None

def find_main_dir(main: Dict, root_dir: Path) -> Optional[Path]:
    search_keys = {main.get("arxiv_id_raw"), main.get("arxiv_id"), clean_filename(main.get("title", ""))}
    search_keys.discard(None)
    search_keys.discard("")
    for paper_dir in root_dir.iterdir():
        if paper_dir.is_dir():
            main_pdf = next(paper_dir.glob("MAIN_*.pdf"), None)
            if main_pdf:
                for key in search_keys:
                    if key in main_pdf.name or key in paper_dir.name:
                        return paper_dir
    return None

class ArxivSearcher:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (ArxivSearcher/0.1)"})

    def search(self, arxiv_id: Optional[str] = None, title: Optional[str] = None, strict_title: bool = False) -> Optional[Dict]:
        if arxiv_id:
            found = self._search_by_id(arxiv_id)
            if found:
                return found
        if title:
            return self._search_by_title(title, strict=strict_title)
        return None

    def _search_by_id(self, aid: str) -> Optional[Dict]:
        if ARXIV_LIB_AVAILABLE:
            try:
                search = arxiv.Search(id_list=[aid.strip()], max_results=1)
                res = next(search.results(), None)
                if res:
                    return {"arxiv_id": res.entry_id.split("/")[-1], "title": res.title.strip(), "pdf_url": res.pdf_url}
            except Exception:
                pass
        return self._http_arxiv_query(f"id:{aid}")

    def _search_by_title(self, title: str, strict: bool = False) -> Optional[Dict]:
        title = str(title or "").strip()
        if not title:
            return None
        if ARXIV_LIB_AVAILABLE:
            try:
                max_results = 8 if strict else 1
                search = arxiv.Search(query=f'ti:"{title}"', max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
                results = list(search.results())
                if results:
                    if not strict:
                        res = results[0]
                        return {"arxiv_id": res.entry_id.split("/")[-1], "title": res.title.strip(), "pdf_url": res.pdf_url}
                    for res in results:
                        candidate = {"arxiv_id": res.entry_id.split("/")[-1], "title": res.title.strip(), "pdf_url": res.pdf_url}
                        matched = is_title_match(title, candidate["title"])
                        print(f"[INFO] arXiv title candidate exact_match={matched}: {candidate['title']}")
                        if matched:
                            return candidate
            except Exception:
                pass
        return self._http_arxiv_query(f'ti:"{title}"', expected_title=title if strict else None, max_results=8 if strict else 1)

    def _http_arxiv_query(self, query: str, expected_title: Optional[str] = None, max_results: int = 1) -> Optional[Dict]:
        url = "http://export.arxiv.org/api/query"
        params = {"search_query": query, "start": 0, "max_results": max_results}
        try:
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code != 200:
                return None
            text = r.text
            if "<entry>" not in text:
                return None
            entries = text.split("<entry>")[1:]
            for entry in entries:
                entry = entry.split("</entry>", 1)[0]
                m_id = re.search(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", entry)
                if not m_id:
                    continue
                pid = m_id.group(1).strip()
                m_title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                found_title = re.sub(r"\s+", " ", m_title.group(1).strip()) if m_title else pid
                if expected_title is not None:
                    matched = is_title_match(expected_title, found_title)
                    print(f"[INFO] arXiv title candidate exact_match={matched}: {found_title}")
                    if not matched:
                        continue
                m_pdf = re.search(r'<link[^>]+href="(https?://arxiv\.org/pdf/[^"]+)"', entry)
                pdf_url = m_pdf.group(1) if m_pdf else f"https://arxiv.org/pdf/{pid}.pdf"
                return {"arxiv_id": pid, "title": found_title, "pdf_url": pdf_url}
            return None
        except Exception:
            return None

class PaperDownloader:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (RefPDFDownloader/0.1)"})
        self.arxiv = ArxivSearcher(session=self.session)

    def download_pdf(self, paper: Dict, save_path: Path) -> bool:
        title = paper.get("title", "").strip()
        print(f"  [Download attempt]: {clean_filename(title)}")
        open_access = paper.get("openAccessPdf")
        if isinstance(open_access, dict):
            pdf_url = open_access.get("url")
            if pdf_url and pdf_url.strip() and self._download_file(pdf_url, save_path):
                print(f"  [OK] Downloaded from JSON openAccessPdf")
                return True

        arxiv_paper = self.arxiv.search(arxiv_id=paper.get("arxiv_id"), title=title, strict_title=True)
        if arxiv_paper and self._download_file(arxiv_paper.get("pdf_url"), save_path):
            print(f"  [OK] Downloaded from arXiv")
            return True

        print(f"  [FAIL] Download failed: {clean_filename(title)}")
        if save_path.exists() and save_path.stat().st_size < 1024:
            try:
                save_path.unlink()
            except Exception:
                pass
        return False

    def _download_file(self, url: str, path: Path, max_retries: int = 3) -> bool:
        if not url:
            return False
        for attempt in range(max_retries):
            try:
                r = self.session.get(url, timeout=60, stream=True, allow_redirects=True)
                r.raise_for_status()
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                    if path.stat().st_size > 1024:
                        return True
            except Exception as e:
                print(f"    [WARN] Download exception (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("    Waiting 2 seconds before retry...")
                    time.sleep(2)
                else:
                    print(f"    [ERROR] Max retries ({max_retries}) reached, giving up on this link.")

        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        return False

class ReferenceHierarchyBuilder:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.last_semantic_request = 0

    def wait_for_semantic_scholar_rate_limit(self, is_batch=False):
        min_interval = 5.0 if is_batch else 3.1
        elapsed = time.time() - self.last_semantic_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_semantic_request = time.time()

    @staticmethod
    def strip_arxiv_version(arxiv_id: str) -> str:
        if not arxiv_id:
            return arxiv_id
        a = arxiv_id.replace('arXiv:', '').strip()
        m = re.match(r'^(\d{4}\.\d{4,5})', a)
        return m.group(1) if m else a

    @staticmethod
    def is_s2_id(identifier: str) -> bool:
        return isinstance(identifier, str) and len(identifier) == 40 and all(c in "0123456789abcdef" for c in identifier.lower())

    @staticmethod
    def canonicalize_ref_arxiv_id(aid: Optional[str]) -> Optional[str]:
        if not aid:
            return aid
        return aid.replace('arXiv:', '').split('v')[0]

    def get_sort_key(self, ref: Dict):
        pub_str = ref.get('publicationDate')
        if pub_str:
            try:
                pub_date = datetime.strptime(pub_str, '%Y-%m-%d')
                return -pub_date.timestamp()
            except ValueError:
                pass
        year = ref.get('year')
        return -(year if year is not None else 0)

    def normalize_identifier(self, identifier: str) -> Dict:
        identifier = (identifier or "").strip()
        candidates = []
        base_arxiv = None
        if self.is_s2_id(identifier):
            candidates.append(identifier)
        else:
            raw = identifier.replace('arXiv:', '')
            base = self.strip_arxiv_version(raw)
            base_arxiv = base
            candidates.append(f"arXiv:{base}")
            if 'v' in raw and raw != base:
                candidates.append(f"arXiv:{raw}")
        uniq = []
        seen = set()
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return {'original': identifier, 'fetch_candidates': uniq, 'base_arxiv_id': base_arxiv}

    def get_references_from_semantic_scholar(self, identifier: str, max_retries=3) -> Optional[Dict]:
        norm = self.normalize_identifier(identifier)
        for fetch_id in norm['fetch_candidates']:
            for attempt in range(max_retries):
                try:
                    self.wait_for_semantic_scholar_rate_limit()
                    url = f"https://api.semanticscholar.org/graph/v1/paper/{fetch_id}"
                    params = {
                        'fields': 'paperId,externalIds,openAccessPdf,references.paperId,references.title,'
                                  'references.year,references.publicationDate,references.externalIds,references.openAccessPdf'
                    }
                    resp = requests.get(url, params=params, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        refs_all = data.get('references', []) or []
                        ref_list = []
                        for ref in refs_all:
                            title = (ref.get('title') or "").strip()
                            if not title:
                                continue
                            exids = ref.get('externalIds') or {}
                            ref_list.append({
                                'title': title,
                                'arxiv_id': self.canonicalize_ref_arxiv_id(exids.get('ArXiv')),
                                's2_id': ref.get('paperId'),
                                'year': ref.get('year'),
                                'publicationDate': ref.get('publicationDate'),
                                'openAccessPdf': ref.get('openAccessPdf')
                            })
                        return {
                            'paper_s2_id': data.get('paperId'),
                            'paper_arxiv_id': self.canonicalize_ref_arxiv_id((data.get('externalIds') or {}).get('ArXiv')),
                            'openAccessPdf': data.get('openAccessPdf'),
                            'references': ref_list
                        }
                    elif resp.status_code == 404:
                        break
                    elif resp.status_code == 429:
                        time.sleep(1)
                        continue
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(8)
                            continue
                        break
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(8)
                        continue
                    break
        return None

    def get_batch_references_from_semantic_scholar(self, identifiers: List[str], max_retries=3, batch_size=50) -> Dict[str, Optional[List[Dict]]]:
        results: Dict[str, Optional[List[Dict]]] = {}
        cleaned = []
        seen = set()
        for i in identifiers:
            if i and i not in seen:
                cleaned.append(i)
                seen.add(i)
        if not cleaned:
            return results

        batches = [cleaned[i:i + batch_size] for i in range(0, len(cleaned), batch_size)]
        for batch in batches:
            for attempt in range(max_retries):
                try:
                    self.wait_for_semantic_scholar_rate_limit(is_batch=True)
                    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
                    params = {
                        'fields': 'paperId,openAccessPdf,references.paperId,references.title,'
                                  'references.year,references.publicationDate,references.externalIds,references.openAccessPdf'
                    }
                    resp = requests.post(url, params=params, json={'ids': batch}, timeout=60)
                    if resp.status_code == 200:
                        data_list = resp.json()
                        for idx, data in enumerate(data_list):
                            fid = batch[idx]
                            if not data or 'references' not in data:
                                results[fid] = None
                                continue
                            refs_all = data.get('references', []) or []
                            ref_list = []
                            for ref in refs_all:
                                title = (ref.get('title') or "").strip()
                                if not title:
                                    continue
                                exids = ref.get('externalIds') or {}
                                ref_list.append({
                                    'title': title,
                                    'arxiv_id': self.canonicalize_ref_arxiv_id(exids.get('ArXiv')),
                                    's2_id': ref.get('paperId'),
                                    'year': ref.get('year'),
                                    'publicationDate': ref.get('publicationDate'),
                                    'openAccessPdf': ref.get('openAccessPdf')
                                })
                            results[fid] = ref_list
                        break
                    elif resp.status_code == 429:
                        time.sleep(10)
                        continue
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(8)
                            continue
                        for fid in batch:
                            results[fid] = None
                        break
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(8)
                        continue
                    for fid in batch:
                        results[fid] = None
                    break
        return results

    def process_single_input(self, query: str, searcher: ArxivSearcher) -> Optional[Dict]:
        print(f"\n[INFO] Searching for paper info: {query}")
        paper_info = None
        if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', query) or query.lower().startswith('arxiv:'):
            aid = query.replace('arxiv:', '', 1).strip()
            paper_info = searcher.search(arxiv_id=aid)
        else:
            paper_info = searcher.search(title=query, strict_title=True)

        if not paper_info:
            print("[ERROR] Could not find this paper on arXiv. Please check the input.")
            return None

        main_title = paper_info['title']
        raw_arxiv_id = paper_info['arxiv_id']
        year = datetime.now().year
        print(f"[OK] Found main paper: {main_title} (arXiv: {raw_arxiv_id})")

        base_arxiv_id = self.strip_arxiv_version(raw_arxiv_id)
        main_result = self.get_references_from_semantic_scholar(base_arxiv_id)

        if not main_result:
            print("[WARN] Could not fetch level-1 references, aborting.")
            return None

        main_s2_id = main_result.get('paper_s2_id')
        level1_all = main_result.get('references', [])

        level1_fetch_ids: List[str] = []
        for ref in level1_all:
            if ref.get('arxiv_id'):
                level1_fetch_ids.append(f"arXiv:{ref['arxiv_id']}")
            elif ref.get('s2_id'):
                level1_fetch_ids.append(ref['s2_id'])

        if not level1_fetch_ids:
            sorted_level2 = []
        else:
            batch_results = self.get_batch_references_from_semantic_scholar(level1_fetch_ids)
            title_to_count = defaultdict(int)
            title_to_info: Dict[str, Dict] = {}
            for fid, refs in batch_results.items():
                if refs:
                    for r in refs:
                        t = r['title']
                        title_to_count[t] += 1
                        title_to_info[t] = r
            sorted_level2 = sorted(
                title_to_info.values(),
                key=lambda r: (-title_to_count[r['title']], self.get_sort_key(r))
            )

        return {
            'main': {
                'title': main_title,
                'year': year,
                'arxiv_id_raw': raw_arxiv_id,
                'arxiv_id': base_arxiv_id,
                's2_id': main_s2_id,
                'openAccessPdf': main_result.get('openAccessPdf')
            },
            'level1': level1_all,
            'level2': sorted_level2
        }

def supplement_paper_database(main_title: str, data: Dict, downloader: PaperDownloader,
                              paper_dir: Path, max_total: int = DEFAULT_MAX_TOTAL):
    """
    Incrementally fill the database:
    - Preserve existing MAIN / REF files
    - Skip already-downloaded papers
    - Continue downloading until max_total is reached
    - Do not delete old REF files
    - Level-2 refs are traversed in sorted order (citation frequency first, then recency)
    """
    main, level1, level2 = data["main"], data["level1"], data["level2"]

    existing_refs_cache = get_existing_refs(paper_dir)
    existing_keys = set(existing_refs_cache.keys())

    main_count, ref_count, total_count = count_existing_pdfs(paper_dir)
    print(f"[INFO] Current local database before supplement: MAIN={main_count}, REF={ref_count}, TOTAL={total_count}")

    if total_count >= max_total and main_count >= 1:
        print(f"[OK] Database already complete, no need to supplement.")
        return

    # Find the current maximum REF index for continued numbering
    existing_ref_indices = []
    for pdf_path in paper_dir.glob("REF_*.pdf"):
        match = re.match(r"REF_(\d+)_(.*)\.pdf", pdf_path.name)
        if match:
            try:
                existing_ref_indices.append(int(match.group(1)))
            except Exception:
                pass
    next_ref_index = max(existing_ref_indices, default=0) + 1

    processed_keys = set(existing_keys)

    # Supplement level-1 references first
    for paper in level1:
        main_count, ref_count, total_count = count_existing_pdfs(paper_dir)
        if total_count >= max_total:
            break

        key = clean_filename(paper.get("title", ""))
        if not key or key in processed_keys:
            continue

        final_path = paper_dir / f"REF_{next_ref_index:03d}_{key}.pdf"
        if downloader.download_pdf(paper, final_path):
            print(f"  [OK] Added REF: {final_path.name}")
            next_ref_index += 1
        processed_keys.add(key)

    # Supplement level-2 references in sorted order (duplicate count first, recency second).
    if count_existing_pdfs(paper_dir)[2] < max_total:
        for paper in level2:
            if count_existing_pdfs(paper_dir)[2] >= max_total:
                break

            key = clean_filename(paper.get("title", ""))
            if not key or key in processed_keys:
                continue

            final_path = paper_dir / f"REF_{next_ref_index:03d}_{key}.pdf"
            if downloader.download_pdf(paper, final_path):
                print(f"  [OK] Added REF: {final_path.name}")
                next_ref_index += 1
            processed_keys.add(key)

    main_count, ref_count, total_count = count_existing_pdfs(paper_dir)
    print(f"[INFO] Database after supplement: MAIN={main_count}, REF={ref_count}, TOTAL={total_count}")

def process_main_paper(main_title: str, data: Dict, downloader: PaperDownloader,
                       paper_dir: Path, max_total: int = DEFAULT_MAX_TOTAL):
    """
    Used for first-time database creation; delegates to the incremental
    supplement logic for backward compatibility.
    """
    supplement_paper_database(main_title, data, downloader, paper_dir, max_total)

def download_paper_if_needed(config: Dict) -> Optional[str]:
    """
    Check if a local paper folder already exists:
    - If absent: create and download
    - If present and complete: reuse directly
    - If present but incomplete: continue downloading until max_total
    Returns the absolute path to the paper folder.
    """
    query = config.get("paper_name", "").strip()
    if not query:
        print("[ERROR] No paper_name provided in config.")
        return None

    max_total = config.get("max_total_papers", DEFAULT_MAX_TOTAL)

    root_dir = Path(config['paths']['database_dir'])
    root_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Checking local database for paper: {query}")
    print(f"[INFO] Target total papers: {max_total}")

    norm_query = _normalize_for_match(query)
    matched_paper_dir = None

    # Iterate local database; use normalized prefix matching to find existing folders
    for paper_dir in root_dir.iterdir():
        if paper_dir.is_dir():
            norm_dir = _normalize_for_match(paper_dir.name)
            if len(norm_dir) < 15:
                continue
            if norm_query.startswith(norm_dir) or norm_dir.startswith(norm_query):
                matched_paper_dir = paper_dir
                break

    downloader = PaperDownloader()
    builder = ReferenceHierarchyBuilder()

    # Case 1: local directory already exists
    if matched_paper_dir:
        print(f"[OK] Found existing paper folder (Local Real Path): {matched_paper_dir}")

        if is_paper_database_complete(matched_paper_dir, max_total):
            print("[OK] Existing local database is complete. Reusing it directly.")
            return str(matched_paper_dir.resolve())

        print("[WARN] Existing local database is incomplete. Start supplementing...")

        data = builder.process_single_input(query, downloader.arxiv)
        if not data:
            print("[ERROR] Failed to fetch paper metadata for supplement.")
            return str(matched_paper_dir.resolve())

        # If MAIN pdf is missing, try to download it
        main_pdf = find_main_pdf_path(matched_paper_dir)
        if not main_pdf:
            main_title = data['main']['title']
            safe_title = clean_filename(main_title)[:100]
            main_pdf_path = matched_paper_dir / f"MAIN_{safe_title}.pdf"
            print(f"\n[INFO] MAIN paper missing, downloading to: {main_pdf_path}")
            main_download_info = {
                "title": main_title,
                "arxiv_id": data['main']['arxiv_id_raw'],
                "openAccessPdf": data['main']['openAccessPdf']
            }
            success = downloader.download_pdf(main_download_info, main_pdf_path)
            if not success:
                print("[WARN] Main paper download failed, but continuing with reference supplement.")

        print(f"\n{'='*30} Supplementing References {'='*30}")
        supplement_paper_database(data['main']['title'], data, downloader, matched_paper_dir, max_total)

        return str(matched_paper_dir.resolve())

    # Case 2: not found locally, perform full download
    print(f"\n[INFO] Paper not found locally. Starting download...")

    data = builder.process_single_input(query, downloader.arxiv)
    if not data:
        return None

    main_title = data['main']['title']
    safe_title = clean_filename(main_title)[:100]
    paper_dir = root_dir / safe_title
    paper_dir.mkdir(parents=True, exist_ok=True)

    main_pdf_path = paper_dir / f"MAIN_{safe_title}.pdf"
    print(f"\n[INFO] Downloading main paper to: {main_pdf_path}")

    main_download_info = {
        "title": main_title,
        "arxiv_id": data['main']['arxiv_id_raw'],
        "openAccessPdf": data['main']['openAccessPdf']
    }
    success = downloader.download_pdf(main_download_info, main_pdf_path)
    if not success:
        print("[WARN] Main paper download failed, but continuing with references.")

    print(f"\n{'='*30} Downloading References {'='*30}")
    supplement_paper_database(main_title, data, downloader, paper_dir, max_total)

    print("\n[OK] Paper crawling and database construction completed!")

    return str(paper_dir.resolve())