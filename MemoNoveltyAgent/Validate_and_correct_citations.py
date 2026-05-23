import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import PyPDF2
import time


class CitationValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm_config', {})
        self.prompts = config.get('prompts', {}).get('validation', {})
        self._model_family = self._detect_model_family()
        self._confirmed_low_effort = self._is_confirmed_low_effort_model()
        self._low_effort_supported = True

        model_name = self.llm_config.get('model', '')
        if self._confirmed_low_effort:
            print(f"[CitationValidator] Model '{model_name}' (family: {self._model_family}) "
                  f"— confirmed low-effort support ✓")
        else:
            print(f"[CitationValidator] Model '{model_name}' (family: {self._model_family}) "
                  f"— will attempt low-effort (not guaranteed, may fall back)")
        print(f"[CitationValidator] All validation stages will attempt LOW reasoning effort")

    # ------------------------------------------------------------------
    #  Model family detection & low-effort support check
    # ------------------------------------------------------------------

    def _detect_model_family(self) -> str:
        """
        Detect model family from the configured model name.
          - 'gemini' or 'gemma' in name  → 'gemini'
          - 'gpt' in name or o-series    → 'openai'
          - 'claude' in name             → 'claude'
          - 'deepseek' in name           → 'deepseek'
          - 'qwen' in name              → 'qwen'
          - otherwise                    → 'unknown'
        """
        model = (self.llm_config.get('model', '') or '').lower().strip()
        if 'gemini' in model or 'gemma' in model:
            return 'gemini'
        if 'gpt' in model or re.search(r'\bo[1-9]', model):
            return 'openai'
        if 'claude' in model:
            return 'claude'
        if 'deepseek' in model:
            return 'deepseek'
        if 'qwen' in model:
            return 'qwen'
        return 'unknown'

    def _is_confirmed_low_effort_model(self) -> bool:
        """
        Returns True for model families KNOWN to support reasoning_effort='low':
          - gemini
          - openai (GPT / o-series)
        Other models will be attempted but are not guaranteed.
        """
        return self._model_family in ('gemini', 'openai')

    # ------------------------------------------------------------------
    #  Core helper: attempt model call with low effort
    # ------------------------------------------------------------------

    def _try_create_with_low_effort(self, client, model, messages, temperature, stream=False):
        """
        Try to call the model with reasoning_effort='low'.
        If the parameter is rejected, disable low-effort for future calls
        and fall back to default.
        """
        if not self._low_effort_supported:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
                extra_body={"reasoning_effort": "low"},
            )
            return response
        except Exception as e:
            error_msg = str(e).lower()
            is_param_error = any(kw in error_msg for kw in [
                "reasoning_effort", "extra_body", "unrecognized",
                "invalid", "not supported", "unknown parameter",
                "unexpected keyword", "additional properties",
            ])
            if not is_param_error:
                raise

            self._low_effort_supported = False
            print(f"[WARNING] Low-effort reasoning not supported by '{model}', falling back to default.")

            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )

    # ------------------------------------------------------------------
    #  Main model call dispatcher
    # ------------------------------------------------------------------

    def call_model(self, prompt: str, stage: str = "") -> str:
        model = self.llm_config.get('model', 'gpt-5-mini')
        temperature = self.llm_config.get('temperature', 0.3)
        use_openai = self.llm_config.get('use_openai', True)
        timeout = self.config['api'].get('openai_timeout', 600.0)
        max_retries = self.llm_config.get('max_retries', 5)
        retry_delay = self.llm_config.get('retry_delay', 3)
        use_stream = self.llm_config.get('use_stream', True)

        is_extract_stage = (stage == "extract_citations")

        effort_label = "low-effort" if self._low_effort_supported else "default-effort"

        for attempt in range(1, max_retries + 1):
            request_start = time.time()

            try:
                if is_extract_stage:
                    confirmed_str = "confirmed" if self._confirmed_low_effort else "best-effort"
                    print(f"\n[extract_citations] LLM attempt {attempt}/{max_retries} started")
                    print(f"[extract_citations] prompt length: {len(prompt)} chars")
                    print(f"[extract_citations] model={model} (family={self._model_family}, "
                          f"low_effort={effort_label} [{confirmed_str}]), "
                          f"timeout={timeout}s, stream={use_stream}")

                if use_openai:
                    import openai
                    client = openai.OpenAI(
                        api_key=self.config['api']['openai_api_key'],
                        base_url=self.config['api']['openai_base_url'],
                        timeout=timeout,
                        max_retries=0
                    )

                    messages = [{"role": "user", "content": prompt}]

                    if use_stream:
                        # ========== Streaming call ==========
                        if is_extract_stage:
                            print(f"[extract_citations] sending STREAM request ({effort_label})...")
                            print("=" * 60)
                            print("[STREAM OUTPUT START]")
                            print("=" * 60)

                        response_stream = self._try_create_with_low_effort(
                            client, model, messages, temperature, stream=True
                        )

                        collected_content = ""
                        token_count = 0
                        last_print_time = time.time()

                        for chunk in response_stream:
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if delta and delta.content:
                                    piece = delta.content
                                    collected_content += piece
                                    token_count += 1

                                    print(piece, end="", flush=True)

                                    now = time.time()
                                    if now - last_print_time > 10:
                                        elapsed = now - request_start
                                        print(f"\n  ... [{elapsed:.0f}s elapsed, ~{token_count} chunks received] ...", flush=True)
                                        last_print_time = now

                        elapsed = time.time() - request_start

                        print()
                        print("=" * 60)
                        print(f"[STREAM OUTPUT END]")
                        print(f"[{stage or 'call_model'}] completed in {elapsed:.2f}s, "
                              f"~{token_count} chunks, {len(collected_content)} chars")
                        print("=" * 60)

                        return collected_content or ""

                    else:
                        # ========== Non-streaming call ==========
                        if is_extract_stage:
                            print(f"[extract_citations] sending request ({effort_label})...")

                        response = self._try_create_with_low_effort(
                            client, model, messages, temperature, stream=False
                        )

                        content = response.choices[0].message.content
                        elapsed = time.time() - request_start

                        if is_extract_stage:
                            print(f"[extract_citations] succeeded in {elapsed:.2f}s, "
                                  f"response length: {len(content) if content else 0} chars")

                        return content or ""

                else:
                    # ========== requests-based call ==========
                    import requests as req_lib
                    headers = {
                        "Authorization": f"Bearer {self.config['api']['api_key']}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "stream": use_stream
                    }

                    added_low_effort = False
                    if self._low_effort_supported:
                        data["reasoning_effort"] = "low"
                        added_low_effort = True

                    if is_extract_stage:
                        print(f"[extract_citations] sending {'STREAM' if use_stream else ''} "
                              f"request via requests.post ({effort_label})...")

                    response = req_lib.post(
                        f"{self.config['api']['base_url']}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=timeout,
                        stream=use_stream
                    )

                    if response.status_code != 200:
                        error_text = response.text[:500]
                        if added_low_effort and any(
                            kw in error_text.lower() for kw in [
                                "reasoning_effort", "unknown", "invalid",
                                "not supported", "additional properties",
                            ]
                        ):
                            self._low_effort_supported = False
                            print(f"[WARNING] Low-effort reasoning not supported by '{model}', falling back to default.")

                            data.pop("reasoning_effort", None)
                            response = req_lib.post(
                                f"{self.config['api']['base_url']}/chat/completions",
                                headers=headers,
                                json=data,
                                timeout=timeout,
                                stream=use_stream
                            )
                            if response.status_code != 200:
                                raise Exception(
                                    f"API returned status {response.status_code}: {response.text[:200]}"
                                )
                        else:
                            raise Exception(
                                f"API returned status {response.status_code}: {error_text}"
                            )

                    if use_stream:
                        if is_extract_stage:
                            print("=" * 60)
                            print("[STREAM OUTPUT START]")
                            print("=" * 60)

                        collected_content = ""
                        token_count = 0
                        last_print_time = time.time()

                        for line in response.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            if line.startswith("data: "):
                                line_data = line[6:]
                                if line_data.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(line_data)
                                    if chunk.get("choices") and len(chunk["choices"]) > 0:
                                        delta = chunk["choices"][0].get("delta", {})
                                        piece = delta.get("content", "")
                                        if piece:
                                            collected_content += piece
                                            token_count += 1
                                            print(piece, end="", flush=True)

                                            now = time.time()
                                            if now - last_print_time > 10:
                                                elapsed = now - request_start
                                                print(f"\n  ... [{elapsed:.0f}s elapsed, "
                                                      f"~{token_count} chunks] ...", flush=True)
                                                last_print_time = now
                                except json.JSONDecodeError:
                                    continue

                        elapsed = time.time() - request_start
                        print()
                        print("=" * 60)
                        print(f"[STREAM OUTPUT END]")
                        print(f"[{stage or 'call_model'}] completed in {elapsed:.2f}s, "
                              f"~{token_count} chunks, {len(collected_content)} chars")
                        print("=" * 60)

                        return collected_content or ""

                    else:
                        content = response.json()['choices'][0]['message']['content']
                        elapsed = time.time() - request_start
                        if is_extract_stage:
                            print(f"[extract_citations] succeeded in {elapsed:.2f}s")
                        return content or ""

            except Exception as e:
                elapsed = time.time() - request_start if 'request_start' in locals() else 0
                print(f"\n[{stage or 'call_model'}] attempt {attempt} failed after {elapsed:.2f}s")
                print(f"[{stage or 'call_model'}] error: {repr(e)}")

                if attempt >= max_retries:
                    print(f"[{stage or 'call_model'}] max retries reached, returning empty string")
                    return ""

                print(f"[{stage or 'call_model'}] retrying after {retry_delay}s...\n")
                time.sleep(retry_delay)

        return ""

    # ------------------------------------------------------------------
    #  Static utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def clean_escape(input_text: str) -> str:
        return input_text.replace("\\>", ">").replace("\\<", "<").replace("\\+", "+").replace("\\~", "~")

    @staticmethod
    def extract_pdf_text(pdf_path: str, max_pages: Optional[int] = None) -> str:
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                pages_to_read = min(total_pages, max_pages) if max_pages else total_pages
                for i in range(pages_to_read):
                    text += pdf_reader.pages[i].extract_text() + "\n"
            return text
        except Exception:
            return ""

    @staticmethod
    def find_pdf_by_name(reference_name: str, base_dir: str) -> Optional[str]:
        clean_name = reference_name.replace('##', '').replace('$$', '').strip()
        clean_name = re.sub(r'^\[\d+\]\s*', '', clean_name).strip()
        if clean_name.endswith('.pdf'):
            clean_name = clean_name[:-4].strip()

        base_path = Path(base_dir)
        if not base_path.exists():
            return None

        for pdf_file in base_path.rglob("*.pdf"):
            if clean_name == pdf_file.name or clean_name == pdf_file.stem or clean_name.lower() == pdf_file.stem.lower():
                return str(pdf_file)

        clean_name_without_prefix = re.sub(r'^REF_\d+_', '', clean_name, flags=re.IGNORECASE).strip()
        clean_name_without_prefix = re.sub(r'\s+', '_', clean_name_without_prefix)
        clean_name_without_prefix = re.sub(r'[-\u2013\u2014]', '_', clean_name_without_prefix)

        for pdf_file in base_path.rglob("*.pdf"):
            pdf_name_without_prefix = re.sub(r'^REF_\d+_', '', pdf_file.stem, flags=re.IGNORECASE).strip()
            pdf_name_without_prefix = re.sub(r'\s+', '_', pdf_name_without_prefix)
            pdf_name_without_prefix = re.sub(r'[-\u2013\u2014]', '_', pdf_name_without_prefix)
            if clean_name_without_prefix.lower() == pdf_name_without_prefix.lower():
                return str(pdf_file)
        return None

    @staticmethod
    def is_database_reference(reference_name: str) -> bool:
        """
        Only REF-prefixed RAG/database references are self-validated.
        Expert-memory paper-title citations are intentionally kept for source
        attribution but skipped here because they are not RAG chunks.
        """
        clean_name = (reference_name or '').replace('##', '').replace('$$', '').strip()
        clean_name = re.sub(r'^\[\d+\]\s*', '', clean_name).strip()
        return bool(re.search(r'\bREF_\d+_', clean_name, flags=re.IGNORECASE))

    # ------------------------------------------------------------------
    #  Pipeline steps
    # ------------------------------------------------------------------

    def extract_citations(self, article: str) -> List[Dict[str, str]]:
        print("[extract_citations] preparing extraction prompt...")
        prompt = self.prompts['extraction_prompt'].replace('{report_text}', article)
        print(f"[extract_citations] article length: {len(article)} chars")

        total_start = time.time()
        response = self.call_model(prompt, stage="extract_citations")
        total_elapsed = time.time() - total_start

        print(f"[extract_citations] total model phase time: {total_elapsed:.2f}s")

        if response:
            try:
                cleaned = self.clean_escape(
                    response.replace("```json", "").replace("```", "").strip()
                )
                citations = json.loads(cleaned)
                if not isinstance(citations, list):
                    citations = []
                database_citations = [
                    citation for citation in citations
                    if self.is_database_reference(citation.get('reference_name', ''))
                ]
                skipped = len(citations) - len(database_citations)
                print(
                    f"[extract_citations] JSON parse success, extracted {len(citations)} citations; "
                    f"kept {len(database_citations)} REF/database citations for validation, "
                    f"skipped {skipped} expert-memory/non-REF citations"
                )
                return database_citations
            except Exception as e:
                print(f"[extract_citations] JSON parse failed: {repr(e)}")
                preview = response[:500] if response else ""
                print(f"[extract_citations] response preview:\n{preview}\n")
        else:
            print("[extract_citations] empty response from model")

        return []

    def deduplicate_citations(self, citations: List[Dict[str, str]]) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
        citation_groups = {}
        for citation in citations:
            ref_name = citation['reference_name']
            if not self.is_database_reference(ref_name):
                continue
            if ref_name not in citation_groups:
                citation_groups[ref_name] = []
            citation_groups[ref_name].append(citation)

        deduped_groups = {}
        for ref_name, citation_list in citation_groups.items():
            if len(citation_list) == 1:
                deduped_groups[ref_name] = citation_list
                continue

            print(f"  [dedup] {ref_name}: {len(citation_list)} citations, deduplicating...")

            statements = '\n'.join([f'{i+1}. {cit["claim_explanation"]}' for i, cit in enumerate(citation_list)])
            prompt = self.prompts['dedup_prompt'].replace('{statements}', statements)

            try:
                response = self.call_model(prompt, stage="dedup")
                deduped_idx = json.loads(response.replace("```json", "").replace("```", "").strip())
                if not deduped_idx or 0 in deduped_idx or len(deduped_idx) > len(citation_list):
                    deduped_idx = [i+1 for i in range(len(citation_list))]
            except:
                deduped_idx = [i+1 for i in range(len(citation_list))]

            print(f"  [dedup] {ref_name}: kept {len(deduped_idx)}/{len(citation_list)}")
            deduped_groups[ref_name] = [citation_list[i-1] for i in deduped_idx]
        return deduped_groups, {}

    def validate_citations(self, ref_name: str, citation_list: List[Dict[str, str]], pdf_path: str) -> List[Dict[str, Any]]:
        ref_text = self.extract_pdf_text(pdf_path)
        if not ref_text or len(ref_text.strip()) < 100:
            print(f"  [validate] {ref_name}: PDF text too short, skipping")
            return []
        if len(ref_text) > 30000:
            ref_text = ref_text[:30000] + "\n\n[... content truncated due to length ...]"

        print(f"  [validate] {ref_name}: {len(citation_list)} claims vs PDF ({len(ref_text)} chars)")

        claims = '\n'.join([f'{i+1}. {cit["claim_explanation"]}' for i, cit in enumerate(citation_list)])
        prompt = self.prompts['validation_prompt'].replace('{reference_text}', ref_text).replace('{claims}', claims)

        try:
            response = self.call_model(prompt, stage="validate")
            validation_results = json.loads(response.replace("```json", "").replace("```", "").strip())
            for result in validation_results:
                result['idx'] = result['idx'] - 1
            incorrect_count = sum(1 for r in validation_results if r.get('result') == 'incorrect')
            print(f"  [validate] {ref_name}: {incorrect_count} incorrect out of {len(validation_results)}")
            return validation_results
        except Exception as e:
            print(f"  [validate] {ref_name}: parse failed: {repr(e)}")
            return []

    def correct_report(self, original_report: str, validation_results: Dict[str, Any]) -> str:
        corrections_needed = []
        for ref_name, ref_data in validation_results.items():
            citation_list = ref_data.get('citations', [])
            validations = ref_data.get('validation', [])
            for validation in validations:
                if validation.get('result') == 'incorrect':
                    idx = validation.get('idx', 0)
                    if idx < len(citation_list):
                        citation = citation_list[idx]
                        if validation.get('correction', ''):
                            corrections_needed.append({
                                'reference_name': ref_name,
                                'original_statement': citation.get('original_statement', ''),
                                'incorrect_claim': citation.get('claim_explanation', ''),
                                'corrected_claim': validation.get('correction', ''),
                                'reason': validation.get('error_reason', '')
                            })

        if not corrections_needed:
            print("[correct] No corrections needed!")
            return original_report

        print(f"[correct] {len(corrections_needed)} corrections to apply...")

        validation_summary = json.dumps(corrections_needed, ensure_ascii=False, indent=2)
        prompt = self.prompts['correction_prompt'].replace('{original_report}', original_report).replace('{validation_results}', validation_summary)

        corrected_report = self.call_model(prompt, stage="correct")
        if corrected_report and len(corrected_report) > 100:
            return corrected_report
        return original_report

    def validate_and_correct_single_report(self, report_content: str, pdf_base_dir: str) -> str:
        effort_label = "low-effort" if self._low_effort_supported else "default-effort"
        confirmed_str = "confirmed" if self._confirmed_low_effort else "best-effort"

        print(f"\n{'='*60}")
        print(f"[Step 1/4] Extracting citations... "
              f"(model: {self._model_family}, effort: {effort_label} [{confirmed_str}])")
        print(f"{'='*60}")
        citations = self.extract_citations(report_content)
        if not citations:
            print("[Step 1/4] No citations extracted, returning original report.")
            return report_content

        effort_label = "low-effort" if self._low_effort_supported else "default-effort"

        print(f"\n{'='*60}")
        print(f"[Step 2/4] Deduplicating {len(citations)} citations... ({effort_label})")
        print(f"{'='*60}")
        deduped_groups, _ = self.deduplicate_citations(citations)

        total_deduped = sum(len(v) for v in deduped_groups.values())
        print(f"[Step 2/4] After dedup: {total_deduped} citations across {len(deduped_groups)} references")

        effort_label = "low-effort" if self._low_effort_supported else "default-effort"

        print(f"\n{'='*60}")
        print(f"[Step 3/4] Validating citations against local PDFs... ({effort_label})")
        print(f"{'='*60}")
        validation_results = {}
        for ref_name, citation_list in deduped_groups.items():
            pdf_path = self.find_pdf_by_name(ref_name, pdf_base_dir)
            if not pdf_path:
                print(f"  [validate] {ref_name}: PDF not found, skipping")
                continue
            print(f"  [validate] {ref_name}: found PDF -> {pdf_path}")
            validation = self.validate_citations(ref_name, citation_list, pdf_path)
            validation_results[ref_name] = {
                'pdf_path': pdf_path,
                'citations': citation_list,
                'validation': validation
            }

        effort_label = "low-effort" if self._low_effort_supported else "default-effort"

        print(f"\n{'='*60}")
        print(f"[Step 4/4] Correcting report... ({effort_label})")
        print(f"{'='*60}")
        corrected_article = self.correct_report(report_content, validation_results)
        return corrected_article
