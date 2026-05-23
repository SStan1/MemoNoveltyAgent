import os
import openai
import traceback
import time
import fitz

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text")
        doc.close()
        print(f"[OK] Successfully extracted {len(full_text)} characters from {os.path.basename(pdf_path)}")
        return full_text
    except Exception as e:
        print(f"[ERROR] Error extracting text from PDF {pdf_path}: {e}")
        traceback.print_exc()
        return None

def get_paper_innovation(config, paper_name, main_pdf_path):
    try:
        print(f"[INFO] Starting innovation extraction for paper: {paper_name}")
        
        client = openai.OpenAI(
            api_key=config['api']['openai_api_key'],
            base_url=config['api']['openai_base_url'],
            timeout=config['api']['openai_timeout']
        )
        
        print(f"[INFO] Extracting full text from '{os.path.basename(main_pdf_path)}'...")
        paper_text = extract_text_from_pdf(main_pdf_path)
        
        if not paper_text:
            print("[ERROR] Failed to extract text from PDF. Aborting innovation extraction.")
            return None
            
        system_prompt = config['prompts']['innovation_extraction']['system_prompt']
        user_prompt = config['prompts']['innovation_extraction']['user_prompt'].format(
            paper_name=paper_name,
            paper_text=paper_text
        )
        
        max_retries = config['llm_config']['max_retries']
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Calling LLM for innovation extraction. Attempt {attempt + 1}/{max_retries}...")
                response = client.chat.completions.create(
                    model=config['llm_config']['model'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=config['llm_config']['temperature'],
                    stream=False
                )
                
                if response and response.choices and len(response.choices) > 0:
                    full_response = response.choices[0].message.content
                    if full_response and full_response.strip():
                        print("[OK] LLM response received successfully.")
                        return full_response
                
                print(f"[WARN] Received an empty or invalid response on attempt {attempt + 1}. Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                
            except Exception as e:
                print(f"[ERROR] LLM API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        print(f"[ERROR] All {max_retries} retry attempts failed. Could not extract innovations.")
        return None
        
    except Exception as e:
        print(f"[ERROR] A fatal error occurred in get_paper_innovation: {e}")
        traceback.print_exc()
        return None