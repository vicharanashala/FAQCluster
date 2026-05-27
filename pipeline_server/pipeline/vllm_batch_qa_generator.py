#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vllm_batch_qa_generator.py — Stage 7: FAQ Q&A Generation via vLLM

Generates professional English Q&A pairs from the unique_questions_freq.csv
produced by Stages 1–6, using vLLM offline batch inference with a local
instruction-tuned model (default: Qwen/Qwen2.5-7B-Instruct).

Usage (standalone):
    python pipeline/vllm_batch_qa_generator.py \\
        --input  outputs/repair/maize_makka/unique_questions_freq.csv \\
        --crop   "Maize Makka" \\
        --model  /path/to/qwen2.5-7b-instruct

The script is also importable:
    from pipeline.vllm_batch_qa_generator import run_qa_generation
    run_qa_generation(input_csv, output_csv, crop, model_path, ...)
"""

import os
import pandas as pd
import re
import json
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ══════════════════════════════════════════════════════════════════════════════
# Prompt Engineering
# ══════════════════════════════════════════════════════════════════════════════

def get_system_prompt(crop_name: str, other_crops: list = None) -> str:
    """Build the system prompt for FAQ generation, with crop-specific hints."""

    # ── Other-crop rejection list ────────────────────────────────────────────
    DEFAULT_OTHER_CROPS = [
        "Wheat", "Paddy", "Rice", "Sugarcane", "Potato", "Mustard", "Mango",
        "Black Gram", "Urad", "Mentha", "Mint", "Maize", "Chillies", "Tomato",
        "Pea", "Brinjal", "Green Gram", "Moong", "Sunnhemp", "Bengal Gram",
        "Chickpea", "Chana", "Onion", "Groundnut", "Okra", "Pearl Millet",
        "Bajra", "Guava", "Barseem", "Bottle Gourd", "Lauki", "Guar",
        "Cluster Bean", "Cotton", "Soybean", "Lentil", "Masoor", "Cauliflower",
        "Apple", "Banana", "Arecanut", "Coconut", "Pigeon Pea",
    ]

    # Handle common synonyms to avoid rejecting valid queries for the same crop
    synonyms = {
        "pearl millet": ["pearl millet", "bajra"],
        "bajra": ["pearl millet", "bajra"],
        "bottle gourd": ["bottle gourd", "lauki", "ghiya"],
        "lauki": ["bottle gourd", "lauki", "ghiya"],
        "ghiya": ["bottle gourd", "lauki", "ghiya"],
        "green gram": ["green gram", "moong", "mung"],
        "moong": ["green gram", "moong", "mung"],
        "onion": ["onion", "pyaz", "piaz"],
        "pyaz": ["onion", "pyaz", "piaz"],
        "paddy": ["paddy", "rice", "dhan", "dhaan"],
        "rice": ["paddy", "rice", "dhan", "dhaan"],
        "pigeon pea": ["pigeon pea", "arhar", "tur", "toor"],
        "arhar": ["pigeon pea", "arhar", "tur", "toor"],
        "bengal gram": ["bengal gram", "chickpea", "chana"],
        "chickpea": ["bengal gram", "chickpea", "chana"],
        "chana": ["bengal gram", "chickpea", "chana"],
        "black gram": ["black gram", "urad"],
        "urad": ["black gram", "urad"],
        "maize": ["maize", "makka", "corn"],
        "cotton": ["cotton", "kapas", "narma"],
        "wheat": ["wheat", "gehun", "gehu"],
        "mango": ["mango", "aam"],
        "banana": ["banana", "kela"],
    }

    current_syns = [c.lower() for c in synonyms.get(crop_name.lower(), [crop_name.lower()])]
    reject_list = [c for c in (other_crops or DEFAULT_OTHER_CROPS)
                   if c.lower() not in current_syns]
    reject_str = ", ".join(reject_list)

    # ── Crop-specific expert hints ───────────────────────────────────────────
    cn = crop_name.lower()
    if "sugarcane" in cn or "ganna" in cn:
        hints = (
            "- FOCUS on sugar recovery, brix value, ratoon management, and inter-cropping.\n"
            "- COVER red rot, top borer, termite control, and pyrilla.\n"
            "- INCLUDE trash mulching, earthing-up, and irrigation scheduling."
        )
    elif "potato" in cn or "aloo" in cn:
        hints = (
            "- FOCUS on tuber development, certified seed selection, and cold storage management.\n"
            "- COVER Late Blight, Early Blight, Aphids, and White Grub.\n"
            "- INCLUDE seed treatment, hilling, and nitrogen split-application."
        )
    elif "tomato" in cn or "tamatar" in cn:
        hints = (
            "- FOCUS on fruit setting, staking/trellising, and post-harvest handling.\n"
            "- COVER leaf curl virus, early blight, fruit borer, and whitefly management.\n"
            "- INCLUDE pinching, drip irrigation, and calcium sprays for blossom end rot."
        )
    elif "onion" in cn or "pyaz" in cn:
        hints = (
            "- FOCUS on bulb development, neck-fall timing, and curing/storage.\n"
            "- COVER purple blotch, thrips, basal rot, and damping-off.\n"
            "- INCLUDE transplanting vs. direct sowing, irrigation cutoff before harvest."
        )
    elif "brinjal" in cn or "baingan" in cn or "eggplant" in cn:
        hints = (
            "- FOCUS on fruit and shoot borer as the key pest.\n"
            "- COVER little leaf phytoplasma, bacterial wilt, Cercospora leaf spot.\n"
            "- INCLUDE neem-based sprays, pheromone traps, crop rotation strategy."
        )
    elif "pearl millet" in cn or "bajra" in cn:
        hints = (
            "- FOCUS on downy mildew as the primary disease.\n"
            "- COVER stem borer, shoot fly, ergot, and smut.\n"
            "- INCLUDE Metalaxyl seed treatment, hybrid selection, and kharif sowing windows."
        )
    elif "green gram" in cn or "moong" in cn or "mung" in cn:
        hints = (
            "- FOCUS on mung bean yellow mosaic virus (MYMV), whitefly vector management.\n"
            "- COVER pod borer, cercospora leaf spot, and powdery mildew.\n"
            "- INCLUDE Rhizobium seed inoculation, phosphorus application, and short-duration varieties."
        )
    elif "guar" in cn or "cluster bean" in cn:
        hints = (
            "- FOCUS on guar gum content, pod maturity for vegetable vs. seed crop.\n"
            "- COVER alternaria blight, bacterial blight, and pod gall midge.\n"
            "- INCLUDE drought tolerance management, Rhizobium inoculation, and harvesting at correct stage."
        )
    elif "paddy" in cn or "rice" in cn or "dhan" in cn:
        hints = (
            "- FOCUS on transplanting depth, water management (AWD), and nursery raising.\n"
            "- COVER blast, sheath blight, BPH, and stem borer.\n"
            "- INCLUDE SRI method, zinc application, and basmati vs. non-basmati varieties."
        )
    elif "wheat" in cn or "gehun" in cn:
        hints = (
            "- FOCUS on sowing time (timely vs. late), seed rate, and irrigation scheduling.\n"
            "- COVER yellow rust, Karnal bunt, loose smut, and aphid management.\n"
            "- INCLUDE zero-till technology, nitrogen split doses, and variety selection (PBW, HD, DBW)."
        )
    elif "maize" in cn or "makka" in cn or "corn" in cn:
        hints = (
            "- FOCUS on fall armyworm as the primary emerging pest.\n"
            "- COVER stem borer, turcicum leaf blight, and downy mildew.\n"
            "- INCLUDE hybrid selection, earthing-up, and intercropping with legumes."
        )
    elif "cotton" in cn or "kapas" in cn:
        hints = (
            "- FOCUS on bollworm complex (pink, American, spotted) and Bt resistance management.\n"
            "- COVER sucking pest complex (whitefly, jassid, thrips, mealybug).\n"
            "- INCLUDE refuge planting, defoliant use, and picking schedules."
        )
    elif "mango" in cn or "aam" in cn:
        hints = (
            "- FOCUS on flowering induction, fruit drop management, and post-harvest handling.\n"
            "- COVER anthracnose, powdery mildew, mango hopper, and fruit fly.\n"
            "- INCLUDE paclobutrazol use, pruning, and carbide-free ripening."
        )
    elif "apple" in cn or "seb" in cn:
        hints = (
            "- FOCUS on chilling requirement, high-density planting, and rootstock selection.\n"
            "- COVER scab, canker, woolly aphid, and codling moth.\n"
            "- INCLUDE dormancy-breaking sprays, calcium sprays, and cold storage management."
        )
    elif "banana" in cn or "kela" in cn:
        hints = (
            "- FOCUS on sucker selection, desuckering, and bunch management.\n"
            "- COVER Panama wilt (Fusarium TR4), sigatoka, and banana bunchy top virus.\n"
            "- INCLUDE tissue culture planting, propping, and ripening chambers."
        )
    elif "groundnut" in cn or "peanut" in cn or "moongphali" in cn:
        hints = (
            "- FOCUS on pod development, calcium requirement, and aflatoxin prevention.\n"
            "- COVER tikka disease (leaf spots), stem rot, and white grub.\n"
            "- INCLUDE gypsum application, earthing-up, and harvesting at right maturity."
        )
    elif "coconut" in cn or "nariyal" in cn:
        hints = (
            "- FOCUS on basin management, intercropping, and toddy/copra production.\n"
            "- COVER rhinoceros beetle, red palm weevil, and root wilt.\n"
            "- INCLUDE crown cleaning, husk burial, and balanced fertilisation."
        )
    elif "arecanut" in cn or "areca" in cn or "supari" in cn:
        hints = (
            "- FOCUS on Koleroga (fruit rot), yellow leaf disease, and nut splitting.\n"
            "- COVER mite damage, inflorescence dieback, and root grub.\n"
            "- INCLUDE Bordeaux mixture spray schedule, intercropping, and irrigation management."
        )
    elif "pigeon pea" in cn or "arhar" in cn or "tur" in cn:
        hints = (
            "- FOCUS on pod borer (Helicoverpa) as the primary pest.\n"
            "- COVER wilt complex, sterility mosaic, and phytophthora blight.\n"
            "- INCLUDE NSKE sprays, trap crops, and short-duration vs. long-duration varieties."
        )
    elif "chilli" in cn or "mirch" in cn:
        hints = (
            "- FOCUS on fruit rot complex, leaf curl virus, and thrips management.\n"
            "- COVER die-back (Colletotrichum), mite damage, and bacterial wilt.\n"
            "- INCLUDE nursery management, mulching, and drip fertigation."
        )
    else:
        hints = "- Provide accurate, evidence-based, and crop-specific guidance."

    return f"""You are an AI agricultural expert specialising exclusively in **{crop_name.upper()}** for the Kisan Call Centre (KCC) FAQ system. Generate one high-quality Q&A pair per call.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 CRITICAL SCOPE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This task is EXCLUSIVELY for **{crop_name.upper()}**.
Out of scope: {reject_str}
If the farmer question is clearly about any of these, you MUST return a JSON with `"category": "IRRELEVANT_CROP"` and `"answer": ""`.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 CRITICAL FIDELITY RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The "Cluster Topic" in the user request is the AUTHORITATIVE subject for this FAQ entry.
- Generate the ANSWER strictly about the Cluster Topic.
- Do NOT hallucinate crop names, chemical names, variety names, or dosages.
- Do NOT expand the scope beyond what the Cluster Topic specifies.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌱 CROP EXPERT HINTS — {crop_name.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{hints}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 MANDATORY LANGUAGE RULE — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The final output (CATEGORY, QUESTION, and ANSWER) MUST be written EXCLUSIVELY in **PROFESSIONAL AGRI-ENGLISH**.
- Even if the "Representative Question" or context is in Hindi, Hinglish, or any other regional language, you MUST translate and generate the response in **ENGLISH**.
- DO NOT use Hindi script (Devanagari) or transliterated Hindi in the output.
- All technical terms should use their standard English agricultural names.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 GENERATION GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **QUESTION**: Translate the "Representative Question" into formal English ONLY.
   - DO NOT add, remove, or change any information or detail present in the original.
   - DO NOT expand it, rephrase the meaning, or align it to the Cluster Topic.
   - The ONLY allowed change is: translate to formal English. Nothing else.
2. **ANSWER (200–400 words)**:
   - Must be written entirely in English.
   - Step-by-step technical guide using clear headings or bullet points.
   - Use generic (non-brand) names for all chemicals with metric dosages.
   - **SAFETY**: Include PPE requirements and Pre-Harvest Interval (PHI) where relevant.
   - **MANDATORY FOOTER**: For specific recommendations or field-level diagnosis, please contact your nearest Krishi Vigyan Kendra (KVK) or Block Agriculture Officer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏷 CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use exactly ONE of:
  Disease | Pest | Fertilizer and Nutrient | Variety | Agronomy | Other | IRRELEVANT_CROP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📤 OUTPUT FORMAT — EXCLUSIVELY IN ENGLISH (NO JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output EXACTLY in this format with these three headers.
Even if the user query is in Hindi, the response below MUST be in ENGLISH.

CATEGORY: [Classification]
QUESTION: [Formal English translation of the Representative Question — no additions]
ANSWER:
[The detailed technical answer in English...]
"""


# ══════════════════════════════════════════════════════════════════════════════
# Response Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_text_response(text: str):
    """Parse the CATEGORY / QUESTION / ANSWER text block from the model."""
    if not text or not isinstance(text, str):
        return None, "Invalid input"

    # Strip markdown code fences (any language tag: ```json, ```text, ```markdown, etc.)
    cleaned = re.sub(
        r"^```[a-zA-Z]*\s*|\s*```$", "",
        text.strip(), flags=re.DOTALL,
    ).strip()

    # Try JSON parsing first — the model sometimes returns JSON for IRRELEVANT_CROP
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "category" in obj:
            return {
                "question": str(obj.get("question", obj.get("q", ""))).strip(),
                "category": str(obj.get("category", "Other")).strip(),
                "answer":   str(obj.get("answer",   "")).strip(),
            }, None
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex extraction based on headers
    c_match = re.search(
        r'(?:^|\n)\s*(?:\*\*)?CATEGORY(?:\*\*)?:?\s*(.*?)(?=\n\s*(?:\*\*)?QUESTION|\n\s*(?:\*\*)?ANSWER|$)',
        cleaned, re.IGNORECASE | re.DOTALL,
    )
    q_match = re.search(
        r'(?:^|\n)\s*(?:\*\*)?QUESTION(?:\*\*)?:?\s*(.*?)(?=\n\s*(?:\*\*)?CATEGORY|\n\s*(?:\*\*)?ANSWER|$)',
        cleaned, re.IGNORECASE | re.DOTALL,
    )
    a_match = re.search(
        r'(?:^|\n)\s*(?:\*\*)?ANSWER(?:\*\*)?:?\s*(.*)',
        cleaned, re.IGNORECASE | re.DOTALL,
    )

    # If no headers found at all, flag as parse error
    if not c_match and not q_match and not a_match:
        return {"question": "", "category": "PARSE_ERROR", "answer": cleaned}, None

    # Strip bold markdown markers (**) the model sometimes wraps around category/answer
    cat = re.sub(r'^\*+\s*|\s*\*+$', '', c_match.group(1).strip()) if c_match else "Other"
    q   = q_match.group(1).strip() if q_match else ""
    # Strip "ANSWER:" that leaked onto the question line (model put ANSWER: on same line as QUESTION:)
    q   = re.sub(r'^ANSWER\s*:?\s*', '', q, flags=re.IGNORECASE).strip()
    ans = a_match.group(1).strip() if a_match else cleaned

    # Strip leading bold/asterisk artifact lines from the answer (e.g. "**\n", "** \n")
    ans = re.sub(r'^\*{1,3}\s*\n+', '', ans).strip()

    return {"question": q, "category": cat, "answer": ans}, None


# ══════════════════════════════════════════════════════════════════════════════
# Core Generation Logic (importable)
# ══════════════════════════════════════════════════════════════════════════════

import os as _os
from pathlib import Path as _Path
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parents[2] / ".env", override=True)
except ImportError:
    pass

_API_URL            = _os.environ.get("LLM_API_URL",   "http://100.100.108.44:8013/v1/chat/completions")
_API_MODEL          = _os.environ.get("LLM_MODEL",     "google/gemma-4-26B-A4B-it")
_API_KEY            = _os.environ.get("LLM_API_KEY",   "")
_DISABLE_THINKING   = _os.environ.get("LLM_DISABLE_THINKING", "").lower() == "true"


def _call_api(session, messages: list, max_tokens: int = 4000) -> str:
    payload = {
        "model": _API_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        **( {"thinking": {"type": "disabled"}} if _DISABLE_THINKING else {} ),
    }
    if _API_KEY:
        session.headers.update({"Authorization": f"Bearer {_API_KEY}"})
    resp = session.post(_API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or "").strip()


_WORKERS = int(_os.environ.get("LLM_STAGE5_WORKERS", "16"))  # concurrent API calls; vLLM queues extras automatically


def _process_row(args):
    """Process a single row; returns (index, question, category, answer)."""
    import requests as _requests
    i, row, system_prompt, crop, prefix = args
    question      = row.get('QueryText', row.get('representative_question', 'N/A'))
    freq          = row.get('count', row.get('raw_frequency', 1))
    cluster_label = str(row.get('cluster_label', '')).strip()
    answer_label  = str(row.get('answer_label',  '')).strip()
    # answer_label is the specific answer-distinct topic from unique_question_finder;
    # cluster_label is the broader cluster topic — use answer_label when available.
    topic = answer_label if answer_label and answer_label != cluster_label else cluster_label
    user_msg = prefix + f"""
Generate a {crop} FAQ entry based on:
- Cluster Topic: {topic}
- Representative Question: {question}
- Freq: {freq}

The "Cluster Topic" is the authoritative subject. Reframe the "Representative Question" to match the Cluster Topic if they differ.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    session = _requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    try:
        text = _call_api(session, messages)
    except Exception as exc:
        print(f"  [row {i}] API error: {exc}")
        text = ""
    parsed, _ = parse_text_response(text)
    if parsed and parsed.get("category") != "PARSE_ERROR":
        return i, parsed.get("question", "").strip(), parsed.get("category", "").strip(), parsed.get("answer", "").strip()
    return i, "", "PARSE_ERROR", (parsed.get("answer", text) if parsed else text)


def run_qa_generation(
    input_csv: str,
    output_csv: str,
    crop: str,
    model: str = _API_MODEL,
    tp: int = 1,
    gpu_util: float = 0.90,
    max_rows: int = None,
    workers: int = _WORKERS,
) -> str:
    """
    Run Q&A generation on a unique_questions_freq.csv file via remote API.

    Args:
        input_csv:  Path to the input CSV (must have 'representative_question' column).
        output_csv: Path to write the output CSV with generated Q&A columns.
        crop:       Crop name used in the system prompt.
        model:      Ignored — uses remote API at _API_URL.
        tp:         Ignored (legacy vLLM parameter).
        gpu_util:   Ignored (legacy vLLM parameter).
        max_rows:   If set, only process the first N rows.
        workers:    Number of concurrent API threads (default: 16).

    Returns:
        The output CSV path.
    """
    df = pd.read_csv(input_csv)
    if max_rows:
        df = df.head(max_rows)

    n = len(df)
    print(f"Initializing Remote API Q&A Generator for {crop} ({n} rows)")
    print(f"   API: {_API_URL} | Model: {_API_MODEL} | Workers: {workers}")

    system_prompt = get_system_prompt(crop)
    prefix = "MANDATORY: Translate and generate this FAQ entry EXCLUSIVELY in English.\n\n"

    print(f"\nStarting generation over {n} prompts...")
    start_time = time.time()

    rows_list = [(i, row) for i, (_, row) in enumerate(df.iterrows())]
    tasks = [(i, row, system_prompt, crop, prefix) for i, row in rows_list]

    results = [None] * n
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_row, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx, q, cat, ans = future.result()
            results[idx] = (q, cat, ans)
            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {completed}/{n} rows ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    rqs = n / elapsed if elapsed > 0 else 0
    print(f"Generation complete! Time: {elapsed:.1f}s ({rqs:.2f} rq/s)")

    gen_qs   = [r[0] for r in results]
    gen_cats = [r[1] for r in results]
    gen_ans  = [r[2] for r in results]

    output_df = df.copy()
    output_df["Generated_Question"] = gen_qs
    output_df["Generated_Category"] = gen_cats
    output_df["Generated_Answer"] = gen_ans

    output_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved results to {output_csv}")

    try:
        import torch
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    except ImportError:
        pass

    return output_csv


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Stage 7: FAQ Q&A Generation via vLLM Batch Inference',
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to unique_questions_freq.csv from Stage 6')
    parser.add_argument('--output', '-o', type=str,
                        help='Output CSV path (default: <input>_qa.csv)')
    parser.add_argument('--crop', '-c', type=str, required=True,
                        help='Crop name for the system prompt')
    parser.add_argument('--rows', '-r', type=str, default='all',
                        help='Limit rows to process (default: all)')

    # vLLM parameters
    parser.add_argument('--model', '-m', type=str,
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='Local model path or HuggingFace model ID')
    parser.add_argument('--tp', type=int, default=1,
                        help='Tensor parallel size — number of GPUs (default: 1)')
    parser.add_argument('--gpu-util', type=float, default=0.90,
                        help='GPU memory utilisation factor (default: 0.90)')

    return parser.parse_args()


def main():
    args = parse_arguments()
    output_csv = args.output or args.input.replace('.csv', '_qa.csv')
    max_rows = None if args.rows.lower() == 'all' else int(args.rows)

    run_qa_generation(
        input_csv=args.input,
        output_csv=output_csv,
        crop=args.crop,
        model=args.model,
        tp=args.tp,
        gpu_util=args.gpu_util,
        max_rows=max_rows,
    )


if __name__ == "__main__":
    main()
