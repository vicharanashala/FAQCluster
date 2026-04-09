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

import pandas as pd
import re
import argparse
import sys
import time
from pathlib import Path


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
Stay strictly grounded in the provided "Representative Question".
- Expand into a comprehensive guide, but do NOT change the technical meaning.
- Do NOT hallucinate crop names, chemical names, or dosages.

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
1. **QUESTION**: Rewrite in clear, professional English. Retain agronomic intent. Translate to English if original is in another language.
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
QUESTION: [The polished English question]
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

    # Remove markdown code blocks if the model wraps output
    cleaned = re.sub(
        r"^```(?:markdown|text)?\s*|\s*```$", "",
        text.strip(), flags=re.DOTALL,
    ).strip()

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

    cat = c_match.group(1).strip() if c_match else "Other"
    q = q_match.group(1).strip() if q_match else ""
    ans = a_match.group(1).strip() if a_match else cleaned

    # If no headers found at all, flag as parse error
    if not c_match and not q_match and not a_match:
        return {"question": "", "category": "PARSE_ERROR", "answer": cleaned}, None

    return {"question": q, "category": cat, "answer": ans}, None


# ══════════════════════════════════════════════════════════════════════════════
# Core Generation Logic (importable)
# ══════════════════════════════════════════════════════════════════════════════

def run_qa_generation(
    input_csv: str,
    output_csv: str,
    crop: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    tp: int = 1,
    gpu_util: float = 0.90,
    max_rows: int = None,
) -> str:
    """
    Run vLLM batch Q&A generation on a unique_questions_freq.csv file.

    Args:
        input_csv:  Path to the input CSV (must have 'representative_question' column).
        output_csv: Path to write the output CSV with generated Q&A columns.
        crop:       Crop name used in the system prompt.
        model:      Local model path or HuggingFace model ID.
        tp:         Tensor parallel size (number of GPUs for this job).
        gpu_util:   GPU memory utilisation fraction (0.0–1.0).
        max_rows:   If set, only process the first N rows.

    Returns:
        The output CSV path.
    """
    # Lazy import — vllm is heavy and optional for Stages 1–6
    from vllm import LLM, SamplingParams

    df = pd.read_csv(input_csv)
    if max_rows:
        df = df.head(max_rows)

    print(f"🚀 Initializing vLLM Batch Generator for {crop} ({len(df)} rows)")
    print(f"   Model: {model} | TP: {tp} | GPU Util: {gpu_util}")

    # Launch vLLM engine
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        enforce_eager=True,  # Helpful for stability on large batches
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4000,
        top_p=0.95,
    )

    system_prompt = get_system_prompt(crop)

    # Mandatory English override at start of user message —
    # handles cases where models ignore system prompt for regional inputs
    prefix = "MANDATORY: Translate and generate this FAQ entry EXCLUSIVELY in English.\n\n"

    # Build batch prompts using Chat Templating for Instruct models
    prompts_for_vllm = []
    for _, row in df.iterrows():
        question = row.get('QueryText', row.get('representative_question', 'N/A'))
        freq = row.get('count', row.get('raw_frequency', 1))

        user_msg = prefix + f"""
Generate a {crop} FAQ entry based on:
- Representative Question: {question}
- Freq: {freq}
"""
        prompts_for_vllm.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ])

    print(f"\n⏳ Starting generation over {len(prompts_for_vllm)} prompts...")
    start_time = time.time()

    # Apply chat template and generate
    tokenizer = llm.get_tokenizer()
    raw_prompts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in prompts_for_vllm
    ]

    outputs = llm.generate(raw_prompts, sampling_params=sampling_params)

    elapsed = time.time() - start_time
    rqs = len(prompts_for_vllm) / elapsed if elapsed > 0 else 0
    print(f"✅ Generation complete! Time: {elapsed:.1f}s ({rqs:.2f} rq/s)")

    # Parse results
    gen_qs, gen_cats, gen_ans = [], [], []
    for out in outputs:
        text = out.outputs[0].text
        parsed, _ = parse_text_response(text)
        if parsed and parsed.get("category") != "PARSE_ERROR":
            gen_qs.append(parsed.get("question", "").strip())
            gen_cats.append(parsed.get("category", "").strip())
            gen_ans.append(parsed.get("answer", "").strip())
        else:
            gen_qs.append("")
            gen_cats.append("PARSE_ERROR")
            gen_ans.append(parsed.get("answer", text) if parsed else text)

    # Save back to DataFrame
    output_df = df.copy()
    output_df["Generated_Question"] = gen_qs
    output_df["Generated_Category"] = gen_cats
    output_df["Generated_Answer"] = gen_ans

    output_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Saved results to {output_csv}")

    # Clean up vLLM resources to free GPU memory
    del llm
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
