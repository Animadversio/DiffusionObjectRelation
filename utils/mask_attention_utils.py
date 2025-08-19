"""
Attention Masking Utilities for Text Prompts

Tools for manipulating attention masks in text prompts, including masking specific
semantic parts (objects, colors, spatial relations) using spaCy NLP and tokenizers.

Features:
- Semantic Token Extraction:
  * get_meaningful_token_indices(prompt) -> dict - Extract objects, colors, spatial relations

- Attention Masking:
  * mask_padding_attention(prompt, tokenizer, model_max_length, device) -> (embeds, mask)
  * mask_all_attention(prompt, tokenizer, model_max_length, device) -> (embeds, mask)
  * mask_semantic_parts_attention(prompt, tokenizer, model_max_length, device, mask_parts) -> (embeds, mask)

- Supported Mask Parts: ['color1', 'object1', 'spatial', 'color2', 'object2']

Author: Jingxuan
"""

import spacy
import torch

nlp = spacy.load("en_core_web_sm")

def get_meaningful_token_indices(prompt: str):
    """
    Extract indices and token strings for two objects, their colors (adjectival modifiers),
    and a spatial relationship from a prompt.

    Example prompt: "red circle is to the left of blue square"

    Returns:
        dict with keys:
            - color1_index,  color1_text
            - object1_index, object1_text
            - spatial,       spatial_text
            - color2_index,  color2_text
            - object2_index, object2_text
    """
    spatial_keywords = {
        "left", "right", "above", "below", "behind", "front",
        "center", "top", "bottom"
    }

    doc = nlp(prompt)
    
    # Debug: Print all tokens and their dependencies
    print(f"DEBUG: Analyzing prompt: '{prompt}'")
    print("DEBUG: Tokens and their dependencies:")
    for tok in doc:
        print(f"  {tok.text}: {tok.dep_} (head: {tok.head.text}, index: {tok.i})")

    # 1. Find the first spatial keyword
    spatial_tok = next(
        (tok for tok in doc if tok.text.lower() in spatial_keywords),
        None
    )
    if spatial_tok is None:
        raise ValueError("No spatial keyword found in prompt.")
    spatial_index, spatial_text = spatial_tok.i, spatial_tok.text

    # 2. Find all adjectival modifiers (amod), sorted by token index
    amod_toks = sorted(
        [tok for tok in doc if tok.dep_ == "amod" or tok.dep_ == "compound"],
        key=lambda t: t.i
    )
    
    # Debug: Print found adjectival modifiers
    print(f"DEBUG: Found {len(amod_toks)} adjectival/compound modifiers:")
    for i, tok in enumerate(amod_toks):
        print(f"  {i+1}. {tok.text} (index: {tok.i}, head: {tok.head.text}, dep: {tok.dep_})")
    
    if len(amod_toks) < 2:
        print(f"DEBUG: ERROR - Need at least 2 adjectival/compound modifiers, but only found {len(amod_toks)}")
        raise ValueError("Need at least two adjectival modifiers (colors).")

    # 3. Assign colors and their head nouns
    color1_tok = amod_toks[0]
    obj1_tok   = color1_tok.head

    color2_tok = amod_toks[1]
    obj2_tok   = color2_tok.head

    return {
        "color1_index":  color1_tok.i,
        "color1_text":   color1_tok.text,
        "object1_index": obj1_tok.i,
        "object1_text":  obj1_tok.text,
        "spatial":       spatial_index,
        "spatial_text":  spatial_text,
        "color2_index":  color2_tok.i,
        "color2_text":   color2_tok.text,
        "object2_index": obj2_tok.i,
        "object2_text":  obj2_tok.text,
    }


def mask_padding_attention(prompt: str, tokenizer, model_max_length: int, device):
    """
    Masks attention for positions where input_ids is 1 (padding token for some tokenizers).
    Adjust the padding token ID if yours is different (e.g., 0 for BERT).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=model_max_length,
        return_attention_mask=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    manipulated_mask = attention_mask.clone()
    # Mask positions where input_ids is 1.
    # Common padding token IDs can be tokenizer.pad_token_id.
    # Assuming 1 is the padding token ID as per the request.
    # If your tokenizer uses a different pad_token_id (e.g., 0), change `1` to `tokenizer.pad_token_id`.
    padding_token_id = 1 # Explicitly using 1 as requested.
    # If your tokenizer has pad_token_id attribute, it's safer to use:
    # if tokenizer.pad_token_id is not None:
    #     padding_token_id = tokenizer.pad_token_id
    # else:
    #     print("Warning: tokenizer.pad_token_id is not set. Assuming padding token ID is 1.")
    
    manipulated_mask[input_ids == padding_token_id] = 0
    return manipulated_mask


def mask_all_attention(prompt: str, tokenizer, model_max_length: int, device):
    """
    Masks all positions in the attention mask, effectively allowing no attention.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=model_max_length,
        return_attention_mask=True 
    )
    # Get the original mask to know the shape, then create a zero tensor of that shape.
    attention_mask = inputs["attention_mask"].to(device) 
    manipulated_mask = torch.zeros_like(attention_mask) # Create a mask of all zeros
    #print(manipulated_mask.shape)
    manipulated_mask[0,-1] = 1
    return manipulated_mask

# TODO: check if this is correct
def mask_semantic_parts_attention(prompt: str,
                                  tokenizer,
                                  model_max_length: int,
                                  device,
                                  part_to_mask: str):
    """
    Masks attention for specified semantic parts in the prompt.
    Supported part_to_mask values:
      - "object1", "object2", "spatial", "color1", "color2"
      - "objects"  (masks both object1 & object2)
      - "colors"   (masks both color1  & color2)
    """
    # tokenize + get offsets & attention
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=model_max_length,
        return_offsets_mapping=False,
        return_attention_mask=True
    )
    mask    = inputs["attention_mask"].to(device).clone()
    #offsets = inputs["offset_mapping"].squeeze(0).tolist()

    # parse indices from spaCy
    parsed = get_meaningful_token_indices(prompt)
    #doc    = nlp_model(prompt)
    #print(doc)

    # decide which spaCy‐token fields we want to mask
    key_map = {
        "object1": "object1_index",
        "object2": "object2_index",
        "spatial": "spatial",
        "color1":  "color1_index",
        "color2":  "color2_index",
    }
    #print(key_map)
    if part_to_mask == "objects":
        parts = ["object1", "object2"]
    elif part_to_mask == "colors":
        parts = ["color1",  "color2"]
    elif part_to_mask in key_map:
        parts = [part_to_mask]
    else:
        raise ValueError(f"Unknown part_to_mask '{part_to_mask}'")

    # build list of (char_start, char_end) spans to zero‐out
    #spans = []
    for part in parts:
        idx = parsed.get(key_map[part])
        #print(idx)
        if idx is None:
            continue
        mask[0,idx] = 0
        #spans.append((tok.idx, tok.idx + len(tok.text)))
    #print(spans)
    # for each HF token, if its char‐span is fully inside any target span → mask it
    #for hf_idx, (cs, ce) in enumerate(offsets):
    #    if cs == ce:
    #        continue
    #    if any(cs >= s and ce <= e for s, e in spans):
            #mask[0, hf_idx] = 0

    return mask

