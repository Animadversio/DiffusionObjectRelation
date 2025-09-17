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

def get_meaningful_token_indices(prompt: str, color_optional: bool = False):
    """
    Extract indices and token strings for two objects, their colors (adjectival modifiers),
    and a spatial relationship from a prompt.

    Example prompt: "red circle is to the left of blue square"

    Args:
        prompt: The input prompt to analyze
        color_optional: If True, colors are not required. If False, requires at least 2 colors.

    Returns:
        dict with keys:
            - color1_index,  color1_text (None if color_optional=True and no colors found)
            - object1_index, object1_text
            - spatial,       spatial_text
            - spatial_general, spatial_general_text (complete spatial phrase)
            - color2_index,  color2_text (None if color_optional=True and no colors found)
            - object2_index, object2_text
    """
    spatial_keywords = {
        "left", "right", "above", "below", "behind", "front",
        "center", "top", "bottom", "near"
    }
    
    # Multi-word spatial phrases
    spatial_phrases = [
        "next to", "close to", "far from", "in front of", "in back of"
    ]

    doc = nlp(prompt)
    
    # Debug: Print all tokens and their dependencies
    print(f"DEBUG: Analyzing prompt: '{prompt}'")
    print("DEBUG: Tokens and their dependencies:")
    for tok in doc:
        print(f"  {tok.text}: {tok.dep_} (head: {tok.head.text}, index: {tok.i})")

    # 1. Find the first spatial keyword or phrase
    spatial_tok = None
    spatial_phrase_match = None
    
    # First check for multi-word phrases
    prompt_lower = prompt.lower()
    for phrase in spatial_phrases:
        if phrase in prompt_lower:
            # Find the position of this phrase
            phrase_start = prompt_lower.find(phrase)
            # Find the token that corresponds to the first word of the phrase
            for tok in doc:
                if tok.idx <= phrase_start < tok.idx + len(tok.text):
                    spatial_tok = tok
                    spatial_phrase_match = phrase
                    break
            if spatial_tok:
                break
    
    # If no multi-word phrase found, look for single-word keywords
    if spatial_tok is None:
        spatial_tok = next(
            (tok for tok in doc if tok.text.lower() in spatial_keywords),
            None
        )
    
    if spatial_tok is None:
        raise ValueError("No spatial keyword found in prompt.")
    spatial_index, spatial_text = spatial_tok.i, spatial_tok.text

    # 2. Extract the complete spatial phrase
    def get_spatial_phrase(spatial_token):
        """Extract the complete spatial phrase containing the spatial keyword."""
        # If we found a multi-word phrase, handle it specially
        if spatial_phrase_match:
            # Find all tokens that make up the multi-word phrase
            phrase_tokens = []
            phrase_words = spatial_phrase_match.split()
            
            # Starting from the spatial_token, collect tokens for the phrase
            current_idx = spatial_token.i
            for word in phrase_words:
                if current_idx < len(doc) and doc[current_idx].text.lower() == word.lower():
                    phrase_tokens.append(doc[current_idx])
                    current_idx += 1
                else:
                    # Fallback if tokens don't match perfectly
                    break
            
            if phrase_tokens:
                start_idx = phrase_tokens[0].i
                end_idx = phrase_tokens[-1].i
                phrase_text = spatial_phrase_match
                return start_idx, end_idx, phrase_text
        
        # Original logic for single-word spatial keywords
        # Go backwards to find the start of the phrase
        start_tokens = []
        prev_tok = spatial_token
        while prev_tok.i > 0:
            prev_tok = doc[prev_tok.i - 1]
            # Include prepositions, determiners, and related words
            if (prev_tok.pos_ in ["ADP", "DET"] or  # prepositions, determiners
                prev_tok.dep_ in ["prep", "det"] or
                prev_tok.text.lower() in ["to", "on", "in", "at", "of"]):
                start_tokens.insert(0, prev_tok)
            else:
                break
        
        # Go forwards to find the end of the phrase, but stop before articles that precede objects
        end_tokens = []
        next_tok = spatial_token
        while next_tok.i < len(doc) - 1:
            next_tok = doc[next_tok.i + 1]
            # Only include "of" as it's part of spatial phrases like "to the left of", "on the bottom of"
            # But don't include articles like "a", "an", "the" that come after "of"
            if next_tok.text.lower() == "of":
                end_tokens.append(next_tok)
            else:
                break
        
        # Combine all tokens
        phrase_tokens = start_tokens + [spatial_token] + end_tokens
        
        if phrase_tokens:
            # Get the span from first to last token
            start_idx = phrase_tokens[0].i
            end_idx = phrase_tokens[-1].i
            phrase_text = doc[start_idx:end_idx+1].text
            return start_idx, end_idx, phrase_text
        else:
            # Fallback to just the spatial token
            return spatial_token.i, spatial_token.i, spatial_token.text
    
    spatial_general_start, spatial_general_end, spatial_general_text = get_spatial_phrase(spatial_tok)
    
    # Debug: Print spatial phrase info
    print(f"DEBUG: Spatial phrase: '{spatial_general_text}' (indices {spatial_general_start}-{spatial_general_end})")

    # 3. Find all adjectival modifiers (amod), sorted by token index
    amod_toks = sorted(
        [tok for tok in doc if tok.dep_ == "amod" or tok.dep_ == "compound"],
        key=lambda t: t.i
    )
    
    # Debug: Print found adjectival modifiers
    print(f"DEBUG: Found {len(amod_toks)} adjectival/compound modifiers:")
    for i, tok in enumerate(amod_toks):
        print(f"  {i+1}. {tok.text} (index: {tok.i}, head: {tok.head.text}, dep: {tok.dep_})")
    
    # 4. Handle color requirements based on color_optional flag
    if not color_optional and len(amod_toks) < 2:
        print(f"DEBUG: ERROR - Need at least 2 adjectival/compound modifiers, but only found {len(amod_toks)}")
        raise ValueError("Need at least two adjectival modifiers (colors).")

    # 5. Assign colors and their head nouns
    if len(amod_toks) >= 2:
        # Standard case: we have at least 2 modifiers
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
            "spatial_general": spatial_general_start,
            "spatial_general_text": spatial_general_text,
            "color2_index":  color2_tok.i,
            "color2_text":   color2_tok.text,
            "object2_index": obj2_tok.i,
            "object2_text":  obj2_tok.text,
        }
    elif len(amod_toks) == 1:
        # One modifier case: assign to first object, find second object differently
        color1_tok = amod_toks[0]
        obj1_tok   = color1_tok.head
        
        # Find the second object (noun) that's not the first object
        nouns = [tok for tok in doc if tok.pos_ == "NOUN" and tok != obj1_tok]
        if not nouns:
            raise ValueError("Could not find a second object in the prompt.")
        obj2_tok = nouns[0]  # Take the first available noun
        
        return {
            "color1_index":  color1_tok.i,
            "color1_text":   color1_tok.text,
            "object1_index": obj1_tok.i,
            "object1_text":  obj1_tok.text,
            "spatial":       spatial_index,
            "spatial_text":  spatial_text,
            "spatial_general": spatial_general_start,
            "spatial_general_text": spatial_general_text,
            "color2_index":  None,
            "color2_text":   None,
            "object2_index": obj2_tok.i,
            "object2_text":  obj2_tok.text,
        }
    else:
        # No modifiers case: find two objects directly
        nouns = [tok for tok in doc if tok.pos_ == "NOUN"]
        if len(nouns) < 2:
            raise ValueError("Could not find two objects in the prompt.")
        
        obj1_tok = nouns[0]
        obj2_tok = nouns[1]
        
        return {
            "color1_index":  None,
            "color1_text":   None,
            "object1_index": obj1_tok.i,
            "object1_text":  obj1_tok.text,
            "spatial":       spatial_index,
            "spatial_text":  spatial_text,
            "spatial_general": spatial_general_start,
            "spatial_general_text": spatial_general_text,
            "color2_index":  None,
            "color2_text":   None,
            "object2_index": obj2_tok.i,
            "object2_text":  obj2_tok.text,
        }
#indices = get_meaningful_token_indices("a microwave on the right of a bee", color_optional=True)
#print(indices)

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

