def align_pos_to_tokens(doc, hf_tokens, tokenizer):
    pos_tags_aligned = ['UNKNOWN'] * len(hf_tokens)
    current_hf_token_idx = 0
    
    for spacy_token in doc:
        temp_hf_token_chunk = ""
        start_hf_idx = current_hf_token_idx
        matched = False
        
        while current_hf_token_idx < len(hf_tokens):
            temp_hf_token_chunk += hf_tokens[current_hf_token_idx].replace('Ġ', '')
            if spacy_token.text.startswith(temp_hf_token_chunk):
                for i in range(start_hf_idx, current_hf_token_idx + 1):
                    pos_tags_aligned[i] = spacy_token.pos_
                if spacy_token.text == temp_hf_token_chunk:
                    matched = True
                    current_hf_token_idx += 1
                    break
            else:
                current_hf_token_idx = start_hf_idx
                break
            current_hf_token_idx += 1
            
        if not matched and current_hf_token_idx < len(hf_tokens):
            pos_tags_aligned[current_hf_token_idx] = spacy_token.pos_
            current_hf_token_idx += 1
            
    return pos_tags_aligned