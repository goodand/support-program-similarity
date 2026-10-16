"""Split original text using tokenizer offsets without decoding token fragments."""


def text_windows(tokenizer, text, limit=4096):
    if not isinstance(limit, int) or limit < 1 or not text:
        raise ValueError('Positive token limit and nonempty text required')
    pieces, weights = [], []
    remaining = text
    while remaining:
        encoded = tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        if not ids:
            raise ValueError('Tokenizer produced no tokens')
        end = len(remaining) if len(ids) <= limit else offsets[limit-1][1]
        # Multiple byte tokens can share a character offset. Re-encode the original
        # slice and move back whole characters until it fits, preserving Unicode.
        while end > 0:
            weight = len(tokenizer.encode(remaining[:end], add_special_tokens=False))
            if 0 < weight <= limit:
                break
            end -= 1
        if end <= 0:
            raise ValueError('A character cannot fit within the requested token limit')
        pieces.append(remaining[:end])
        weights.append(weight)
        remaining = remaining[end:]
    if ''.join(pieces) != text:
        raise ValueError('Window reconstruction failed')
    return pieces, weights
