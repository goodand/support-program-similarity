"""Score only the first 20 cosine candidates with Qwen3-Reranker-0.6B."""
import math

INSTRUCTION = ('Given a government support program, retrieve programs with similar objectives, '
               'supported activities, beneficiaries and eligibility conditions.')
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def reorder(prior, scores, top_k=20):
    head = prior[:top_k]
    if top_k < 1 or len(prior) != len(set(prior)) or set(scores) != set(head):
        raise ValueError('Reranker scores must match the unique first-stage shortlist')
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError('Nonfinite reranker score')
    position = {cid: i for i, cid in enumerate(prior)}
    return sorted(head, key=lambda cid: (-scores[cid], position[cid], cid)) + prior[top_k:]


def rerank(rows, rankings, model_path, device='cuda'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_path), padding_side='left', local_files_only=True)
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype,
              attn_implementation='sdpa', local_files_only=True).to(device).eval()
    no, yes = [tok.convert_tokens_to_ids(t) for t in ['no', 'yes']]
    if no is None or yes is None or no == yes or tok.unk_token_id in (no, yes):
        raise ValueError('Model tokenizer does not support Qwen yes/no tokens')
    by_id = {row['id']: row for row in rows}
    output = {}
    for qid, prior in rankings.items():
        scores = {}
        for cid in prior[:20]:
            text = PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {by_id[qid]['text']}\n<Document>: {by_id[cid]['text']}" + SUFFIX
            inputs = tok(text, add_special_tokens=False, truncation=False, return_tensors='pt').to(device)
            if inputs.input_ids.shape[1] > 32768:
                raise ValueError('Reranker pair exceeds 32768 tokens; no silent truncation')
            with torch.inference_mode():
                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1, [no, yes]].float()
                scores[cid] = torch.softmax(logits, dim=-1)[1].item()
        output[qid] = reorder(prior, scores)
    return output
