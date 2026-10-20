# Repository Handoff: Support Program Similarity

**Audience:** the next Codex session that will continue updating this repository
**Repository:** `goodand/support-program-similarity`
**Default branch:** `main`
**Handoff language:** English, to make the technical state and constraints portable across sessions.

## 1. Mission and current state

This repository contains a selected retrieval and ranking pipeline for comparing Korean government support-program descriptions. It is the downstream retrieval/evaluation part of a larger local workflow. It does not currently contain source-document parsing, program-boundary detection, chunk construction, human-review UI, or original evaluation data.

The selected retrieval configuration is:

1. Encode every prepared program text with local `Qwen3-Embedding-8B`.
2. For long inputs, split the original text into Unicode-preserving windows of at most 4096 model tokens. Encode each window with normalization, compute the token-count-weighted mean, then L2-normalize the pooled vector.
3. Add the domain retrieval instruction to the query text only. Do not rely on implicit model prompts.
4. Rank candidates by cosine similarity, excluding caller-specified IDs, identical IDs, and exact duplicate query text.
5. Score the first 20 cosine candidates with `Qwen3-Reranker-0.6B` using the Qwen yes/no next-token probability. Keep the first-stage order after rank 20.
6. Save ranked IDs, vectors, and a manifest containing input and code hashes and software versions.

At the last verified handoff point, local `main` and `origin/main` both pointed to `d79b159` (`Normalize README line endings`). The working tree was clean. The public README was subsequently updated to explain the actual parser/input workflow and synchronized to GitHub through the GitHub connector. Before changing anything, the next session must inspect the live branch and working tree because the connector commit may have advanced `origin/main` beyond that local SHA.

The repository is public. The user explicitly stated that original **HWP, HWPX, PDF, and Excel files must not be uploaded**. Keep those files, real program texts, human labels, Gold/Silver sets, model weights, vectors, raw requests, and experiment logs out of the repository.

## 2. First actions for the next session

Start from the current repository checkout and verify the remote state. Do not assume that the prior session left local and remote history synchronized.

```powershell
git status --short --branch
git remote -v
git fetch origin main
git log --oneline --decorate -5 --all
git diff --stat HEAD origin/main
git ls-files
```

Read this handoff and the README before editing. If local `main` is behind the remote README-only commits, fast-forward or otherwise reconcile without discarding local commits or staged changes. Do not use a blind hard reset. Do not stage the whole mixed workspace; stage only reviewed paths.

Before any push, inspect staged blobs, run the repository publication gate, and examine the complete staged diff. This is a public repository and local ignore rules can be bypassed with `git add -f`.

## 3. Repository layout

```text
README.md                              Project scope, experiment selection, parsing provenance, usage
HANDOFF.md                             This continuation guide
support_program_similarity/
  retrieval.py                         Embedding, cosine ranking, input validation, run manifest
  windows.py                           Original-text slicing at tokenizer offsets
  reranking.py                         Qwen 0.6B shortlist scoring and stable ordering
  evaluation.py                        Offline MAP/MRR/recall metrics
  __main__.py                          Explicit offline evaluation CLI
examples/
  rankings.synthetic.json              Synthetic ranking example
  positives.synthetic.json             Synthetic labels; unrelated to real programs
schemas/                                Not currently present; do not assume a model-output schema exists
tests/test_core.py                     11 focused tests at last verification
tools/check_publish.py                 Scans the Git index blobs for publication hazards
release/selection.json                 Selection rationale, metrics, source hashes, limitations
.github/workflows/check.yml            CI: install base requirements, unit tests, publish gate
requirements.txt                       Lightweight runtime/evaluation dependency
requirements-inference.txt             Local model inference dependencies
.gitignore                             Explicit root allowlist and private-data exclusions
```

The README describes install and CLI usage. It also states that inference reads the prepared JSON input and does not call the parser MCP server directly.

## 4. Why this configuration was selected

The user clarified the selection principle: do not choose the numerically highest score alone. Prefer the configuration backed by the broadest sample and transfer/robustness evidence, then compare the maximum rank needed to retrieve all known positives (`K_all`), using MAP as a secondary ranking-quality measure.

The principal integrated comparison used 13 queries, 140 candidates, and 19 known positive relations. Selected Qwen embedding plus Qwen reranker results:

| Metric | Result |
|---|---:|
| MAP | 0.567435 |
| MRR | 0.586993 |
| Recall@10 | 0.692308 |
| Recall@20 | 0.884615 |
| Recall@27 | 1.000000 |
| Maximum rank needed for every known positive (`K_all`) | 27 |

These are known-positive retrieval metrics. Unjudged candidates are not proven negatives, and labels are incomplete. `K_all=27` means all currently known positives appeared by rank 27 for this particular sample; it is not a universal cutoff guarantee.

Other results that informed selection:

- Qwen cosine without reranking on the same integrated set: MAP 0.547913; Recall@10 0.807692; Recall@20 0.884615; `K_all=27`. Reranking improved MAP and top-1 tendency, but **lowered Recall@10**. The README records this tradeoff.
- Qwen + LCS + reranker on that set required `K_all=50`; the LCS path was not selected for the general pipeline.
- A beneficiary/eligibility-only field extraction condition had `K_all=25` on the integrated set, but reverse-direction Recall@10 fell from 0.7368 to 0.5789. Purpose/content extraction also degraded reverse Recall@10 to 0.6842. These are useful counterexamples; neither should be presented as a robust overall winner.
- In a separate older 10-query reverse-transfer experiment, Qwen Reranker improved `K_all` from 20 to 8. That experiment had a different candidate pool and first-stage retrieval condition. Do not merge its scores with the 13-query integrated set.
- The reverse/transfer checks reused previously seen programs and judgments. They are not independent, held-out generalization tests on unseen programs. The next session should preserve this qualification.

Selection is a robustness-informed choice, not a claim of dominance on every metric or every evaluation set.

## 5. Parser and preparation provenance

The related parser repository is [goodand/document-parser-mcp](https://github.com/goodand/document-parser-mcp). It provides document parsing and source-evidence access. The actual historical preparation paths were mixed:

- For the original budget-document paragraph corpus, a preparation script directly called `govparse.native_toc` HWP/HWPX reading and paragraph classification helpers. It retained source hashes and paragraph anchors such as section, ordinal, and paragraph-header record. Subsequent chunk code selected program paragraphs.
- The `field_leaf_v3` experiment combined the prior leaf chunk with paragraphs selected by field-routing rules and deduplicated paragraphs using `(source, paragraph_index)`.
- For additional practitioner opinion forms, source-linked preparation used `pyhwp` extracted tables/cells, retaining table/row/column references and the source hash. It assembled query texts from matched overview cells, removing exact duplicate text. Candidate inputs came from supplementary notice bodies or older prepared program chunks.
- A manual source registry and, in some rows, a parent-program proxy were used. Historical input-audit files explicitly mark this mapping as **not a generalized name-resolution rule** and identify unvalidated scope/alias cases.
- Practitioner similarity judgments and their quoted rationales were held separately from the text input; judgment quotations were not used as business-description text.

The MCP interface and direct Python library calls are both part of the broader workflow, but the current `retrieval.py` accepts prepared JSON and does not invoke an MCP server. The published metrics were not regenerated end-to-end from original files using the current parser repository revision. Do not imply that they were.

When updating the parser-to-retrieval boundary, record at least: parser repository commit/version; original-file SHA-256; parsed artifact SHA-256; paragraph/table/cell source anchors; program identity/link decision; chunk-generation rule and version; resulting text hash; and any human verification. Keep all real documents and labels external.

## 6. Important implementation contracts

### Corpus JSON

The inference CLI accepts a JSON array. Every record has a globally unique string `id`, a `role` of `query` or `candidate`, and nonempty `text`. Query records may include `excluded_candidate_ids`, a list of candidate IDs. At least one query and one candidate are required. The CLI validates these structural requirements before loading models.

```json
[
  {"id":"Q1","role":"query","text":"prepared query program text","excluded_candidate_ids":[]},
  {"id":"C1","role":"candidate","text":"prepared candidate program text"}
]
```

The output directory must be new. The runner writes `rankings.json`, `vectors.npy`, and `manifest.json`. Avoid changing this behavior to silently overwrite an existing run. The manifest records the corpus hash, model paths, method, instruction, windowing/pooling description, library versions, code hashes, and per-input text hashes.

### Window preservation

`windows.text_windows` uses tokenizer offsets to slice the **original string**. It re-tokenizes each proposed original-text slice and moves the boundary back until the token limit is satisfied. It verifies that concatenating slices reconstructs the exact original text. Do not replace it with token-ID decoding that can alter malformed sequences, unusual Unicode, or tokenizer-normalized text. The previous publication audit found boundary corruption in 4 of 153 historical inputs from the decode/re-encode path; the new method preserved all 153 originals and changed four embeddings.

### Embedding and reranking

- Query instruction: defined in `retrieval.py`; passed explicitly as literal query text with `prompt=''` to prevent the local SentenceTransformer snapshot from silently applying its default web-search prompt.
- Snapshot role routing: the implementation rejects a module whose type name contains `Router`. Do not remove this check without confirming the target local Qwen snapshot has the expected non-routed architecture.
- Window embeddings are normalized individually, token-weight averaged, then normalized again.
- Candidate ranking ties are broken by candidate ID, making cosine ordering deterministic for equal computed scores.
- Reranker instruction/template and model revision are frozen in code/`release/selection.json`. The reranker scores exactly the first 20 cosine candidates; the tail retains cosine order. Token overflow above 32768 is an error, never silently truncated.
- Stable score ties use the first-stage rank and then ID.
- The GPU verification after fixing Unicode windows recomputed the four changed vectors. Their top-20 candidate sets did not change, so prior reranker scores could be reused. Three queries had some ordering changes; the aggregate known-positive metrics and `K_all=27` remained unchanged.

### Gold/label isolation

The inference runner must not load positive labels, prior scores, or existing rankings. The evaluation CLI is a separate process and accepts rankings plus a private positive-label file. It freezes/hash-records the ranking file before it reads labels. This is process separation in the code; it is **not an OS sandbox** preventing another process or model worker from reading arbitrary files.

`evaluation.evaluate` computes MAP@K, MRR@K, macro Recall@K, and the maximum rank required to cover all known positives. AP divides by the full known-positive count, including positives below the cutoff. A missing known positive therefore lowers AP and recall. Queries with zero labels are currently rejected and need an explicit policy if they enter future evaluation.

## 7. Verification already performed

At the last implementation check before handoff documentation was requested:

- 11 unit tests passed, including Unicode reconstruction, rank ordering, duplicate exclusion, evaluation denominator behavior, and publication-gate cases.
- `python tools/check_publish.py` passed against the staged Git index. It checks staged blobs, allowed roots, blocked extensions, binary/oversize files, private paths, likely secret patterns, and JSON allowlisting.
- `git diff --cached --check` passed before the first publication.
- The GitHub Actions `Source checks` workflow completed successfully for the initial source publication commit `ba5e84184daa063097eb9ef57e4315fb37582a1f`.
- A Codex CLI review identified a weakness in the initial publication gate: force-adding private JSON under an allowed folder could bypass it. The gate was tightened to allow only explicitly reviewed synthetic/provenance JSON files, with regression cases. The later Codex review found no further blocker after the Unicode-preserving window change; the GPT-6.1-Sol / low review also found no further blocker.
- Afterward the parser-workflow README text was updated directly through the GitHub connector and line endings were normalized. Check the current Actions status for the newest remote commit; do not assume the earlier CI result covers that documentation-only commit.

Useful commands:

```powershell
python -m compileall -q support_program_similarity tools tests
python -m unittest discover -s tests -v
python tools/check_publish.py
python -m support_program_similarity --rankings examples/rankings.synthetic.json --positives examples/positives.synthetic.json --k 3
git diff --check
git diff --cached --check
```

The source-only test suite does not prove model quality. Reproducing retrieval requires the local model snapshots, GPU/runtime setup, and the corresponding private prepared input. Never include those private assets in a public test fixture.

## 8. Publication and data-safety rules

The `.gitignore` is intentionally restrictive because the checkout sits among unrelated local experiments. It only permits selected top-level roots/files. Before staging, inspect the index with `git ls-files --stage`; after staging, run `python tools/check_publish.py` and review the full staged diff.

Never publish or force-add:

- `.hwp`, `.hwpx`, `.pdf`, Excel formats or source archives;
- real corpus, query, candidate, Gold/Silver, human-vote, source-link, or identity-mapping files;
- original text snippets, literal evidence quotes, non-synthetic business names, budgets, or document paths;
- `.npy`/`.npz` vectors, model weights/checkpoints, raw inference requests/responses, local model caches, generated run folders, or personal workstation paths;
- `.env` files, API keys, access tokens, or credentials.

Only synthetic, non-identifying examples are permitted. If a future artifact needs JSON, add the exact reviewed filename to the publication allowlist rather than broadening all JSON paths.

## 9. Recommended next work, in order

1. **Refresh current remote evidence.** Check `main`, current workflow results, local status, and README content. Reconcile the local checkout if it is behind the connector-written README commit.
2. **Add a parser-to-corpus adapter contract.** Define a generic input contract based on parsed paragraphs/tables and stable source anchors. Make parser provenance and the program-level linking step explicit. Keep direct MCP invocation optional; accept already parsed records too.
3. **Separate business-level preparation from retrieval.** The public repo currently assumes program text is ready. A future adapter should emit the corpus JSON plus an auditable mapping manifest, and should fail closed on ambiguous business-to-source mappings.
4. **Add deterministic replay metadata.** Include immutable embedding model revision/hash, tokenizer/config hashes, reranker revision/hash, exact numerical settings, runtime versions, and parser/chunk-generation revision. Do not claim reproducibility from a model name alone.
5. **Strengthen unseen-program evaluation.** Obtain independently held-out, human-reviewed programs and judgments. Keep query/program overlap out of the holdout. Report sample size, per-query metrics, macro metrics, confidence intervals or bootstrap uncertainty, and label coverage. Unknown candidates are not negatives.
6. **Reassess selection under the user's criteria.** Compare configurations on matched larger/transfer/held-out samples, putting breadth and robustness first, then `K_all`, then MAP. Preserve the observed Recall@10 regression of reranking in all reports.
7. **Re-run all public gates before every push.** Unit tests, synthetic CLI example, index-based publication gate, staged diff inspection, and GitHub Actions for the exact pushed SHA.

Do not add unrelated experiments or models merely because they exist locally. A configuration belongs in the public repository only if it is selected by evidence under the stated evaluation policy and can be shared without exposing real documents or labels.

## 10. Suggested opening prompt for the next session

> Read `HANDOFF.md` and `README.md` first. Inspect the current local and GitHub `main` states; do not assume they are synchronized. Continue updating `goodand/support-program-similarity` toward a traceable parser-to-program-text-to-retrieval workflow. Preserve the selected Qwen3-Embedding-8B + Qwen3-Reranker-0.6B baseline and all stated metric tradeoffs. Do not publish HWP/HWPX/PDF/Excel files, real source text, human labels, model weights, vectors, or private logs. Start by reporting the concrete gap between the present prepared-JSON-only runner and an auditable parser adapter, then implement the smallest end-to-end generic adapter with source anchors, strict ambiguity handling, synthetic fixtures, tests, publication-gate review, and a verified push of only reviewed files.

