# Parameter Golf: results and model study guide

Review and source collection as of **September 9, 2026**.

This is a study fork of [openai/parameter-golf](https://github.com/openai/parameter-golf), available at [incrediblecrab/2026-parameter-golf](https://github.com/incrediblecrab/2026-parameter-golf). The full upstream checkout is retained, with four additional PR-only submission directories. **Fourteen representative models are featured below**, rather than treating every small leaderboard variation as a different architecture.

The original project README, setup instructions, and complete published leaderboard are preserved in [UPSTREAM_README.md](UPSTREAM_README.md). Exact source revisions and entry points are recorded in [model_sources.json](model_sources.json). Submission code, write-ups, tokenizers, and existing logs are preserved without modification.

## What was the challenge?

OpenAI's **Model Craft Challenge: Parameter Golf** asked participants to train a language model that predicts held-out FineWeb text as well as possible, subject to:

| Constraint | Budget |
|---|---|
| Submission artifact | **16,000,000 bytes**, including compressed model weights **and code**; not 16 MiB |
| Training | **10 minutes on eight H100 SXM GPUs** |
| Evaluation | A **separate 10-minute budget** on the same hardware |
| Score | **Bits per byte (BPB)** on the prescribed FineWeb validation split; lower is better |

Think of BPB as the model's predicted compression cost for unseen text, not a chatbot-quality score. Scoring against original text bytes allows different tokenizers to compete. It does not permit dropping information or changing the byte-count denominator to improve the apparent result.

The training limit applied to each submitted run, **not to the entire research or hyperparameter-search budget**. Evaluation could use additional computation and causal test-time learning, but not network access, uncounted training data, or answers from unscored validation tokens. See the [original rules](UPSTREAM_README.md#faq).

## Who won?

**The top accepted leaderboard entry is codemath3000's PR #2135, at 1.05651 BPB.** Its May 1 submission was accepted under the maintainer's grace policy. These are the accepted leaderboard results, not a new ranking reconstructed from arbitrary PR titles or later experimental numbers.

| Place | Author and local write-up | Reported three-seed mean | Distinguishing changes | Source code |
|---|---|---:|---|---|
| 1 | [codemath3000: Calib32 + token-only n-gram tilt][winner-doc] | **1.05651** | Causal n-gram probability adjustments, asymmetric logit scaling, and 32-batch GPTQ calibration on an inherited recurrent-transformer stack | [train_gpt.py][winner-code] |
| 2 | [simonbissonnette: progressive context + short-document TTT][progressive-doc] | **1.05759** | Train at 1,024, then 2,048, then 3,072 tokens; adapt more frequently within short documents | [train_gpt.py][progressive-code] |
| 3 | [andrewbaggio1: long context + selective TTT][selective-doc] | **1.05855** | 2,560-token evaluation context, no query/value LoRA updates, and learning-rate/attention-gain tuning | [train_gpt.py][selective-code] |

The [original baseline][baseline-doc] scored **1.2244 BPB**. Using the rounded leaderboard values, the winner is approximately **13.7% lower** than that baseline, while first and second differ by only **0.10%**. This is a reduction in BPB, not a percentage improvement in general intelligence.

The top entries are related but not controlled one-variable experiments. For example, the winner uses n-gram tilt whereas the runner-up uses progressive 3k context and no n-gram predictor. Their validation-tail handling also differs: the winner and third-place entry report 47,851,520 scored targets, while the runner-up includes the tail and reports 47,853,343.

## What strategies worked?

The main lesson is **whole-pipeline optimization**, not a single architectural trick.

| Strategy | What the leading submissions did | Why it matters |
|---|---|---|
| Preserve quality after compression | Low-bit weights, GPTQ, mixed precision, small low-rank quantization-error corrections, and increasingly effective entropy compression | Training loss alone is not the submitted result. A slightly worse floating-point model can be better after fitting into the artifact budget. |
| Improve text representation | An 8,192-token vocabulary and lossless **CaseOps** capitalization operators | Lowercase lexical pieces can be shared across casing variants while retaining enough information to reconstruct the original text. |
| Reuse weights | Run selected transformer blocks repeatedly, alongside parallel residual paths and lightweight gates | Buy deeper computation without paying for another complete set of stored weights. |
| Make training faster | FlashAttention, fused GPU kernels, Muon refinements, and carefully tuned schedules | More useful learning fits into the fixed training window. |
| Spend the evaluation budget | Longer context, sliding evaluation, and score-first LoRA test-time training | Score a chunk first, then learn from those already-scored tokens to improve subsequent predictions. Never update from the answers before scoring them. |
| Combine inherited improvements | Build on public leaders, then measure small changes and interactions | The last winning change over PR #2130 was simply `GPTQ_CALIBRATION_BATCHES=16` to `32`; the substantial machinery already existed underneath it. |

For concrete implementations, the winner includes [CaseOps preprocessing][winner-caseops], [the n-gram probability adjustment][winner-tilt], and [the causal n-gram state implementation][winner-ngram]. The [runner-up's write-up][progressive-doc] gives a particularly useful table separating training-context changes, adapter selection, short-document updates, and inherited compression.

OpenAI's [official retrospective](https://openai.com/index/what-parameter-golf-taught-us/) reports **2,000+ submissions from 1,000+ participants**, widespread use of coding agents, and extensive reuse of earlier improvements. It explicitly distinguishes interesting research from leaderboard performance.

## Start with the baseline

Read the [baseline write-up][baseline-doc], then its [training script][baseline-code]. It is a 9-layer, 512-dimensional transformer with a 1,024-token vocabulary and tied input/output embeddings.

Its write-up separates pre-quantization BPB from the final int8-compressed roundtrip score. That distinction is the right starting point for understanding the rest of the competition: **model quality, export damage, artifact size, and evaluation behavior are different measurements**.

The root [train_gpt.py](train_gpt.py) and [train_gpt_mlx.py](train_gpt_mlx.py) are also retained. They are launching-off points, not the final winning configurations.

## Different models worth studying

These ten alternatives, together with the baseline and top three above, make up the fourteen-model reading list. The scores are **published or author-reported results**, not measurements from this checkout. This table is organized for learning, **not as an apples-to-apples ranking**.

| Model and write-up | Reported BPB | What to learn | Important qualification |
|---|---:|---|---|
| [Ternary U-Net transformer][ternary-doc] ([code][ternary-code]) | **1.1570** | Store most weights as `-1, 0, +1`; fit a reported 73.7M-parameter model through ternary packing, factored embeddings, and quantization-aware training | A record-track entry with a reported three-seed result. A different compression strategy from the final winners' 6-bit matrices. |
| [Binary U-Net transformer][binary-doc] ([code][binary-code]) | **1.1239** | Pack most weights into one bit and trade additional capacity against harder optimization; reported 106.2M parameters | The headline run trained for about **2.15 hours**, not 10 minutes. Read alongside the ternary study, not as a direct win over its 10-minute result. |
| [Masked diffusion language model][diffusion-doc] ([code][diffusion-code]) | **1.1465**, variational | Bidirectional masked-token prediction, timestep conditioning, and an absorbing-mask ELBO rather than ordinary autoregressive scoring | The write-up uses **approximate byte counting** and reports a 31-minute run on **2xH100**. Its extrapolated 8-GPU time is not a measured record. |
| [Mamba-3 hybrid][mamba-doc] ([code][mamba-code]) | **1.1473** | Combine five state-space blocks with two attention layers, then add quantization and score-first adaptation | A non-record hybrid study, not a pure SSM and not evidence that attention can simply be removed without cost. |
| [JEPA + Mamba-2 / LeWorldModel][jepa-doc] ([code][jepa-code]) | **1.2064** long run; **1.2566** 10-minute BPE run | Add latent-state prediction and anti-collapse regularization to a state-space language model; remove training-only auxiliary modules from the exported artifact | The headline result used about **2.7 hours**. The token-prediction head remains necessary for BPB evaluation. |
| [Byte-level H-Net][hnet-doc] ([byte code][hnet-code]) | **1.3595** at 4 hours; **1.4116** 10-minute mean | Learn chunk boundaries from raw bytes, process a compressed sequence, then expand back to byte predictions | Useful matched byte-versus-subword experiments and boundary analysis, but the headline score is not a 10-minute result. |
| [Universal Transformer][universal-doc] ([code][universal-code]) | **1.2249** | Three unique blocks reused four times, with iteration embeddings/scales; reported **4.95 MB** artifact | A single-seed non-record result near baseline quality. Strong weight sharing saves space but does not automatically improve the score. |
| [LegendreGPT][legendre-doc] ([code][legendre-code]) | **1.2266** | Generate layer weights as smooth functions of depth using learned Legendre-polynomial coefficients; 24 virtual layers | About **27 hours on one RTX 5090**; the headline score uses a separate post-hoc mixed-precision export. |
| [Learned adapters on random linear maps][random-doc] ([code][random-code]) | **1.1971** listed | Regenerate frozen MLP base matrices from seeds and store learned low-rank adapters instead of full dense matrices | These are stored, training-time adapters, not the same mechanism as evaluation-time LoRA. The listed score is one run; the write-up includes three runs. |
| [DG Attention][dg-doc] ([code][dg-code], [paper](paper/dg_attention.pdf)) | **1.1898** listed; **1.1554** later write-up | Deep layers send differential rather than absolute content; study matched controls, memory costs, and informative negative results | The later matched standard-attention result is **1.1516**. The write-up does not establish a quality win; initial metadata also needs the qualification below. |

### Keep the caveats attached to the numbers

Non-record does not mean uninteresting, and a lower number does not make differently evaluated runs comparable.

The MDLM entry's [metadata][diffusion-metadata] also reports 1,819 seconds of evaluation on its 2-GPU setup. Do not interpret its ELBO score or extrapolated training time as meeting the record track's full measured protocol.

DG Attention's [original metadata][dg-metadata] still records **1.1898 BPB and 16,638,468 bytes**, which exceeds the decimal 16 MB cap. Its write-up discusses later experiments with different results. Those are separate pieces of evidence, not interchangeable descriptions of one compliant run. The original files are preserved rather than silently rewritten to resolve this discrepancy.

OpenAI specifically highlighted the JEPA/Mamba-2, H-Net, and DG Attention submissions as interesting non-record work, **not necessarily the best-scoring alternatives**.

## Suggested reading order

1. **Understand the baseline's accounting.** Follow the model, optimizer, timed training loop, compressed export, and final evaluation. Keep pre-export and post-export scores separate.
2. **Read the third-place entry, then the runner-up.** The [selective-TTT write-up][selective-doc] isolates a small set of configuration changes; [progressive context][progressive-doc] shows a larger training/evaluation change on a related base.
3. **Read the winner last among the top three.** Trace its neural model, online n-gram adjustment, and score-before-update ordering separately. Its result is not attributable to the calibration-batch change alone.
4. **Compare ternary and binary together.** Focus on the difference between compression density and optimization difficulty under equal time budgets.
5. **Choose an architectural question.** H-Net for learned tokenization; Mamba/JEPA for recurrent state and training objectives; diffusion for a different likelihood/evaluation construction; Universal/Legendre/random adapters for weight sharing or generation.
6. **Read DG Attention as a negative-result study.** Compare against the matched standard model rather than assuming novelty implies improvement.

For each model, start with its write-up, inspect the linked entry point, and then inspect the supplied logs and metadata. Ask which quantities are held fixed, which ingredients changed, whether the displayed score is before or after quantization/adaptation, and whether the evidence is a single run or a multi-seed comparison.

## What this says about our attempt

Our exact submission has not been identified in this checkout, so this guide does **not** diagnose why it underperformed.

The defensible general lesson is that a standalone idea was competing against an accumulated, highly optimized pipeline. Missing an entire component such as good quantization or causal evaluation-time adaptation can matter, but this collection does not establish which component was missing from our own project or how much it cost. Novelty, scientific usefulness, and leaderboard rank should be assessed separately.

## Source provenance and running the code

The upstream base is commit [`f5c079314c4877fbb0af378c0abade5a8ca33d3a`](https://github.com/openai/parameter-golf/commit/f5c079314c4877fbb0af378c0abade5a8ca33d3a). All merged submissions and original support files are retained.

The following additional submission directories were copied from exact PR revisions without merging unrelated changes:

| PR | Local fetched ref | Pinned revision |
|---|---|---|
| [#2135](https://github.com/openai/parameter-golf/pull/2135) | `upstream/pr-2135` | `ff905226136eb2506fe9505feff87f58eb6dc4ff` |
| [#2014](https://github.com/openai/parameter-golf/pull/2014) | `upstream/pr-2014` | `fcd0b8357f58996a931a515c7da3ee0c999abc1e` |
| [#1953](https://github.com/openai/parameter-golf/pull/1953) | `upstream/pr-1953` | `5a47da61918fd330c88449456307f795b5c2b57e` |
| [#1110](https://github.com/openai/parameter-golf/pull/1110) | `upstream/pr-1110` | `8d582756935d3ac246bf9f6b4b5dbde2b56f877f` |

[model_sources.json](model_sources.json) maps every featured model to its directory, source revision, and training entry point. [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) retain the upstream licensing and attribution.

This is a **source study collection**, not a claim that all experiments have been reproduced. No GPU jobs have been launched or training datasets/neural checkpoints downloaded for this collection. Existing tokenizer files and submitted logs are included.

Use the [upstream setup guide](UPSTREAM_README.md#getting-started) and each entry's own instructions before attempting a run. CUDA, FlashAttention, Mamba, system-compressor, and dataset requirements vary, and some commands contain the authors' original cloud paths. The [MLX starter](train_gpt_mlx.py) is the Apple Silicon entry point; it is **not** a port of all the winning or alternative models.

[baseline-doc]: records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md
[baseline-code]: records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py
[winner-doc]: records/track_10min_16mb/2026-05-01_SP8192_PR2130Base_Calib32/README.md
[winner-code]: records/track_10min_16mb/2026-05-01_SP8192_PR2130Base_Calib32/train_gpt.py
[winner-caseops]: records/track_10min_16mb/2026-05-01_SP8192_PR2130Base_Calib32/lossless_caps.py
[winner-tilt]: records/track_10min_16mb/2026-05-01_SP8192_PR2130Base_Calib32/online_ngram_tilt.py
[winner-ngram]: records/track_10min_16mb/2026-05-01_SP8192_PR2130Base_Calib32/online_ngram_state.c
[progressive-doc]: records/track_10min_16mb/2026-04-30_SP8192_CaseOps_Progressive3k_ShortDocTTT/README.md
[progressive-code]: records/track_10min_16mb/2026-04-30_SP8192_CaseOps_Progressive3k_ShortDocTTT/train_gpt.py
[selective-doc]: records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_on_1945_1.0586/README.md
[selective-code]: records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_on_1945_1.0586/train_gpt.py
[ternary-doc]: records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/README.md
[ternary-code]: records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/train_gpt_cuda_ternary.py
[binary-doc]: records/track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear/README.md
[binary-code]: records/track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear/train_gpt_cuda_binary.py
[diffusion-doc]: records/track_non_record_16mb/2026-03-29_LLaDA_MDLM_Diffusion/README.md
[diffusion-code]: records/track_non_record_16mb/2026-03-29_LLaDA_MDLM_Diffusion/train_mdlm.py
[diffusion-metadata]: records/track_non_record_16mb/2026-03-29_LLaDA_MDLM_Diffusion/submission.json
[mamba-doc]: records/track_non_record_16mb/2026-04-15_Mamba3Hybrid_SP8192_GPTQ_TTT/README.md
[mamba-code]: records/track_non_record_16mb/2026-04-15_Mamba3Hybrid_SP8192_GPTQ_TTT/train_mamba3_hybrid.py
[jepa-doc]: records/track_non_record_16mb/2026-03-26_37M_LeWM_Jepa_Mamba2_10L_UNet_INT4FP8QAT_Brotli/README.md
[jepa-code]: records/track_non_record_16mb/2026-03-26_37M_LeWM_Jepa_Mamba2_10L_UNet_INT4FP8QAT_Brotli/train_jepa_ssm.py
[hnet-doc]: records/track_non_record_16mb/2026-03-29_HNet_ByteVsSubword_Study/README.md
[hnet-code]: records/track_non_record_16mb/2026-03-29_HNet_ByteVsSubword_Study/train_gpt_hnet_byte.py
[universal-doc]: records/track_non_record_16mb/2026-03-29_Universal_Transformer/README.md
[universal-code]: records/track_non_record_16mb/2026-03-29_Universal_Transformer/train_gpt.py
[legendre-doc]: records/track_non_record_16mb/2026-03-31_LegendreGPT/README.md
[legendre-code]: records/track_non_record_16mb/2026-03-31_LegendreGPT/train_gpt_legendre.py
[random-doc]: records/track_non_record_16mb/2026-04-30_Random_Linear_Adapter/README.md
[random-code]: records/track_non_record_16mb/2026-04-30_Random_Linear_Adapter/train_gpt.py
[dg-doc]: records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/README.md
[dg-code]: records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py
[dg-metadata]: records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/submission.json
