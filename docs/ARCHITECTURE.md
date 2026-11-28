# Architecture and implementation

EviSA enriches visible evidence, routes by original-transcript availability, and outputs a single continuous sentiment score. All paths operate on frozen feature caches.

| Component | Implementation under `src/evidence_msa/` | Role |
|---|---|---|
| Input preparation | `features/prepare.py` | Align cached features by sample ID |
| Routing | `inference.py` | Select the text-absent or text-present predictor |
| Text-absent path | `text_absent/v1.py` | Audio/visual evidence, visual dual streams, temporal pooling |
| TEF | `text_present/model.py` | Text anchor, shared/residual evidence, attention, ordinal output |
| Bilinear adaptation | `text_present/raw_adapters.py`, `adapters.py` | Residual interactions over visible evidence |
| Text anchors | `text_present/text_anchor.py`, `cell_text_anchor.py` | Text estimates and interval-aware corrections |
| Calibration | `text_present/calibration.py` | Increment gates, grouped consensus, interval and intensity adjustments |
| Factorized anchor | `text_present/factorized_anchor.py` | Separate polarity and magnitude estimates |
| Polarity repair | `text_present/runner.py:repair` | Confidence-conditioned sign repair near zero |
| Evaluation | `common/metrics.py` | Classification and regression from the same score |

The public name **TEF** corresponds to the legacy `C5` identifier. **ERU** covers the established adaptation, anchor, calibration, consensus, factorized-anchor, and near-zero repair pipeline; `D24` and `R3` are historical implementation identifiers. They remain in code and checkpoint names for compatibility.

The shared component is estimated by a training-only ridge mapping, rather than an additional observed modality. Missing channels cannot affect the prediction for a fixed visible set.

The text-present pipeline includes a main model and four video-grouped, cross-fitted auxiliary models. It is not a single-network forward pass and not a multi-seed ensemble. Its complete inference cost includes the components actually used at prediction time.

Interval calibration preserves the relevant classification boundaries. The final near-zero repair can change binary polarity, so the entire ERU pipeline is not class-preserving. Frozen parameters and selection rules are recorded in `configs/final_protocol.json` and `text_present/protocol.py`.
