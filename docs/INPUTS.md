# Input features and provenance

Prepare `data/ready/{mosi,mosei}/{train,valid,test}.npz`. Each file contains:

| Key | Meaning |
|---|---|
| `T` | Frozen representation of the original transcript |
| `A_native` | Native acoustic representation |
| `A_words` | BERT-encoded Whisper transcription |
| `V_native` | Native visual appearance representation |
| `V_lip` | Visual-only Auto-AVSR representation |
| `V_words` | Encoded visual-only recognized words |
| `V_expression` | Expression features |
| `V_quality` | Visual validity and quality descriptors |
| `V_timeline` | Temporal visual speech representation |
| `ids`, `groups`, `y` | Sample IDs, source-video groups, continuous labels |

All arrays must use the same sample order. Feature dimensions are read from the cache. Equal dimensionality does not make features from different extractors interchangeable. Scalers and learned preprocessing are fitted on training data.

ASR uses the available audio. VSR uses the current utterance's RGB frames, not audio or the original transcript. Derived words belong to the channel that generated them. The cached schema may store all feature keys; the selected path masks or excludes unavailable evidence.

Model versions, preprocessing, and weight hashes are in `configs/input_provenance/`. The expression extractor uses EmotiEffLib; visual speech uses Auto-AVSR. External pretraining is part of the input protocol.

`python run.py --data-root ./data prepare-inputs --dataset both` packs existing `author/` and `generated/` caches. Consult `src/evidence_msa/features/prepare.py` for their exact layout. This command is not an end-to-end raw-media extractor.

Historical extraction implementations are retained under `archive/input_extraction/`. They require external models, third-party packages, and path adaptation. They are provenance references, not a portable one-command download pipeline.

| Dataset | Train | Validation | Test | Non-zero test labels |
|---|---:|---:|---:|---:|
| MOSI | 1,284 | 229 | 686 | 656 |
| MOSEI | 16,326 | 1,871 | 4,659 | 3,634 |

Raw media, sample transcripts, labels, feature arrays, pretrained models, and trained weights are not distributed in this repository.
