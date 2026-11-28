# Third-party resources

Task-head code depends on PyTorch, NumPy, scikit-learn, joblib, and threadpoolctl. Install these separately.

Feature extraction uses CMU-MOSI/CMU-MOSEI and upstream author features, wav2vec 2.0, Whisper, BERT, MA-Net, Auto-AVSR, and EmotiEffLib. Depending on the extraction path, additional dependencies include FFmpeg, OpenCV, Transformers, MediaPipe, ESPnet, and ONNX Runtime. Actual asset provenance is recorded in `configs/input_provenance/`.

Datasets, pretrained weights, and third-party implementations remain subject to their respective terms. This repository does not relicense them. Frozen source snapshots preserve their existing notices and are retained for traceability and source-equivalence checks.

The current release does not assign a new blanket license to code of uncertain ownership. Baseline adapters require the corresponding upstream projects; their complete implementations are not bundled. The repository owner should confirm code ownership and upstream terms before adding a project-wide redistribution license.
