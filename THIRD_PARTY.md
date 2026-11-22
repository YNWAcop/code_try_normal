# 第三方依赖与来源

训练实现依赖 PyTorch、NumPy、scikit-learn、joblib、threadpoolctl；这些依赖未复制进仓库。

输入流水线依赖原始 MOSI/MOSEI 数据和原作者特征、Whisper、BERT、Auto-AVSR、EmotiEffLib，以及对应实现使用的 FFmpeg、OpenCV、Transformers、MediaPipe、ESPnet、ONNX Runtime。下载与使用须遵守各自数据及模型条款。`configs/input_provenance` 保存本实验实际模型及参数信息。仓库不提供这些资产的再许可。

`archive/frozen_sources` 是研究工作区的源码快照，保留源文件原有声明；它仅用于追溯，不代表整个上游项目或完整复现环境。没有为未知来源代码添加统一开源许可证；若要授予公开再利用许可，应先由仓库所有者确认代码权属及依赖许可。

RoME/TMDC 等方法只作为参考结果中的基线，本次发布未复制这些基线的完整实现。
