# 输入特征与来源

`ready/*.npz` 包含 `T,A_native,A_words,V_native,V_lip,V_words,V_expression,V_quality,V_timeline,ids,groups,y`。所有样本按 IDs 对齐，groups 是原视频分组。特征维度从缓存读取，不能把不同模型提取的同维度数组直接替换。

| 输入 | 含义 |
|---|---|
| T | 原文本的冻结表征 |
| A_native / V_native | 原作者音频/视觉特征 |
| A_words | Whisper 音频识别文本经 BERT 编码 |
| V_lip / V_words | 纯视觉 Auto-AVSR 表征及视觉识别文本编码 |
| V_expression | 表情模型特征 |
| V_quality | 视觉有效性与质量信息 |
| V_timeline | 视觉语音时序表征 |

具体提取协议保存在 `configs/input_provenance`。ASR 从音频提取，VSR 从 RGB 提取，不把原文本标签或音频输入视觉识别器。用于输入的外部预训练语音、视觉语音、表情模型需在论文中说明。

`prepare-inputs` 将已有 `data/author` 和 `data/generated` 缓存打包成 ready；它不会自动下载数据或预训练模型。精确路径布局见 `src/evidence_msa/features/prepare.py`。

`archive/input_extraction` 保存原始输入提取实现作溯源，包括音频缓存、Whisper、纯视觉语音、表情和 BERT 词编码。它们保留当时的数据路径，需要自行配置外部代码和模型，不属于已验收的跨机器一键提取入口。`mosei_vsr_common.py` 中还包含历史辅助函数；当前最终训练只导入 `src/evidence_msa`，不导入 archive。

本仓库不分发 MOSI/MOSEI 原始音视频、样本文本、标注、模型权重或第三方大型代码库。复现已有最终模型优先使用本机已经核验的 ready 特征，避免重新提取带来的差异。
