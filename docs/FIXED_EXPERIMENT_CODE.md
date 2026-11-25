# 固定实验代码及修复

本次为代码快照同步，原仓库根目录的固定主模型保持原样。新增目录位于 `experiments/`：

| 目录 | 用途 | 主要入口 |
|---|---|---|
| cloud_ablation_20261123 | 固定 A0–A4 的 T/TA/TV 消融 | ablation.py |
| cloud_av_ablation_20261124 | 固定 Native/Enhanced 的 A/V/AV 消融 | av_ablation.py |
| baseline_repair_20261123 | EMT/MISA 缺失设置修复，含修复补丁 | adapter_emt_tfr.py、adapter_misa_almt.py、run_emt_queue.py |
| local_ours_tav_20261124 | Ours TAV 扩展 | local_tav.py |
| cloud_tav_20261124 | 九种基线的 TAV 适配 | tav_queue.py |
| cloud_tav_rome_tmdc_20261124 | RoME/TMDC 的 TAV 适配 | supplement_queue.py |

仅包含当前结果对应版本。`tef_tatv_experiments_20261124`、`tef_v2_20261125` 及 E0/E1/E2 等后续探索均不在本次新增目录中。未启动训练，未修改种子策略。

## 组件对照

A0 = Native + TEF；A1 = Enhanced + TEF；A2 = Native + TEF + ERU；A3 = Enhanced + TEF + ERU；A4 = Enhanced + Concat。B0/B1 为相同 A/V/AV 分支的 Native/Enhanced 对照。

TEF 为 Text-guided Evidence Fusion，即原内部 C5 展示名称。ERU 指既定完整后处理修正流水线。A0/A2 共享 Native 主干，A1/A3 共享 Enhanced 主干。代码中保留 C5 标识，避免破坏检查点兼容和历史源码哈希。

输入增强使用 B1/B0、A1/A0 或 A3/A2；TEF 使用 A1/A4；ERU 使用 A2/A0 或 A3/A1。不能把 A3/A4 的联合差异全部解释为 TEF。T/A/V 表示可用模态。

## 运行前必须补齐的依赖

本次是**代码归档，不是完整可直接运行的实验包**。为限定公开同步范围，没有上传新增结果 CSV、验证报告、历史运行清单、数据、权重、逐样本预测或第三方源码目录。部分入口会校验这些外部清单；在缺失时失败属于预期保护，不应绕过校验。

- Ours 消融需要本地原实验包的 `expected_data.json`，以及 `release/runtime_manifest.json`（或 A/V/AV 包的 `manifest.json`）；数据需为冻结的 `data/ready/{mosi,mosei}/{train,valid,test}.npz`。
- Ours TAV 同样需要原正式包的输入与运行源码清单。它是独立训练的扩展版，不能把其 T/TA/TV 辅助输出替换旧冻结主表。
- EMT/MISA 修复需要原实验目录的 vendor/emt_dlfr 与 vendor/mmsa、对应配置、数据、预训练文本模型、修复训练计划及前置检查文件。修复补丁保留在本目录；第三方完整实现仍由原始本地依赖提供。
- 基线 TAV 包需要原正式包的 manifest、上游实现及模型；`cloud_tav_20261124/four/vendor/` 和 RoME/TMDC 的上游源码没有随本次上传。适配器的具体 import 与路径为依赖依据。

本地原实验文件仍保留。需复跑时，从对应原实验包补齐上述依赖和历史清单，在新输出目录运行；不要把不同包中的同名 `evidence_msa` 混用，不要覆盖已完成结果。

原始启动脚本和部分队列保留历史 Windows/云端路径，迁移到其他机器前须检查实际数据及输出路径。为保留代码对应关系，本次没有对训练源码做路径或算法重写。可查看各入口的 `--help`；迁移修改后的副本应另行记录版本。

## 修复要点

EMT 修复恢复预定重建维度设置与原生物理 batch；MISA 修复恢复 CMD/DiffLoss 所需的原生 batch 统计，保留原损失权重。映射加载、验证选模、冻结测试与队列控制代码一并归档。

本次拷贝检查确认代码与原工作区一致，既有原模型、文本消融及 Ours TAV 共 15 项机制测试通过。没有重新跑基准训练，也不声称单次结果具有多种子稳定性。第三方依赖仍须遵守原许可证。
