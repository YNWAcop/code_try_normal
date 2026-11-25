# Evidence-enhanced multimodal sentiment analysis

## 固定实验代码补充（2026-11-25）

新增 `experiments/`，保存当前固定结果所用的 A0–A4、A/V/AV 消融代码，以及 EMT/MISA 修复、Ours 与基线 TAV 扩展代码。原主模型不变，后续 TEF 改进尝试未加入。表格中原 C5 称为 **TEF（Text-guided Evidence Fusion）**，代码标识为兼容历史权重保留原名。

本次仅同步代码、补丁和使用说明，不包含新实验结果、实验验证记录或第三方完整源码。运行所需的外部资产及历史清单见 [固定实验代码说明](docs/FIXED_EXPERIMENT_CODE.md)。

本仓库保存 **2026-11-22 最终固定版本**，对应当前论文结果：无文本 A/V/AV 沿用 V1；有文本 T/TA/TV 使用固定重训的 **C5 + D24 + R3**。T=L，TA=LA=AT，TV=LV=VT。这里的 C5/D24/R3 是实验内部名称，并非新的基线方法名称。

代码包含模型、输入特征打包、训练、预测、评测、冻结协议、原始源码快照及参考结果。**没有打包数据、预训练模型或实验权重**；这些文件在本机原目录保留，GitHub 不包含它们。复制代码不会移动、删除原实验，也不会启动训练。

## 目录

```text
run.py                         统一命令入口
src/evidence_msa/
  features/                    已提取特征的对齐与打包
  text_absent/v1.py            A/V/AV 的 V1 网络和固定条件训练
  text_present/
    model.py                  C5 主干：共享/残差证据、注意力、分布预测
    raw_adapters.py            可见模态双线性交互修正
    text_anchor.py            文本锚点
    cell_text_anchor.py       分类区间约束的文本修正
    factorized_anchor.py      极性与强度分解
    calibration.py            D24：折外共识、门控、区间内校准
    runner.py                 R3 近零极性修复、固定测试评测
    train.py                  冻结的组件训练与模型选择
    data.py                   训练内缩放、共享成分估计、分组交叉拟合
    protocol.py               实际训练参数和选择规则
    supervisor.py             执行队列与每 15 分钟状态记录
  inference.py                 有/无文本统一路由，预测不读取标签
  common/                      指标、检查点与锁
configs/                       最终协议、输入来源与路径示例
results/reference/             当前最终结果 CSV/JSON
tests/                         机制与缺失模态隔离检查
release/                       源码哈希、运行代码哈希与验收记录
archive/frozen_sources/         整理前的最终源码原样快照
archive/input_extraction/       输入提取历史实现（需外部模型与路径配置）
docs/                          架构、数据和复现说明
```

## 使用

Python 3.10；已核验环境为 PyTorch 2.3.1+cu118、NumPy 1.24.4、scikit-learn 1.3.2。文本分支训练与完整推理需要 NVIDIA CUDA。先安装与本机驱动相容的 PyTorch，再安装 `requirements.txt`；原有兼容环境可直接使用。

在仓库目录的 PowerShell 执行（将 Python 和数据路径换成实际位置）：

```powershell
$py = 'E:\anaconda\envs\mult-repro-mulT\python.exe'
& $py run.py --help
& $py run.py --data-root 'D:\MSA_Cloud_20261115\data' check
& $py -m unittest discover -s tests -v
```

如需另起一次固定训练，使用**新的输出目录**：

```powershell
& $py run.py --data-root 'D:\MSA_Cloud_20261115\data' --run-root '.\runs\new_fixed_run' train-text
& $py run.py --run-root '.\runs\new_fixed_run' status
```

训练不会自动评测测试集。两数据集模型均锁定后，显式执行：

```powershell
& $py run.py --data-root 'D:\MSA_Cloud_20261115\data' --run-root '.\runs\new_fixed_run' evaluate-text
```

同一次训练的参数和代码被冻结，可用相同命令从已有检查点继续。没有手动固定随机种子或种子扫描；恢复随机数状态仅用于断点续训。新训练的数值不保证与已保存论文结果逐位一致。

预测与评分示例、文件依赖、原始权重位置参见 [使用与复现](docs/REPRODUCE.md)。算法及图示对应关系见 [架构说明](docs/ARCHITECTURE.md)，输入来源见 [输入特征](docs/INPUTS.md)。

## 结果范围

`results/reference` 是既有固定模型结果，不是本次代码整理重新训练的结果。训练和推理均使用冻结的上游特征，不等同于从原始音视频端到端训练。最终方法包含主模型与四个交叉拟合子模型；应如实说明集成和校准成本。

历史开发中已查看过原测试集，固定重训不会消除该历史；参考结果不能称为全新独立测试确认。相关字段保留在协议中。本仓库不承诺全部指标超过基线。

第三方模型和数据的来源及使用边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。
