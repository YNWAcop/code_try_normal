# 使用与复现

## 数据和模型不在 GitHub 中

统一数据目录：`data/ready/{mosi,mosei}/{train,valid,test}.npz`。文件字段见 `INPUTS.md`。默认 `data` 在仓库内；也可通过 `--data-root` 指向已有目录。`configs/paths.example.json` 仅为路径说明，不会自动加载。`paths.local.json` 是本地参考，已被 Git 忽略。

当前机器已有数据位于 `D:/MSA_Cloud_20261115/data`；已有最终文本模型位于原工作区的 `C5_local_training/frozen_retrain_20261122/runs/{mosi,mosei}/fold_0`。此发布不迁移这些文件。一个数据集的推理必须保留整个 `fold_0` 中以下内容，不能只复制 C5 的一个 `best.pt`：

```text
selection_lock.json
prep_main/                 缩放器、共享成分与 data_complete.json
main/{C5,D5,anchor,cell}/best.pt
sub_{0..3}/prep/            各子模型独立拟合的预处理
sub_{0..3}/models/{C5,D5,anchor,cell}/best.pt
post/complete.pt
factor/best.pt
polarity.joblib
```

锁文件中的工件必须完整且哈希相符。A/V/AV 每个条件需要自己的 V1 `best.pt`、`scalers.joblib`、`validation_lock.json`。该权重必须来自原结果对应的训练；重新执行 `train-v1` 会得到新模型，不可称为旧结果的相同权重。

## 只做推理

```powershell
$py = 'E:\anaconda\envs\mult-repro-mulT\python.exe'
& $py run.py predict --condition LA --input 'D:\MSA_Cloud_20261115\data\ready\mosi\test.npz' --model-dir 'E:\UserData\Documents\ChatGPT\科研经历\C5_local_training\frozen_retrain_20261122\runs\mosi\fold_0' --output '.\runs\pred_mosi_LA.npz'
& $py run.py score --predictions '.\runs\pred_mosi_LA.npz' --labels 'D:\MSA_Cloud_20261115\data\ready\mosi\test.npz' --output '.\runs\metrics_mosi_LA.json'
```

`predict` 不取标签；`score` 单独按唯一 ID 对齐标签。同名输出存在时拒绝覆盖。`predict` 接受 A/V/AV；这时 `--model-dir` 必须换成对应 V1 条件目录。

## 固定训练

`train-text` 按 prepare → MOSI → MOSEI 执行；日志位于输出根的 `logs`，每 900 秒写入 `monitor_latest.json` 与 `monitor.jsonl`。该监控仅在命令进程存活时有效；不会自动重启机器或在电脑睡眠期间运行。

训练内缩放、共享分量拟合、分组交叉拟合和组件选择仍按原协议执行。`evaluate-text` 要求两数据集模型选择锁都存在。独立的 `train-v1` 命令仅在需要新无文本结果时使用；默认不会重训旧 A/V/AV。

不要将新输出目录指向原归档，以免混用代码冻结记录。若要修改算法，建立新版本和新的输出目录，并更新运行代码清单，避免把改动后的训练误标为当前最终版本。

## 指标

主要六项为 Acc7、Acc5、非零 Acc2、非零支持加权 F1、MAE、Corr。Acc/F1 原值为 0–1，论文中可乘 100；MAE 越小越好。Acc5/7 先裁剪再用 NumPy round-to-even；不能替换成另一种四舍五入。MSE、RMSE 与含零二分类仅作补充。常量输出的 Corr 标为 null。

## 本次验收范围

见 `release/verification.json`：模块导入、类别边界性质、可见模态隔离、训练源码函数一致性和已存文本权重的预测回放。本次不重新训练模型，也不通过回放结果调参。样本回放证明已检查样本的迁移一致性，不代表完成一次新的完整训练验收。

参考论文结果与历史开发使用的测试集有接触，因此不称独立确认结果；目录重构、固定重训或更换随机初始化均不能改变这一事实。
