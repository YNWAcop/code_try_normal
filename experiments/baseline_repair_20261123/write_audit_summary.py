import difflib,json,shutil,time
from pathlib import Path
from metrics_common import digest,atomic_json
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'whole_priority_20261117'
manifest=ROOT/'repair_manifest.json'
if not (ROOT/'repair_manifest_initial.json').exists():shutil.copy2(manifest,ROOT/'repair_manifest_initial.json')
changes=[]
for name in ('adapter_emt_tfr.py','adapter_misa_almt.py','mmsa_slim_pickle.py','vendor/emt_dlfr/models/missingTask/EMT_DLFR.py'):
    before=OLD/name;after=ROOT/name
    changes.append({'file':name,'previous_sha256':digest(before),'current_sha256':digest(after)})
    patch=''.join(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True),after.read_text(encoding='utf-8').splitlines(True),fromfile='historical/'+name,tofile='repair/'+name))
    (ROOT/(name.replace('/','__')+'.patch')).write_text(patch,encoding='utf-8')
atomic_json(manifest,{'updated':time.time(),'scope':['EMT_DLFR','MISA'],'changes':changes,
 'EMT':'Public tensor-width reconstruction grid; original fusion, SimSiam, reconstruction objectives retained; native physical batch MOSI32/MOSEI16',
 'MISA':'Native physical batch MOSI16/MOSEI64; CMD/DiffLoss batch statistics restored; uses LayerNorm, not BatchNorm',
 'runtime':'Read-only memmap loader, per-sample identical float32 conversion, checkpointing with original stochastic behavior; no manual seeds',
 'checks':['emt_contract_checks.json','mapped_loader_checks.json','input_and_batch_loss_audit.json','misa_memory_mosi.json','misa_memory_mosei.json'],
 'plan':'repair_retrain_plan.json','policy':'Replace every affected row after verification regardless of score; old results retained; no test-driven retries',
 'provenance':'Original whole-modality adaptations; not exact reproduction of original random-missing paper protocols'})
report=ROOT.parent/'outputs/code_protocol_audit_20261123/审计报告.md'
backup=report.with_name('审计报告_修复前.md')
if not backup.exists():shutil.copy2(report,backup)
s=report.read_text(encoding='utf-8')
s=s.replace('MISA每个微批量backward后裁剪累积梯度；BatchNorm和批内差异/相似性损失也受微批量影响。','MISA每个微批量backward后裁剪累积梯度；CMD及批内差异损失受微批量影响。经进一步核查，这份MISA实现使用LayerNorm，先前提及BatchNorm不准确；EMT的投影头另有BatchNorm。')
report.write_text(s,encoding='utf-8')
text='''# 基线复核与修复记录（2026-11-23）

本轮核查以“保留各论文核心方法，适配相同整模态缺失条件”为准，不要求基线使用 Ours 的增强输入。训练仍使用原始 train/valid 划分和各方法预先规定的验证选择规则。修复后的测试预测与旧表使用相同样本和指标；不按测试分数决定是否保留某次重训。

## 已确认、需要重训的问题

| 方法 | 代码证据与影响 | 修复与验证 |
|---|---|---|
| EMT-DLFR | 整条音视频缺失时，不完整分支长度被设为1；重建目标随输出长度截断，仅监督一个位置，缩小了低层特征恢复任务。微批次也改变投影头的批统计。 | 缺失流使用数据张量的公共宽度，避免读取隐藏样本长度；保留原EMT、SimSiam及恢复损失。六条件隐藏内容/长度隔离、后续位置重建梯度、checkpoint输出/梯度一致性通过。恢复MOSI32、MOSEI16物理批次。 |
| MISA | 物理批次缩为4后累积梯度，不能还原CMD与DiffLoss的批内统计；合成同一批样本拆分前后损失与梯度均不同。 | 恢复MOSI16、MOSEI64物理批次及原更新间隔、权重。两数据集显存探测、MOSEI真实数据原批次训练/验证权重回放通过。使用LayerNorm，早先称BatchNorm的问题已更正。 |

全部24个配置已预先登记：两个方法 × 两个数据集 × 六个模态条件。一次训练，无手动种子、无种子搜索；选模锁定后先验证回放，再测试。每15分钟保存监控记录。当前进度以 queue_status.json 为准，不能把已启动写成已全部完成。

MOSEI原加载器曾因一次性展开大数组触发MemoryError。现改为只读内存映射和逐样本float32转换；真实MOSI新旧加载器逐元素一致，协议3/4的数组布局测试通过。修订记录保留在runtime_revision.json；先完成的模型未重训、未更换权重。

## 退化配置的复核结论

| 方法/条件 | 复核证据 | 当前处理 |
|---|---|---|
| ALMT / MOSI A | 原权重验证预测与归档最大差4.17e-7；音频输入梯度与编码权重梯度非零；去掉音频会改变输出。验证输出近常数，后期训练虽有变化但验证MSE未优于选中的早期模型。 | 保留原结果，未发现音频分支断开；不因测试低分改选模型。 |
| API / A、V | 七类原型可更新并注册为checkpoint缓冲区；原型对比、六向重建、可见投影梯度均有效；推理不传标签，改变隐藏输入不改变输出。 | 保留原结果。合成机制检查支持核心路径存在，不等于证明每次真实训练都充分收敛。 |
| Self-MM / MOSI A | 实际网络四头、原加权损失、伪标签与中心更新可执行；音频输入梯度非零，音频参数完成更新。 | 保留原结果。旧权重未在本地找到，不能宣称已重新回放该旧权重；缺失头仍保留原辅助目标。 |
| TFR-Net / MOSEI A | 跨模态注意力、三个生成器与C_GATE保留；缺失视觉重建后续位置有梯度；可见音频对预测有梯度；隐藏输入变化不影响预测。 | 不存在已证实的EMT式单位置重建截断，不照搬EMT修复。实现来源为EMT工程内TFR版本，须如实标注。 |
| EBMC / MOSI V | 两阶段及MSD/CCE/EMC/IMTD保留；损失系数与作者MOSI/MOSEI正式启动脚本一致（MSD=.5，其余=.1），不能以parser默认值误判。 | 暂无应改权重或重训的实现证据。保留作者特征、上下文和验证选模适配说明。 |

这些检查不能把“没有发现错误”升级成“严格复现论文全部结论”，也不能证明退化只有一个原因。低分行不会被删除、补造或用测试选择更好的一次运行替代。

## 结果表发布条件

finalize_tables.py 已在旧表144个配置、864项数值上重新核算通过，并核对已完成修复预测的哈希、样本ID、标签和指标。只有全部24组完成并通过检查，才生成 outputs/baseline_repaired_tables_20261123 下的新CSV、六条件平均值、逐指标修复差值和verification.json。旧表不覆盖，Ours及其他基线行保持原值。

同一测试集、同一模态可见定义和同一指标使数值能够放在同一张系统比较表。各方法使用的表征、预训练资源、上下文、选模规则及实现来源仍应列在实验设置中。该表不能单独证明Ours下游模块优于其他方法的下游模块；这需要相同输入的消融。历史测试暴露也不会因为本次重训被消除。

## 可追溯文件

- repair_retrain_plan.json：固定24组任务、代码哈希、选择规则。
- repair_manifest.json 与 *.patch：修复范围及前后源码差异。
- emt_contract_checks.json：六条件恢复与隔离检查。
- mapped_loader_checks.json：内存映射与原加载器数值一致性。
- input_and_batch_loss_audit.json：真实输入有限性及批内损失差异。
- almt_mosi_A_diagnostic.json、api_mechanism_checks.json、selfmm_mechanism_checks.json、tfr_mechanism_checks.json：诊断结果与适用范围。
- queue_status.json、monitor_15min.jsonl：执行进度。
- table_gate_status.json：结果表发布前的完整性检查。

上游入口：[EMT-DLFR](https://github.com/sunlicai/EMT-DLFR)、[MISA](https://github.com/declare-lab/MISA)、[MMSA](https://github.com/thuiar/MMSA)、[TFR-Net](https://github.com/thuiar/TFR-Net)、[ALMT](https://github.com/Haoyu-ha/ALMT)。本轮具体差异以上述保存的训练源码和哈希为准。
'''
(ROOT/'复核与修复报告.md').write_text(text,encoding='utf-8')
print('Audit report and current repair manifest saved; historical versions preserved.')
