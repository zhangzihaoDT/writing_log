我将扩展该脚本：添加 LS9/CM2 专用筛选，并实现完整的 PSM 流程与报告。

**实现说明**

- 新增专用筛选与 PSM 流程，保持原有筛选逻辑不变的前提下，仅保留`车型分组`包含“LS9”或“CM2”的样本，并将`deposit_amount`映射为 5000（LS9）与 2000（CM2），构造处理变量`treat_high_deposit=1/0`。
- 完整 PSM 步骤：
  - Step1 倾向得分(Logistic)：用除`deposit_amount`与`车型分组`之外的协变量估计`treat_high_deposit`的概率（含目标/频次编码、数值标准化与 One-Hot 的统一预处理）。
  - Step2 最近邻匹配：1:1 且不放回，`caliper=0.05`，基于倾向得分的绝对差做贪心匹配。
  - Step3 平衡检验：计算匹配前后协变量的 SMD，报告最大 SMD 与匹配后`SMD<0.1`比例。
  - Step4 ATT：对匹配对的`purchase`差异做平均，给出 ATT 与 95%置信区间，并生成报告。

**如何运行**

- 直接运行：
  - `python skills/intention_order_binary_model.py`
- 生成两份报告：
  - `models/deposit_effect_report.md`（消融对比：base/deposit-only/full）
  - `models/psm_deposit_report.md`（PSM 四步结果：样本量、倾向模型 AUC、匹配对数、SMD 前后、ATT 与 95%CI）
- 控制台会同步打印 PSM 关键指标，便于快速查看。

**协变量选择（用于倾向模型与 SMD）**

- 类别（One-Hot）：`order_gender`（若存在）
- 数值：`buyer_age`、`interval_touch_to_pay_days`、`interval_assign_to_pay_days`、`is_repeat_buyer`、`cumulative_order_count`、`last_invoice_gap_days`、`interval_presale_to_pay_days`
- 目标/频次编码来源：`first_main_channel_group`、`Parent Region Name`、`License City`
- 不包含：`deposit_amount`与`车型分组`（避免把处理本身或其代指纳入协变量）

**输出与判断**

- 倾向模型的`AUC`用于衡量可分性（非业务效果），SMD 用于检验匹配后是否达到学术级平衡标准（`SMD<0.1`）。
- `ATT`量化定金（5000 vs 2000）对锁单率的因果影响（处理组视角），95%CI 用于业务汇报表达不确定性。

**你可以期待的结论方式**

- 若匹配后`SMD<0.1`比例高、最大 SMD 明显降低，且`ATT`接近于 0（CI 也跨越 0 或很窄接近 0），即可有力支持“定金金额不是影响锁单率的主要原因”。
- 若`ATT`显著不为 0，且在良好平衡下成立，则定金金额存在可量化的因果影响。

需要我把 PSM 协变量按你的业务偏好再微调（比如加入门店或渠道的更细分变量），或者把报告中的平衡表按特征逐项列出吗？我可以继续完善输出格式和细节。
