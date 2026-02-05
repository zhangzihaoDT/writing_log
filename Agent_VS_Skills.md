# Agent 与 Skill：从概念混淆到工程分野

本文旨在厘清 Anthropic 最新提出的 **Agent Skills** 概念与传统语境下的 **Skill** 之间的本质差异，并从工程视角解构 **Agent (智能体)** 与 **Skill (技能)** 的边界。

## 一、 Anthropic Agent Skills vs. 传统 Agent Skill

**核心差异**：这是“模块化标准”与“内部功能实现”的区别。

### 1. 传统 Agent 中的 "Skill"

- **定义**：Agent 内部预定义的单一能力（Function/Tool）。
- **形态**：通常硬编码在 Agent 的 System Prompt 或代码逻辑中（例如：“你是一个擅长写代码的助手”）。
- **局限**：
  - **Context 爆炸**：所有技能的定义都必须塞进上下文，导致 Token 消耗巨大。
  - **不可复用**：技能与 Agent 深度绑定，难以迁移。

### 2. Anthropic 定义的 "Agent Skills"

- **定义**：独立于 Agent 的、模块化的、可复用的能力包标准。
- **机制**：**渐进式披露 (Progressive Disclosure)**。
  - Agent 运行时仅加载 Skill 的 **Metadata** (元数据)。
  - 只有当 Agent 决定调用该 Skill 时，才会动态加载具体的 Prompt 和工具定义。
- **本质**：它是 Agent 生态的**“可插拔插件 (Plugin)”**，实现了能力与载体的解耦。

---

## 二、 Agent 与 Skill 的工程分野

如果将 Agent 系统视为一家公司，**Agent 是拥有主权的决策者 (CEO/Manager)，Skill 是被调度的 SOP (标准作业程序)。**

### 1. Framework 维度：谁负责闭环？

**Agent 是“状态机 + 决策者”。** 它必须站在 Framework 层，负责全局的闭环。

- **Plan (规划)**：拆解目标，判断当前该做什么。
- **Memory (记忆)**：记住历史交互，提取经验。
- **Context (上下文)**：控制 Token 预算，决定信息取舍。

**Skill 是“被调度的行为结构”。** 它只负责把一段“可复用的做事方式”跑清楚。

| 职责              | Agent           | Skill                 |
| :---------------- | :-------------- | :-------------------- |
| **决策 (Plan)**   | ✅ **核心职责** | ❌ 无权决策           |
| **记忆 (Memory)** | ✅ 全局记忆     | ❌ 无记忆 (Stateless) |
| **执行 (Act)**    | ✅ 调度者       | ⚠️ 仅定义步骤         |

### 2. Tools 维度：谁在直接干活？

不要混淆 Tool (工具) 与 Skill (技能)。

- **Tool = 枪 / 炮 / 雷达**
  - _定义_：原子的、无业务含义的功能。
  - _示例_：`query_sql(table, condition)`, `http_get(url)`.
- **Skill = 战术手册**
  - _定义_：工具 + 顺序 + 约束 + 判断规则。
  - _示例_：“销量分析 Skill” = 1.校验口径 → 2.调用`query_sql` → 3.异常值清洗 → 4.生成解释。
- **Agent = 指挥官**
  - _定义_：决定用不用战术，用哪个战术，战术失败了怎么办。

### 3. Sandbox 维度：谁对体验负责？

- **Agent (产品界面)**：Agent 直接面向用户，负责整体体验（User Experience）。它需要处理对话的连贯性、错误时的安抚（兜底策略）以及用户意图的最终确认。
- **Skill (黑盒执行)**：Skill 运行在 Agent 提供的沙箱（Sandbox）中。它不直接面对用户，只对“输入”和“输出”的正确性负责。

---

> **总结**
>
> - **Agent** 管全局，负责“做正确的事” (Do the right things)。
> - **Skill** 管局部，负责“把事做正确” (Do things right)。

![AI Stack V2](/Users/zihao_/Downloads/ai_stack_v2_opt1_portrait.png)
