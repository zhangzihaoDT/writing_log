"你的这个文件就是一个 典型的数据查询 Skill YAML，完全符合 Claude Code 高级 Skill 的结构。
它可以直接作为 Query Agent 的 Skill 被 Claude 调用，帮你把自然语言请求自动映射成数据查询参数。"

根据 [query_skills.yaml](file:///Users/zihao_/Documents/github/W52_reasoning/agents/query_skills.yaml) 的配置，Query Agent 将自然语言转化为数据查询指令的完整流程如下：

```mermaid
graph TD
    %% 用户输入
    Start([用户自然语言输入]) --> Input["例如: 'LS6 过去一周 每一个车系的锁单量'"]

    %% LLM 解析层 (基于 query_skills.yaml)
    subgraph LLM_Reasoning [Query Agent 逻辑解析层]
        direction TB

        %% 指标映射
        Mapping1["<b>指标映射 (Metrics)</b><br/>别名 -> 标准指标名<br/>例: '销量' -> '锁单量'"]

        %% 维度与过滤映射
        Mapping2["<b>维度与过滤映射 (Dimensions)</b><br/>提取: 车系=LS6, 过滤字段: series<br/>模糊匹配: 城市/门店包含匹配"]

        %% 工具选择规则
        Rules{"<b>工具选择规则 (Rules)</b>"}

        Rule1["单点查询 -> query"]
        Rule2["维度拆解 -> rollup"]
        Rule3["趋势/时序 -> query + interval"]

        Mapping1 --> Mapping2
        Mapping2 --> Rules
        Rules --> Rule1
        Rules --> Rule2
        Rules --> Rule3
    end

    Input --> LLM_Reasoning

    %% 参数组装层
    subgraph Param_Build [参数标准化组装]
        direction TB
        P1["metric: '锁单量'"]
        P2["date_range: 'last_7_days'"]
        P3["filters: series = 'LS6'"]
        P4["dimension: 'series' / interval: 'day'"]
    end

    LLM_Reasoning --> Param_Build

    %% 执行层
    subgraph Execution [底层工具执行]
        direction LR
        Tool1{{QueryTool}}
        Tool2{{RollupTool}}
    end

    Param_Build --> Execution

    Execution --> SQL[生成 SQL 并执行]
    SQL --> Output([结构化数据返回])

    %% 示例标注
    style LLM_Reasoning fill:#f9f,stroke:#333,stroke-width:2px
    style Execution fill:#bbf,stroke:#333,stroke-width:2px
```

### 流程核心环节说明：

1.  **语义对齐 (Metric & Dimension Mapping)**：
    - 利用 `metrics` 列表将用户的非规范词（如“销量”、“提车数”）映射为底层物理表对应的 `tool_metric`。
    - 利用 `dimensions` 列表识别用户提到的筛选条件（如“车系”、“城市”），并根据 `filter_field` 确定 SQL 过滤字段。

2.  **逻辑判断 (Rules)**：
    - **工具分流**：如果用户提到“分布”、“按...拆分”，触发 `rollup` 工具；如果提到“趋势”、“每天”，触发带 `interval` 参数的 `query` 工具。
    - **特例处理**：如 `Series vs Series Group` 规则，防止将普通车系名（LS6）错误映射到代码字段（CM2）。

3.  **参数标准化**：
    - 将解析结果组装成 JSON 参数（如 `filters: [{field: 'series', op: '=', value: 'LS6'}]`），这是连接自然语言与底层 Python 工具（`query.py`, `rollup.py`）的标准协议。

4.  **Few-Shot 引导**：
    - `examples` 模块作为 LLM 的“教科书”，通过具体的输入输出对，确保 Agent 在面对复杂组合查询（如“LS6 纯电 2025年12月 上海的开票数”）时能保持极高的转换准确率。
