---
name: research-briefing
description: 用 InsightFlow 的研究工具做一次有引用、可追溯的动态简报：固定时间窗口，检索并读原文，逐条提交带引文的结论，出报告前做质量校验。适用于“最近一周某主题有什么新动态/新发布/新论文”这类问题。
---

# 研究简报（调用 InsightFlow 工具）

你不是在写一篇作文，而是在**收集可核对的证据**，然后让工具把证据组装成报告。
工具会拒绝任何对不上原文的引用，所以不要凭记忆写引文。

## 什么时候用

用户问的是“最近一段时间某主题发生了什么”，例如：

- 最近一周 AI Agent 评测有什么新动态
- 过去一个月开源模型发布了什么
- 某篇论文之后有没有后续工作

如果用户只是问常识、让你解释概念，不要走这套流程。

## 必须按这个顺序

1. **`start_research`** — 先固定问题、`as_of` 与窗口（默认 7 天）：
   ```
   start_research(question="最近一周 AI Agent 评测动态", days=7)
   ```
   返回 `run_id`、`window_start`/`window_end`、`required_themes`、预算。之后每个工具都要带这个 `run_id`。
   窗口是左闭右开：早于 `window_start` 或等于/晚于 `window_end` 的来源都不算“窗口内新增”，工具会在 `in_window` 里标出来。

2. **找来源**（可以多次，但别滥搜）：
   - `search_web(run_id, query)` — 网页；
   - `arxiv_search(run_id, query)` — 论文元数据（要引用摘要仍需 `read_source` 读 abs 页）；
   - `github_releases(run_id, repo)` — 仓库版本。
   每次 2-4 条检索式即可，先用 `required_themes` 里的主题各来一条。

3. **`read_source(run_id, url)`** — 摘要只是线索，引用必须来自正文。返回值里有 `source_id`、`text`、`published_at`、`in_window`、`fetch_scope`。
   读 2-4 篇就够；读失败（`not_found` / `blocked_address` / `unsupported_content`）就换一篇，别反复重试同一个 URL。

4. **`submit_findings(run_id, findings=[...])`** — 一次交一批结论（也可以一次一条用 `submit_finding`）。
   批量是逐条独立校验的：通过的收下，没通过的按 `index` 退回，改完只重交那几条。每条结构：
   ```json
   {
     "text": "某实验室发布了 Agent 评测框架 X，覆盖工具调用与长程任务。",
     "claim_type": "publisher_claim",
     "support_status": "partial",
     "evidence": [{"source_id": "src_...", "quote": "从 text 里原样复制的连续一段", "locator": "正文"}]
   }
   ```
   - `quote` 必须是 `read_source` 返回的 `text` 里的**连续原文**（≥12 字符），不能改写、翻译、拼接、加省略号；
   - 返回 `evidence_mismatch` 时看 `metadata.details` 里是哪一条、什么问题，改完再交；
   - 媒体转述只能算 `publisher_claim` + `partial`，不能当独立验证；
   - 结论是“发生了什么”用 `observable_fact`，是“某人声称”用 `publisher_claim`，是论文实验结果用 `research_result`。

5. **`compare_sources(run_id, source_ids?)`** — 来源之间数字/版本号对照。如果两篇读起来像同一件事
   （转载、同一发布会），先看一眼 `shared_values`：共有只能说明“抄了同一个数”，不能当独立验证，
   这种情况下 `support_status` 用 `partial`，别升到 `supported`。

6. **`coverage_check(run_id)`** — 看看主题缺口与预算。有缺口且预算允许，就再搜一轮（最多 2 轮）；没有就直接出报告。

7. **`publish_report(run_id, summary, limitations=[...])`** — 出报告。返回 `report_id`、`status`（`published` / `partial`）、七项质量校验与 Markdown。
   - 状态 `partial` 不是失败，它表示“有未解决缺口或某项校验没过”，报告里会写明是哪些；
   - 把 `markdown` 原样交给用户，不要自己重写、不要删掉“覆盖情况”与“质量校验”。

## 硬规则

- 报告里的每条结论都必须能追到 `source_id` 与原文引文；没有证据的结论工具不收。
- 不要因为“没搜到”就说“没有发生”。没有来源就写进 `limitations`，让报告体现这个边界。
- 来源的 `published_at` 为空时，不要猜日期；窗口判断会因此变成未知，如实说明。
- 预算错误（`budget_exceeded`）不要绕过：用已有来源出版，并在 `limitations` 里写明还差什么。
- 用户改时间范围时，重新 `start_research`，不要复用旧 `run_id` 的结论。

## 一次典型调用（约 8 次工具）

```
start_research(question, days=7)
search_web(run_id, "AI Agent 评测 发布")        -> 5 条候选
search_web(run_id, "智能体 框架 开源")          -> 5 条候选
read_source(run_id, 候选1)                      -> 正文 + source_id
read_source(run_id, 候选2)                      -> 正文 + source_id
compare_sources(run_id)                          -> 两篇里哪些数字是同一处抄来的
submit_findings(run_id, [结论1, 结论2])           -> 一次交完，坏的按条退回
coverage_check(run_id)                          -> 缺口/预算
publish_report(run_id, summary, limitations)     -> markdown + 校验
```

## 中断之后怎么接

工具进程可能重启（MCP 每次调用都是新进程），所以**不要把状态记在脑子里**——它都在 run 里。

- 拿不准做到哪了：先 `run_status(run_id)`，看已读来源、已提交结论、剩余预算。
- 如果它说 `recoverable`（进程重启过）或你手上只有一个 `run_id`：先 `resume_research(run_id)`，拿到恢复好的状态再继续。
- 续跑不会重复抓已经读过的正文（返回里 `reused: true`），预算也接着上次扣。
- 已经在 `published`/`partial` 的 run 不能再续；同一问题要重跑，就重新 `start_research`（同一个问题 + 同一个 `as_of` 会复用同一个 run_id，所以要么换 `as_of`，要么先确认旧报告够用）。



## 两个 MCP 服务共同使用

同一个主 Agent 同时拥有 research-tools 与 RecallGraph。遇到论文原文、已有资料或历史记忆问题，
按需阅读 recallgraph Skill 并调用该服务。RecallGraph 不是子 Agent。
两个服务编号独立：chunk_id/document_id 不能直接当作本次调研的 source_id 提交。
可以综合两边资料回答，但 publish_report 的正式引用仍要求来源已在研究 run 中登记。
