"""端到端评测的题集：20 道，五类能力，每道题都写明"什么算对"。

五类：
- factual：知识库里有明确答案，考检索 + 引用；
- cross_lingual：中文提问、英文论文，考跨语言检索；
- multihop：需要多跳或跨文档组合；
- memory：考记忆召回（含"记忆与证据冲突时要反驳"）；
- refuse：知识库里没有，考拒答（不该编）。

判分两层：
1. 规则层：must_contain / any_of / must_refuse / 引用格式；
2. 模型层：qwen-plus 当裁判，看事实是否正确、有没有编造。
两层都过才算通过——规则能抓住"没引用"，裁判能抓住"引用了但说错了"。
"""

from __future__ import annotations

CASES = [
    # ── factual ──────────────────────────────────────────────────────────
    {
        "id": "f1",
        "expected": "RAG 训练只更新查询编码器和生成器（BART），文档编码器与文档索引冻结；引用论文第 3 页。",
        "category": "factual",
        "question": "RAG 训练的时候更新哪些参数？哪些被冻结了？",
        "must_contain": ["生成器"],
        "any_of": [["冻结", "固定", "不更新"]],
        "need_citation": True,
    },
    {
        "id": "f2",
        "expected": "残差连接解决网络加深后的退化问题（训练误差反而升高），让深层网络更容易优化。",
        "category": "factual",
        "question": "ResNet 的残差连接是为了解决什么问题？",
        "must_contain": ["退化"],
        "any_of": [["深层", "加深"]],
        "need_citation": True,
    },
    {
        "id": "f3",
        "expected": "RAG 的检索器用 DPR（Dense Passage Retriever）初始化。",
        "category": "factual",
        "question": "RAG 的检索器是用什么初始化的？",
        "must_contain": ["DPR"],
        "need_citation": True,
    },
    {
        "id": "f4",
        "expected": "ResNet 中批归一化放在每个卷积之后、ReLU 激活之前。",
        "category": "factual",
        "question": "ResNet 里批归一化一般放在什么位置？",
        "must_contain": ["卷积"],
        "any_of": [["激活", "ReLU"]],
        "need_citation": True,
    },
    # ── cross_lingual ────────────────────────────────────────────────────
    {
        "id": "c1",
        "expected": "推理时检索器对问题编码，从文档索引里检索 top-k 文档交给生成器。",
        "category": "cross_lingual",
        "question": "推理阶段检索器具体做什么？",
        "must_contain": ["检索"],
        "any_of": [["文档", "top-k", "top k"]],
        "need_citation": True,
    },
    {
        "id": "c2",
        "expected": "论文把“网络加深、训练误差反而上升”称为退化问题（degradation）。",
        "category": "cross_lingual",
        "question": "网络加深以后训练误差反而上升，论文里管这个叫什么？",
        "must_contain": ["退化"],
        "need_citation": True,
    },
    {
        "id": "c3",
        "expected": "RAG 的生成器用的是 BART（预训练 seq2seq 模型）。",
        "category": "cross_lingual",
        "question": "RAG 的生成器用的是什么模型？",
        "must_contain": ["BART"],
        "need_citation": True,
    },
    {
        "id": "c4",
        "expected": "批归一化让每层输入分布稳定，从而加速收敛、允许更大学习率。",
        "category": "cross_lingual",
        "question": "batch normalization 在 ResNet 里起什么作用？",
        "any_of": [["稳定", "加速", "收敛", "归一"]],
        "need_citation": True,
    },
    # ── multihop ─────────────────────────────────────────────────────────
    {
        "id": "m1",
        "expected": "残差连接让梯度有短路可走、缓解退化；批归一化稳定每层输入分布；两者共同让很深的网络可训练。",
        "category": "multihop",
        "question": "残差连接和批归一化分别怎么帮助训练很深的网络？",
        "must_contain": ["残差"],
        "any_of": [["批归一化", "BN", "归一化"]],
        "need_citation": True,
    },
    {
        "id": "m2",
        "expected": "训练时文档编码器和索引冻结，只更新查询编码器与生成器；因为每次更新文档编码器都要重建索引，代价高且收益小。",
        "category": "multihop",
        "question": "RAG 的检索器和生成器是怎么联合训练的？为什么这样设计？",
        "must_contain": ["冻结"],
        "any_of": [["端到端", "联合", "索引"]],
        "need_citation": True,
    },
    {
        "id": "m3",
        "expected": "方案文档写的是 BM25 与向量两路召回（各 20 条）用 RRF 融合，再做 1–2 跳图扩展、重排与证据覆盖检查；文档原话把“BM25、向量召回和图扩展融合”并列，所以答“三路召回+RRF”也不算错。",
        "category": "multihop",
        "question": "方案文档里检索阶段打算用几路召回、怎么融合？",
        "any_of": [["BM25", "向量"], ["RRF", "重排"]],
        "need_citation": True,
    },
    {
        "id": "m4",
        "expected": "Claim 是“某篇文献在什么条件下断言了什么”的索引，Evidence 是指回原文的引文与页码；分开是为了让断言可追溯到原文、可被反驳和更新。",
        "category": "multihop",
        "question": "方案文档里 Claim 和 Evidence 分别是什么、为什么要分开？",
        "must_contain": ["断言"],
        "any_of": [["证据", "原文"]],
        "need_citation": True,
    },
    # ── memory ───────────────────────────────────────────────────────────
    {
        "id": "mem1",
        "expected": "用户之前说二开重点转向记忆评测（LongMemEval 五类能力），该说法目前仍有效。",
        "category": "memory",
        "question": "我之前说二开的重点转向什么了？现在还有效吗？",
        "must_contain": ["记忆评测"],
        "need_citation": False,
    },
    {
        "id": "mem2",
        "expected": "用户记忆里说文档编码器会一起更新，但论文证据是不会更新/冻结；正确做法是指出冲突并以论文第 3 页为准。",
        "category": "memory",
        "question": "RAG 训练时文档编码器会更新吗？我记得我之前说会一起更新。",
        "must_contain": ["不会"],
        "any_of": [["冻结", "固定"]],
        "need_citation": True,
        "note": "冲突题：记忆说会更新，证据说冻结。答对的标准是反驳用户并给出依据。",
    },
    {
        "id": "mem3",
        "expected": "用户偏好先给结论、再给理由。",
        "category": "memory",
        "question": "我偏好的回答风格是什么？",
        "any_of": [["先给结论", "结论在前", "先结论"]],
        "need_citation": False,
    },
    {
        "id": "mem4",
        "expected": "用户认为 RRF 的 k 值影响很大、调到 100 更好；这是用户自己的看法，知识库里没有论文依据。",
        "category": "memory",
        "question": "我之前对 RRF 的 k 值是怎么看的？",
        "any_of": [["100"], ["影响"]],
        "need_citation": False,
    },
    # ── refuse ───────────────────────────────────────────────────────────
    {
        "id": "r1",
        "expected": "知识库里没有 BERT 参数量的信息，应明确说没有，而不是给一个数字。",
        "category": "refuse",
        "question": "BERT 模型一共有多少参数？",
        "must_refuse": True,
    },
    {
        "id": "r2",
        "expected": "知识库里没有引用次数数据（这属于外部实时信息），应明确说明没有、不要给出数字。",
        "category": "refuse",
        "question": "我们入库的这几篇论文，目前在 Google Scholar 上各被引用了多少次？",
        "must_refuse": True,
    },
    {
        "id": "r3",
        "expected": "知识库里没有 2025 年 RAG 新论文，应明确说没有。",
        "category": "refuse",
        "question": "2025 年 RAG 领域最重要的三篇新论文是哪三篇？",
        "must_refuse": True,
    },
    {
        "id": "r4",
        "expected": "知识库里没有线上 QPS 数据，应明确说没有。",
        "category": "refuse",
        "question": "我们这套系统线上 QPS 峰值是多少？",
        "must_refuse": True,
    },
]


def seed_memories(conn) -> None:
    """把评测依赖的记忆补齐（幂等）：记忆题不该因为库是空的而失败。"""
    from recallgraph import memory

    seeds = {
        "project_focus": "二开重点转向记忆评测（LongMemEval 五类能力）",
        "rag_encoder_view": "RAG 训练时文档编码器也会一起更新",
        "answer_style": "先给结论，再给理由",
        "rrf_k_view": "RRF 的 k 值影响很大，调到 100 会明显更好",
    }
    for key, value in seeds.items():
        proposal = memory.propose(conn, key=key, value=value, reason="评测预置")
        if proposal["operation"] != "noop":
            memory.commit(conn, proposal)