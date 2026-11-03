"""评测规则层（不调模型，纯字符串判断）"""

from evals.agent_eval import rule_check


def test_rule_pass_when_everything_present():
    case = {"must_contain": ["冻结"], "any_of": [["BART", "生成器"]], "need_citation": True}
    ok, reasons = rule_check(case, "文档编码器被冻结，只有 BART 生成器训练（见第 3 页）。")
    assert ok, reasons


def test_rule_fails_without_citation():
    case = {"must_contain": ["冻结"], "need_citation": True}
    ok, reasons = rule_check(case, "文档编码器被冻结。")
    assert not ok
    assert any("引用" in reason for reason in reasons)


def test_rule_requires_refusal_language():
    case = {"must_refuse": True}
    assert not rule_check(case, "BERT-base 有 1.1 亿参数。")[0]
    assert rule_check(case, "知识库里没有这个信息，无法回答。")[0]
