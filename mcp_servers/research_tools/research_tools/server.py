"""Research Tools MCP 服务（stdio）：给**宿主 agent** 调用的研究工具。

这里不编排研究流程——搜几条、读几篇、要不要补搜、什么时候收尾，由调用方
（DeerFlow 的 lead agent、Claude Desktop、任何 MCP 客户端）决定。服务端只保证：

- 来源必须真的抓取成功才登记，正文进文件快照，可回溯；
- 结论里的引文必须逐字出现在快照里，否则整条退回并说明原因；
- 报告只能引用本次运行登记过的证据；七项质量校验随报告一起返回；
- 预算（检索/抓取次数）由服务端守，超了明确回 `budget_exceeded`。

调用顺序（skill 里也这么写）：

    start_research -> search_web / arxiv_search / github_releases -> read_source
                   -> submit_finding -> coverage_check -> publish_report

会话按 run_id 放在进程内；来源、证据、报告落 SQLite，正文落快照，进程重启后
用 `run_status` 或任一工具重新挂上同一个 run_id 即可继续。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from insightflow.config.settings import get_settings
from insightflow.research import session as sessions
from insightflow.research.schemas import RunMode

server = MCPServer(
    name="research-tools",
    version="0.2.0",
    instructions=(
        "InsightFlow 的研究工具。先 start_research 固定 as_of 与时间窗口，"
        "再用 search_web / arxiv_search / github_releases 找来源、read_source 读正文，"
        "结论用 submit_finding 提交（引文必须逐字来自 read_source 的 text），"
        "最后 publish_report 出报告。所有工具返回 ok/data/error/metadata 信封。"
    ),
)

_SESSIONS: dict[str, sessions.Session] = {}
MAX_CACHED_SESSIONS = 16  # 只是缓存，状态都在库里；长驻进程不至于把正文全攒在内存


def _remember(run_id: str, session: sessions.Session) -> None:
    _SESSIONS[run_id] = session
    while len(_SESSIONS) > MAX_CACHED_SESSIONS:
        _SESSIONS.pop(next(iter(_SESSIONS)))


def _get(run_id: str) -> tuple[sessions.Session | None, dict[str, Any] | None]:
    cached = _SESSIONS.get(run_id)
    if cached is not None:
        return cached, None
    session, attached = sessions.attach(get_settings(), run_id)
    if session is None:
        return None, attached
    _remember(run_id, session)
    return session, None


@server.tool(
    name="start_research",
    description=(
        "开始一次研究：固定 as_of 与时间窗口（左闭右开），返回 run_id、窗口、必需主题与预算。"
        "后续工具都要带这个 run_id。mode=replay 时只读本地回放资料，用于离线演示。"
    ),
)
def start_research(
    question: str,
    as_of: str | None = None,
    days: int = 7,
    focus: list[str] | None = None,
    mode: Literal["online", "replay"] = "online",
    fixture_id: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    parsed_as_of = datetime.fromisoformat(as_of) if as_of else None
    session, started = sessions.start(
        settings,
        question,
        as_of=parsed_as_of,
        days=days,
        focus=focus,
        mode=RunMode.REPLAY if mode == "replay" else RunMode.ONLINE,
        fixture_id=fixture_id,
    )
    if session is None:
        return started
    _SESSIONS[session.run_id] = session
    return started


@server.tool(
    name="search_web",
    description=(
        "检索公开网页，返回候选的标题、URL、摘要与发布时间。摘要只能当线索，不能直接引用；"
        "要引用必须先 read_source 读正文。未配置搜索 Key 时返回 unauthorized。"
    ),
)
def search_web(run_id: str, query: str, max_results: int = 5) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.search(session, query, max_results=max_results)


@server.tool(
    name="read_source",
    description=(
        "读取一个 URL 的正文：抓取、存快照、登记来源，返回 source_id 与可引用的 text。"
        "在线模式走真实 HTTP（拒绝内网地址），replay 模式下读本地资料。"
    ),
)
def read_source(run_id: str, url: str) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.read(session, url)


@server.tool(
    name="arxiv_search",
    description="检索 arXiv 预印本元数据（标题、作者、摘要、发布时间）。不含全文，引用摘要前请 read_source 读 abs 页。",
)
def arxiv_search(run_id: str, query: str, max_results: int = 5) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.arxiv_tool(session, query, max_results=max_results)


@server.tool(
    name="github_releases",
    description="读取 GitHub 仓库 Releases。没有 Release 不代表没有新 tag，不会推断成“无更新”。",
)
def github_releases(run_id: str, repo: str, limit: int = 5) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.github_tool(session, repo, limit=limit)


@server.tool(
    name="submit_finding",
    description=(
        "提交一条结论。evidence 每项为 {source_id, quote, locator}，quote 必须从 read_source 返回的 "
        "text 里逐字复制一段连续文字（≥12 字符）；对不上就整条退回，并在 metadata.details 说明哪条错。"
        "claim_type 取 observable_fact / publisher_claim / research_result / external_test / system_inference；"
        "support_status 取 supported / partial / conflicted / insufficient。"
    ),
)
def submit_finding(
    run_id: str,
    text: str,
    evidence: list[dict[str, Any]],
    claim_type: str = "publisher_claim",
    support_status: str = "partial",
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.submit_finding(
        session,
        text,
        claim_type=claim_type,
        support_status=support_status,
        evidence=evidence,
        limitations=limitations,
    )


@server.tool(
    name="submit_findings",
    description=(
        "一次提交多条结论（每条结构同 submit_finding）。逐条独立校验：通过的收下，没通过的按条退回，"
        "调用方只需重交被退回的那几条。写多结论的报告时优先用这个，少来回几轮。"
    ),
)
def submit_findings(run_id: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.submit_findings(session, findings)


@server.tool(
    name="compare_sources",
    description=(
        "把已读来源里的数字与版本号摆在一起，看哪些是多个来源共有的、哪些只出现在一处，"
        "每项带上下文便于核对。这是文本层面的对照，不是事实核查：转载常直接抄同一个数字，"
        "共有不代表独立验证。source_ids 省略时取前 3 个已读来源。"
    ),
)
def compare_sources(run_id: str, source_ids: list[str] | None = None) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.compare_sources(session, source_ids)


@server.tool(
    name="coverage_check",
    description=(
        "检查覆盖缺口：用户点到的主题是否有来源、可读来源是否太少、还剩下多少检索预算。"
        "有缺口且预算允许时再补搜；没有就出版，缺口会写进报告的“覆盖情况”。"
    ),
)
def coverage_check(run_id: str) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.coverage(session)


@server.tool(
    name="publish_report",
    description=(
        "用已提交的结论出报告：返回 report_id、状态（published / partial）、七项质量校验与 Markdown 正文。"
        "没有结论时拒绝出版；有未解决缺口或校验未过时状态为 partial，并在报告里写明原因。"
    ),
)
def publish_report(
    run_id: str,
    summary: str,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.publish(session, summary, limitations=limitations)


@server.tool(
    name="resume_research",
    description=(
        "续跑一次中断的运行（进程重启后状态是 recoverable）：恢复来源、证据、草稿结论与剩余预算，"
        "然后接着调 search_web/read_source/submit_findings，最后 publish_report。"
        "已经抓到的正文不会重复抓取，预算从上次停下的地方继续扣。"
    ),
)
def resume_research(run_id: str) -> dict[str, Any]:
    session, payload = sessions.resume(get_settings(), run_id)
    if session is None:
        return payload
    _remember(run_id, session)
    return payload


@server.tool(
    name="run_status",
    description="查看一次运行的状态：已读来源、已提交结论数、剩余预算、是否可以出报告。进程重启后用它重新挂上 run_id。",
)
def run_status(run_id: str) -> dict[str, Any]:
    session, error = _get(run_id)
    if session is None:
        return error or sessions.fail("not_found", run_id)
    return sessions.status(session)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()