"""Use the actual single lead_agent with both independent MCP servers.

Run with the DeerFlow runtime's Python after `make configure-agent`.
Research uses temporary replay storage; RecallGraph is queried read-only.
This invokes the configured real LLM; it is an integration check, not an eval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RUNTIME = PROJECT / 'flow'
sys.path.insert(0, str(PROJECT / 'backend'))
sys.path.insert(0, str(RUNTIME / 'backend/packages/harness'))

QUESTION = '''这是一次双 MCP 接入检查，请使用同一个主 Agent 完成：
1. 阅读 research-briefing 和 recallgraph 的 Skill。
2. 使用研究工具创建调研，question 为“最近一周 AI Agent 评测动态”，mode=replay，
fixture_id=briefing-2026-10-27，as_of=2026-10-27T12:00:00+08:00，days=7。
调用 search_web 查“AI Agent”，选一个候选调用 read_source。
3. 使用 RecallGraph 的 search_materials 查“retrieval augmented generation”，
取一个实际返回的 chunk_id，调用 read_evidence 核对原文。
4. 用中文简短说明两边各读到了什么。论文资料给出标题与页码。
本次不需要提交结论或发布报告，不写笔记或记忆，不调用 arxiv_search/github_releases。
如果某服务失败，请如实说明，不能伪造工具结果。不要创建子 Agent。'''


def tool_payload(message) -> dict | None:
    artifact = getattr(message, 'artifact', None)
    if artifact is not None:
        structured = artifact.get('structured_content') if isinstance(artifact, dict) else getattr(artifact, 'structured_content', None)
        if isinstance(structured, dict):
            return structured
    content = getattr(message, 'content', '')
    if isinstance(content, list):
        content = '\n'.join(block.get('text', '') for block in content if isinstance(block, dict))
    try:
        result = json.loads(content)
    except (ValueError, TypeError):
        return None
    return result if isinstance(result, dict) else None


def successful_tool_results(messages: list) -> set[str]:
    successful = set()
    for message in messages:
        if getattr(message, 'type', '') != 'tool':
            continue
        item = tool_payload(message)
        if not item or item.get('ok') is not True:
            continue
        name = getattr(message, 'name', '')
        data = item.get('data')
        if name == 'research-tools_start_research' and isinstance(data, dict) and data.get('run_id'):
            successful.add(name)
        elif name == 'research-tools_search_web' and isinstance(data, list) and data:
            successful.add(name)
        elif name == 'research-tools_read_source' and isinstance(data, dict) and data.get('text'):
            successful.add(name)
        elif name == 'recallgraph_search_materials' and isinstance(data, dict) and (data.get('claims') or data.get('chunks')):
            successful.add(name)
        elif name == 'recallgraph_read_evidence' and isinstance(data, dict) and any(chunk.get('text') for chunk in data.get('chunks', [])):
            successful.add(name)
    return successful


async def run(*, output: Path | None = None, timeout: float = 240) -> dict:
    from insightflow.config.settings import Settings

    settings = Settings()
    if not settings.model_configured:
        raise RuntimeError('未配置宿主模型；先配置 InsightFlow 的 MODEL_* 环境变量。')
    for name, value in {
        'MODEL_API_KEY': settings.model_api_key,
        'MODEL_NAME': settings.model_name,
        'MODEL_BASE_URL': settings.model_base_url,
    }.items():
        if value and not os.environ.get(name):
            os.environ[name] = value
    os.environ['DEER_FLOW_CONFIG_PATH'] = str(RUNTIME / 'config.yaml')
    os.environ.setdefault('FASTEMBED_CACHE_PATH', str(PROJECT / '.cache/fastembed'))

    with tempfile.TemporaryDirectory(prefix='insightflow-agent-smoke-') as scratch:
        config = json.loads((RUNTIME / 'extensions_config.json').read_text())
        config['mcpServers']['research-tools']['env']['INSIGHTFLOW_DATA_DIR'] = scratch
        if os.environ.get('RECALLGRAPH_DSN'):
            config['mcpServers']['recallgraph']['env']['RECALLGRAPH_DSN'] = os.environ['RECALLGRAPH_DSN']
        temp_config = Path(scratch) / 'extensions_config.json'
        temp_config.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
        os.environ['DEER_FLOW_EXTENSIONS_CONFIG_PATH'] = str(temp_config)

        from flow.agents.lead_agent.agent import make_lead_agent
        from flow.config.memory_config import get_memory_config
        from flow.config.title_config import get_title_config
        from flow.tools import get_available_tools

        # Only this smoke process: avoid an unrelated title request or memory write.
        get_memory_config().enabled = False
        get_title_config().enabled = False
        cfg = {'configurable': {'subagent_enabled': False}, 'recursion_limit': 80}
        agent = make_lead_agent(cfg)
        names = {tool.name for tool in get_available_tools(subagent_enabled=False)}
        research_names = sorted(name for name in names if name.startswith('research-tools_'))
        recall_names = sorted(name for name in names if name.startswith('recallgraph_'))
        assert len(research_names) == 12 and len(recall_names) == 6

        thread_id = f'dual-mcp-{uuid.uuid4().hex}'
        print('工具发现成功：research-tools=12，recallgraph=6；开始真实主 Agent 调用。', flush=True)
        state = await asyncio.wait_for(agent.ainvoke(
            {'messages': [{'role': 'user', 'content': QUESTION}]},
            config=cfg, context={'thread_id': thread_id},
        ), timeout=timeout)
        calls = [
            call['name'] for message in state['messages']
            for call in (getattr(message, 'tool_calls', None) or [])
        ]
        required = {
            'research-tools_start_research', 'research-tools_search_web', 'research-tools_read_source',
            'recallgraph_search_materials', 'recallgraph_read_evidence',
        }
        missing = sorted(required - set(calls))
        missing_success = sorted(required - successful_tool_results(state['messages']))
        final = next((message for message in reversed(state['messages'])
                      if getattr(message, 'type', '') == 'ai' and not getattr(message, 'tool_calls', None)), None)
        report = {
            'ok': not missing and not missing_success and final is not None,
            'model': settings.model_name,
            'servers': {'research-tools': len(research_names), 'recallgraph': len(recall_names)},
            'tool_calls': calls, 'missing_required_tools': missing,
            'missing_successful_results': missing_success,
            'answer': str(final.content) if final is not None else '',
            'scope': 'single lead_agent; temporary research replay; read-only RecallGraph',
        }
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'data/integration/dual-agent-smoke.json')
    parser.add_argument('--timeout', type=float, default=240)
    args = parser.parse_args()
    report = asyncio.run(run(output=args.output, timeout=args.timeout))
    print(json.dumps({key: value for key, value in report.items() if key != 'answer'}, ensure_ascii=False, indent=2))
    print(f'完整回答与轨迹：{args.output}')
    if not report['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
