"""Real stdio smoke check for research-tools + RecallGraph, without an LLM.

RecallGraph is queried read-only. The research run uses temporary replay storage.
No database initialization, note writes, memory commits or paid model calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

RESEARCH_NAMES = {
    'start_research', 'search_web', 'read_source', 'arxiv_search', 'github_releases',
    'submit_finding', 'submit_findings', 'compare_sources', 'coverage_check',
    'publish_report', 'resume_research', 'run_status',
}
RECALL_NAMES = {
    'search_materials', 'read_evidence', 'query_knowledge_graph', 'recall_memories',
    'propose_memory_update', 'create_or_update_note',
}


def payload(result) -> dict:
    result_data = getattr(result, 'structured_content', None)
    if result_data is None:
        result_data = getattr(result, 'structuredContent', None)
    if result_data is None:
        raise AssertionError('MCP 结果缺少结构化信封')
    if not result_data.get('ok'):
        error = result_data.get('error') or {}
        # Connection errors may embed a DSN; do not dump raw server messages.
        raise AssertionError(f'MCP 业务失败：{error.get("code", "unknown")}')
    return result_data['data']


async def check(query: str = 'retrieval augmented generation', *, dsn: str | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix='insightflow-dual-mcp-') as scratch:
        async with AsyncExitStack() as stack:
            sessions = {}
            counts = {}
            for name, module, expected in [
                ('research-tools', 'research_tools.server', RESEARCH_NAMES),
                ('recallgraph', 'recallgraph.mcp_server', RECALL_NAMES),
            ]:
                env = {
                    **os.environ,
                    'INSIGHTFLOW_DATA_DIR': scratch,
                    'TAVILY_API_KEY': '',
                    'GITHUB_TOKEN': '',
                    'DASHSCOPE_API_KEY': '',
                    'RECALLGRAPH_EMBEDDING_MODEL': 'text-embedding-v3',
                }
                if dsn:
                    env['RECALLGRAPH_DSN'] = dsn
                params = StdioServerParameters(command=sys.executable, args=['-m', module], env=env)
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                info = await session.initialize()
                assert info.server_info.name == name
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert names == expected, f'{name} 工具清单不一致'
                sessions[name] = session
                counts[name] = len(names)

            research = sessions['research-tools']
            started = payload(await research.call_tool('start_research', {
                'question': '最近一周 AI Agent 评测动态', 'mode': 'replay',
                'fixture_id': 'briefing-2026-10-27', 'as_of': '2026-10-27T12:00:00+08:00',
            }))
            run_id = started['run_id']
            hits = payload(await research.call_tool('search_web', {'run_id': run_id, 'query': 'AI Agent'}))
            assert hits
            source = payload(await research.call_tool('read_source', {'run_id': run_id, 'url': hits[0]['url']}))
            assert source['text']

            library = sessions['recallgraph']
            found = payload(await library.call_tool('search_materials', {'query': query, 'top_k': 3}))
            chunk_ids = list(dict.fromkeys(
                [claim['chunk_id'] for claim in found['claims']] +
                [chunk['chunk_id'] for chunk in found['chunks']]
            ))[:2]
            assert chunk_ids, '知识库没有命中；请换 --query 或先导入资料'
            evidence = payload(await library.call_tool('read_evidence', {'chunk_ids': chunk_ids, 'max_chars': 1000}))
            assert evidence['chunks'] and any(chunk['text'] for chunk in evidence['chunks'])
            document_id = evidence['chunks'][0]['document_id']
            assert isinstance(document_id, str)
            page = int(evidence['chunks'][0]['page'].split('-')[0])
            by_page = payload(await library.call_tool('read_evidence', {
                'document_id': document_id, 'page': page, 'page_end': page, 'max_chars': 1000,
            }))
            assert by_page['chunks']
            recalled = payload(await library.call_tool('recall_memories', {'query': 'answer_style'}))
            return {
                'ok': True, 'servers': counts, 'business_tools': sum(counts.values()),
                'research': {'sources_read': 1, 'mode': 'replay'},
                'recallgraph': {
                    'claims': len(found['claims']), 'chunks_read': len(evidence['chunks']),
                    'document_page_read': True, 'memory_items': len(recalled['memories']),
                },
                'writes': 'temporary research run only; RecallGraph queried read-only',
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='retrieval augmented generation')
    args = parser.parse_args()
    report = asyncio.run(check(args.query))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
