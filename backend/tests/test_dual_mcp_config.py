from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]


def _load_configurer():
    spec = importlib.util.spec_from_file_location('configure_dual_mcp', PROJECT / 'integrations/flow/configure_mcp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_keeps_servers_separate_and_preserves_existing_settings(tmp_path):
    project = tmp_path / 'InsightFlow'
    runtime = project / 'flow'
    runtime.mkdir(parents=True)
    interpreter = project / '.venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    template = project / 'integrations/flow/extensions_config.example.json'
    template.parent.mkdir(parents=True)
    template.write_text((PROJECT / 'integrations/flow/extensions_config.example.json').read_text())
    for name in ('research-briefing', 'recallgraph'):
        path = project / 'skills' / name / 'SKILL.md'
        path.parent.mkdir(parents=True)
        path.write_text(f'---\nname: {name}\ndescription: test\n---\n')
    (runtime / 'extensions_config.json').write_text(json.dumps({
        'mcpServers': {
            'research-tools': {'env': {'INSIGHTFLOW_DATA_DIR': '/existing/data', 'MODEL_BASE_URL': 'https://dashscope.aliyuncs.com/compatible-mode/v1'}},
            'unrelated': {'enabled': False, 'type': 'stdio', 'command': 'keep-me'},
        },
    }))
    configure = _load_configurer().configure
    configure(project, runtime)
    first = (runtime / 'extensions_config.json').read_text()
    configure(project, runtime)
    assert (runtime / 'extensions_config.json').read_text() == first
    config = json.loads(first)
    research = config['mcpServers']['research-tools']
    recall = config['mcpServers']['recallgraph']
    assert research['args'] == ['-m', 'research_tools.server']
    assert recall['args'] == ['-m', 'recallgraph.mcp_server']
    assert research['command'] == recall['command'] == str(interpreter)
    assert research['env']['INSIGHTFLOW_DATA_DIR'] == '/existing/data'
    assert recall['env']['DASHSCOPE_API_KEY'] == '$MODEL_API_KEY'
    assert config['mcpServers']['unrelated']['command'] == 'keep-me'
    assert '/absolute/path/' not in first
    for name in ('research-briefing', 'recallgraph'):
        assert (runtime / 'skills/custom' / name / 'SKILL.md').is_file()
        assert config['skills'][name]['enabled'] is True


def test_template_has_two_independent_services_and_no_nested_agent():
    config = json.loads((PROJECT / 'integrations/flow/extensions_config.example.json').read_text())
    assert set(config['mcpServers']) == {'research-tools', 'recallgraph'}
    assert config['mcpServers']['research-tools']['args'][-1] == 'research-tools'
    assert config['mcpServers']['recallgraph']['args'][-1] == 'recallgraph-mcp'


def test_agent_smoke_requires_actual_successful_data_not_only_tool_calls():
    spec = importlib.util.spec_from_file_location('dual_smoke', PROJECT / 'integrations/flow/dual_agent_smoke.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failed = SimpleNamespace(type='tool', name='recallgraph_read_evidence', artifact=None,
                             content=json.dumps({'ok': False, 'data': None}))
    empty = SimpleNamespace(type='tool', name='recallgraph_read_evidence', artifact=None,
                            content=[{'type': 'text', 'text': json.dumps({'ok': True, 'data': {'chunks': []}})}])
    valid = SimpleNamespace(type='tool', name='recallgraph_read_evidence', content='',
                            artifact={'structured_content': {'ok': True, 'data': {'chunks': [{'text': 'real evidence'}]}}})
    assert module.successful_tool_results([failed, empty]) == set()
    assert module.successful_tool_results([failed, empty, valid]) == {'recallgraph_read_evidence'}
