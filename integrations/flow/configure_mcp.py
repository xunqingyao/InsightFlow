"""Install two independent MCP servers and their Skills into the local DeerFlow host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def _resolve_paths(value, project: Path):
    if isinstance(value, str):
        return value.replace('/absolute/path/to/InsightFlow', str(project))
    if isinstance(value, list):
        return [_resolve_paths(item, project) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_paths(item, project) for key, item in value.items()}
    return value


def configure(project: Path = PROJECT, runtime: Path | None = None) -> dict:
    project = project.resolve()
    runtime = (runtime or project / 'flow').resolve()
    if not runtime.is_dir():
        raise FileNotFoundError(f'DeerFlow 目录不存在：{runtime}')
    interpreter = project / '.venv/bin/python'
    if not interpreter.is_file():
        raise FileNotFoundError('先在 InsightFlow 执行 make sync，安装两个服务到同一工作区虚拟环境。')

    template = _resolve_paths(
        json.loads((project / 'integrations/flow/extensions_config.example.json').read_text()), project)
    target = runtime / 'extensions_config.json'
    current = json.loads(target.read_text()) if target.is_file() else {}
    servers = current.setdefault('mcpServers', {})
    for name, module in [('research-tools', 'research_tools.server'), ('recallgraph', 'recallgraph.mcp_server')]:
        old = servers.get(name, {})
        entry = {**template['mcpServers'][name], **old}
        entry['env'] = {**template['mcpServers'][name].get('env', {}), **old.get('env', {})}
        entry.update(enabled=True, type='stdio', command=str(interpreter), args=['-m', module])
        if name == 'recallgraph' and not old:
            # This local host already uses DashScope via MODEL_API_KEY. Keep only a
            # variable reference in JSON; never copy the credential into config.
            research_env = servers.get('research-tools', {}).get('env', {})
            if 'dashscope.aliyuncs.com' in research_env.get('MODEL_BASE_URL', ''):
                entry['env']['DASHSCOPE_API_KEY'] = '$MODEL_API_KEY'
        servers[name] = entry

    skill_state = current.setdefault('skills', {})
    for name in ('research-briefing', 'recallgraph'):
        source = project / 'skills' / name / 'SKILL.md'
        dest = runtime / 'skills/custom' / name / 'SKILL.md'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
        skill_state[name] = {'enabled': True}

    target.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {
        'config': str(target),
        'servers': ['research-tools', 'recallgraph'],
        'skills': ['research-briefing', 'recallgraph'],
        'python': str(interpreter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path)
    args = parser.parse_args()
    print(json.dumps(configure(runtime=args.runtime), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
