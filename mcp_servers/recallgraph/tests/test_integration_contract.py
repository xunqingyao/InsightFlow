from __future__ import annotations

import asyncio
from pathlib import Path

from recallgraph import mcp_server
from recallgraph.storage.db import SQL_DIR


def test_mcp_document_id_matches_text_database_key():
    listing = asyncio.run(mcp_server.server.list_tools())
    tool = next(item for item in listing if item.name == 'read_evidence')
    schema = tool.input_schema['properties']['document_id']
    assert any(branch.get('type') == 'string' for branch in schema.get('anyOf', [schema]))
    assert 'max_chars' in tool.input_schema['properties']


def test_mcp_read_evidence_forwards_string_id_and_character_budget(monkeypatch):
    observed = {}
    monkeypatch.setattr(mcp_server, '_db', lambda: object())

    def dispatch(conn, name, args):
        observed.update(name=name, args=args)
        return {'chunks': [], 'truncated': False}, {'tool': name}

    monkeypatch.setattr(mcp_server.tools, 'dispatch', dispatch)
    result = asyncio.run(mcp_server.server.call_tool('read_evidence', {
        'document_id': 'DOC-test-identifier', 'page': 2, 'page_end': 3, 'max_chars': 900,
    }))
    assert result.structured_content['ok'] is True
    assert observed['args']['document_id'] == 'DOC-test-identifier'
    assert observed['args']['max_chars'] == 900


def test_migrations_are_inside_distributable_package():
    package_dir = Path(mcp_server.__file__).resolve().parent
    assert SQL_DIR == package_dir / 'sql'
    assert (SQL_DIR / '001_init.sql').is_file()
    assert (SQL_DIR / '008_memory_vectors.sql').is_file()


def test_server_closes_database_connection_on_shutdown(monkeypatch):
    class Connection:
        closed = False

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(mcp_server, '_conn', connection)
    monkeypatch.setattr(mcp_server.server, 'run', lambda **kwargs: None)
    mcp_server.main()
    assert connection.closed
