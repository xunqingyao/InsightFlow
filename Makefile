# InsightFlow 常用命令。所有 Python 命令都通过 uv 使用锁定依赖。
UV := UV_CACHE_DIR=$(CURDIR)/.uv-cache uv
FIXTURE := briefing-2026-10-27

.PHONY: sync test replay replay-supplement eval serve mcp clean

sync:            ## 安装/同步锁定依赖
	$(UV) sync --all-packages

test:            ## 单元 + 离线端到端测试
	cd backend && $(UV) run --all-packages pytest -q

replay:          ## 无凭据跑一次回放研究，报告写入 evaluation/results
	$(UV) run --all-packages python -m insightflow.cli run \
		--query "最近一周 AI Agent 评测动态" \
		--days 7 --replay $(FIXTURE) --offline --force \
		--out evaluation/results/replay-sample.md

replay-supplement:  ## 回放“发现缺口→补搜”的场景，报告写入 evaluation/results
	$(UV) run --all-packages python -m insightflow.cli run \
		--query "最近一周 AI Agent 评测动态" \
		--days 7 --replay supplement-2026-10-27 --offline --force \
		--out evaluation/results/replay-supplement.md

# 用真模型驱动一次工具链（需要 .env 里的 MODEL_*），默认走回放资料
live:
	$(UV) run --all-packages python evaluation/runners/live_model.py

eval:            ## 跑 evaluation/tasks 下的开发题，结果写 evaluation/results
	$(UV) run --all-packages python evaluation/runners/run_tasks.py

serve:           ## 启动 HTTP API（http://127.0.0.1:8000/docs）
	$(UV) run --all-packages uvicorn insightflow.api.app:app --reload --port 8000

mcp:             ## 以 stdio 启动本地 Research Tools MCP 服务
	$(UV) run --all-packages research-tools

clean:           ## 清理本地运行数据（不含源码）
	rm -rf data backend/data .pytest_cache backend/.pytest_cache

# 两个独立 MCP 服务，同一个宿主 Agent。
.PHONY: recallgraph-mcp recallgraph-init recallgraph-stats test-recallgraph configure-agent check-mcp agent-smoke

recallgraph-mcp:
	$(UV) run --all-packages recallgraph-mcp

recallgraph-init:
	$(UV) run --all-packages recallgraph init-db

recallgraph-stats:
	$(UV) run --all-packages recallgraph stats

test-recallgraph:
	cd mcp_servers/recallgraph && $(UV) run --all-packages pytest -q

configure-agent:
	$(UV) run --all-packages python integrations/flow/configure_mcp.py

check-mcp:
	$(UV) run --all-packages python integrations/flow/check_dual_mcp.py

agent-smoke:
	flow/backend/.venv/bin/python integrations/flow/dual_agent_smoke.py

.PHONY: check-secrets
check-secrets:
	python3 scripts/check_secrets.py --staged
