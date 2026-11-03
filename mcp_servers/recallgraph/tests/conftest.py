import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

os.environ["RECALLGRAPH_DSN"] = os.environ.get(
    "RECALLGRAPH_TEST_DSN",
    "postgresql://recallgraph:recallgraph@127.0.0.1:5434/insightflow_recallgraph_test",
)
