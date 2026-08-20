from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from observability import record_http_request, record_llm_usage, snapshot_metrics
from qdrant_retriever import qdrant_enabled


class ObservabilityTest(unittest.TestCase):
    def test_record_http_and_llm_usage(self) -> None:
        before = snapshot_metrics()
        record_http_request("/api/test", 12.5, True)
        record_llm_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 3.0)
        after = snapshot_metrics()
        self.assertEqual(after["http"]["requests"], before["http"]["requests"] + 1)
        self.assertGreaterEqual(after["llm"]["total_tokens"], before["llm"]["total_tokens"] + 15)

    def test_qdrant_disabled_without_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QDRANT_URL", None)
            os.environ.pop("QDRANT_COLLECTION", None)
            self.assertFalse(qdrant_enabled())


if __name__ == "__main__":
    unittest.main()
