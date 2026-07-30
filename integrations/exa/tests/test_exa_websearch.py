# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.exa import ExaWebSearch

MOCK_RESPONSE = {
    "requestId": "req-1",
    "results": [
        {
            "title": "Example Title",
            "url": "https://example.com",
            "author": "Jane Doe",
            "publishedDate": "2026-01-01",
            "score": 0.42,
            "highlights": ["First highlight", "second highlight"],
        }
    ],
}


class TestExaWebSearch:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        ws = ExaWebSearch()
        assert ws.top_k == 10
        assert ws.search_type == "auto"
        assert ws.category is None
        assert ws.contents is None
        assert ws.extra_params is None
        assert ws.timeout == 30
        assert ws.max_retries == 3
        assert ws.api_key.resolve_value() == "test-key"

    def test_init_with_params(self):
        ws = ExaWebSearch(
            api_key=Secret.from_token("custom-key"),
            top_k=5,
            search_type="neural",
            category="research paper",
            contents={"text": True},
            extra_params={"includeDomains": ["arxiv.org"]},
            timeout=20,
            max_retries=5,
        )
        assert ws.top_k == 5
        assert ws.search_type == "neural"
        assert ws.category == "research paper"
        assert ws.contents == {"text": True}
        assert ws.extra_params == {"includeDomains": ["arxiv.org"]}
        assert ws.timeout == 20
        assert ws.max_retries == 5

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        ws = ExaWebSearch(top_k=5, category="news")
        data = component_to_dict(ws, "ExaWebSearch")
        assert data["type"] == "haystack_integrations.components.websearch.exa.exa_websearch.ExaWebSearch"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["category"] == "news"
        assert data["init_parameters"]["timeout"] == 30
        assert data["init_parameters"]["max_retries"] == 3

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.websearch.exa.exa_websearch.ExaWebSearch",
            "init_parameters": {
                "top_k": 3,
                "search_type": "keyword",
                "category": "news",
                "contents": {"text": True},
                "extra_params": None,
                "timeout": 15,
                "max_retries": 2,
                "api_key": {"env_vars": ["EXA_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        ws = component_from_dict(ExaWebSearch, data, "ExaWebSearch")
        assert ws.top_k == 3
        assert ws.search_type == "keyword"
        assert ws.category == "news"
        assert ws.contents == {"text": True}
        assert ws.timeout == 15
        assert ws.max_retries == 2

    def test_run_returns_documents_and_links(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        ws = ExaWebSearch(api_key=Secret.from_token("test-key"), top_k=5)

        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_RESPONSE

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.request_with_retry",
            return_value=mock_response,
        ):
            result = ws.run(query="test query")

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "First highlight second highlight"
        assert result["documents"][0].meta["url"] == "https://example.com"
        assert result["documents"][0].meta["title"] == "Example Title"
        assert result["documents"][0].meta["author"] == "Jane Doe"
        assert result["documents"][0].meta["published_date"] == "2026-01-01"
        assert result["documents"][0].meta["score"] == 0.42
        assert result["links"] == ["https://example.com"]

    def test_run_falls_back_to_summary_and_text(self):
        ws = ExaWebSearch(api_key=Secret.from_token("test-key"))

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com/1", "summary": "A summary"},
                {"url": "https://example.com/2", "text": "Page text"},
                {"url": "https://example.com/3"},
            ]
        }

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.request_with_retry",
            return_value=mock_response,
        ):
            result = ws.run(query="test query")

        assert [doc.content for doc in result["documents"]] == ["A summary", "Page text", ""]

    def test_run_passes_correct_payload(self):
        ws = ExaWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=5,
            search_type="neural",
            category="news",
            extra_params={"includeDomains": ["arxiv.org"]},
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.request_with_retry",
            return_value=mock_response,
        ) as mock_req:
            ws.run(query="test query")

        payload = mock_req.call_args.kwargs["json"]
        assert payload["query"] == "test query"
        assert payload["numResults"] == 5
        assert payload["type"] == "neural"
        assert payload["category"] == "news"
        assert payload["includeDomains"] == ["arxiv.org"]
        # content options are nested under `contents`, never at the top level
        assert payload["contents"] == {"highlights": True}
        assert "highlights" not in payload
        assert mock_req.call_args.kwargs["headers"]["x-api-key"] == "test-key"
        assert (
            mock_req.call_args.kwargs["headers"]["x-exa-integration"]
            == "deepset-ai/haystack-core-integrations-integration"
        )

    def test_run_custom_contents(self):
        ws = ExaWebSearch(api_key=Secret.from_token("test-key"), contents={"text": {"maxCharacters": 1000}})

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.request_with_retry",
            return_value=mock_response,
        ) as mock_req:
            ws.run(query="test query")

        assert mock_req.call_args.kwargs["json"]["contents"] == {"text": {"maxCharacters": 1000}}

    def test_run_top_k_override(self):
        ws = ExaWebSearch(api_key=Secret.from_token("test-key"), top_k=10)

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.request_with_retry",
            return_value=mock_response,
        ) as mock_req:
            ws.run(query="test", top_k=3)

        assert mock_req.call_args.kwargs["json"]["numResults"] == 3

    @pytest.mark.asyncio
    async def test_run_async_returns_documents_and_links(self):
        ws = ExaWebSearch(api_key=Secret.from_token("test-key"), top_k=5)

        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_RESPONSE

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.async_request_with_retry",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await ws.run_async(query="test query")

        assert len(result["documents"]) == 1
        assert result["links"] == ["https://example.com"]

    def test_run_empty_results(self):
        ws = ExaWebSearch(api_key=Secret.from_token("test-key"))

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.request_with_retry",
            return_value=mock_response,
        ):
            result = ws.run(query="very obscure query")

        assert result["documents"] == []
        assert result["links"] == []

    def test_run_missing_results_key(self):
        ws = ExaWebSearch(api_key=Secret.from_token("test-key"))

        mock_response = MagicMock()
        mock_response.json.return_value = {}

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.request_with_retry",
            return_value=mock_response,
        ):
            result = ws.run(query="test")

        assert result["documents"] == []
        assert result["links"] == []

    def test_run_raises_on_http_error(self):
        ws = ExaWebSearch(api_key=Secret.from_token("test-key"))

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=MagicMock()
        )

        with patch(
            "haystack_integrations.components.websearch.exa.exa_websearch.request_with_retry",
            return_value=mock_response,
        ):
            with pytest.raises(httpx.HTTPStatusError):
                ws.run(query="test")

    @pytest.mark.skipif(
        not os.environ.get("EXA_API_KEY"),
        reason="Export EXA_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        ws = ExaWebSearch(api_key=Secret.from_env_var("EXA_API_KEY"), top_k=3)
        result = ws.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)

    @pytest.mark.skipif(
        not os.environ.get("EXA_API_KEY"),
        reason="Export EXA_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        ws = ExaWebSearch(api_key=Secret.from_env_var("EXA_API_KEY"), top_k=3)
        result = await ws.run_async(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
