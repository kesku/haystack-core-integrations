# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, logging
from haystack.utils import Secret
from haystack.utils.requests_utils import async_request_with_retry, request_with_retry

logger = logging.getLogger(__name__)

EXA_SEARCH_API_URL = "https://api.exa.ai/search"


@component
class ExaWebSearch:
    """
    A component that uses the Exa Search API to search the web and return results as Haystack Documents.

    You need an Exa API key from [dashboard.exa.ai](https://dashboard.exa.ai/api-keys).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.exa import ExaWebSearch
    from haystack.utils import Secret

    websearch = ExaWebSearch(
        api_key=Secret.from_env_var("EXA_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("EXA_API_KEY"),
        top_k: int | None = 10,
        search_type: str = "auto",
        category: str | None = None,
        contents: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the ExaWebSearch component.

        :param api_key:
            Exa API key. Defaults to the `EXA_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return. Maps to the `numResults` parameter in the Exa API.
        :param search_type:
            Search type: `"auto"` (default), `"neural"`, `"keyword"` or `"fast"`.
        :param category:
            Optional category to focus the search on (e.g. `"research paper"`, `"news"`, `"company"`).
        :param contents:
            Content retrieval options, passed as the `contents` object of the Exa API. Defaults to
            `{"highlights": True}`, which returns the most relevant passages of each page. Use
            `{"text": True}` to get the full page text instead.
        :param extra_params:
            Additional parameters passed directly to the Exa Search API, for example
            `{"includeDomains": ["arxiv.org"]}`.
        :param timeout:
            Timeout in seconds for the HTTP request. Defaults to 30.
        :param max_retries:
            Maximum number of retry attempts on transient failures. Defaults to 3.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.search_type = search_type
        self.category = category
        self.contents = contents
        self.extra_params = extra_params
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using Exa and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        payload = self._build_payload(query=query, top_k=top_k)
        headers = self._build_headers()

        response = request_with_retry(
            attempts=self.max_retries,
            method="POST",
            url=EXA_SEARCH_API_URL,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using Exa and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        payload = self._build_payload(query=query, top_k=top_k)
        headers = self._build_headers()

        response = await async_request_with_retry(
            attempts=self.max_retries,
            method="POST",
            url=EXA_SEARCH_API_URL,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _build_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.api_key.resolve_value() or "",
            "x-exa-integration": "deepset-ai/haystack-core-integrations-integration",
        }

    def _build_payload(self, query: str, top_k: int | None) -> dict[str, Any]:
        effective_top_k = top_k if top_k is not None else self.top_k
        payload: dict[str, Any] = {
            "query": query,
            "type": self.search_type,
            "contents": self.contents if self.contents is not None else {"highlights": True},
        }
        if effective_top_k is not None:
            payload["numResults"] = effective_top_k
        if self.category is not None:
            payload["category"] = self.category
        if self.extra_params:
            payload.update(self.extra_params)
        return payload

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an Exa Search API response to Haystack Documents and links.

        :param response: Exa Search API response dictionary.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        for result in response.get("results", []):
            url = result.get("url", "")
            highlights = result.get("highlights") or []
            content = " ".join(highlights) if highlights else (result.get("summary") or result.get("text") or "")

            documents.append(
                Document(
                    content=content,
                    meta={
                        "title": result.get("title", ""),
                        "url": url,
                        "author": result.get("author"),
                        "published_date": result.get("publishedDate"),
                        "score": result.get("score"),
                    },
                )
            )
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
