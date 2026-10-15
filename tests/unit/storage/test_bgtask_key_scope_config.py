from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.backend.common.exception import BackendAISchemaValidationFailed
from ai.backend.storage.config.unified import StorageProxyConfig
from ai.backend.storage.server import bgtask_ctx

_REQUIRED: dict[str, Any] = {"node-id": "node-1", "secret": "s", "session-expire": "1h"}


async def _created_with(config: StorageProxyConfig) -> dict[str, Any]:
    """The keyword arguments the background task context creates its Valkey client with."""
    local_config: Any = SimpleNamespace(storage_proxy=config)
    with (
        patch(
            "ai.backend.storage.server.ValkeyBgtaskClient.create", new=AsyncMock()
        ) as mock_create,
        patch("ai.backend.storage.server.BackgroundTaskManager") as mock_manager,
        patch("ai.backend.storage.server.BgtaskHandlerRegistryCreator"),
    ):
        mock_manager.return_value.init = AsyncMock()
        mock_manager.return_value.shutdown = AsyncMock()
        async with bgtask_ctx(local_config, MagicMock(), MagicMock(), MagicMock()):
            pass
    return dict(mock_create.call_args.kwargs)


class TestBgtaskKeyScopeConfig:
    def test_the_scope_is_unset_by_default(self) -> None:
        assert StorageProxyConfig.model_validate(_REQUIRED).bgtask_key_scope is None

    @pytest.mark.parametrize("key", ["bgtask-key-scope", "bgtask_key_scope"])
    def test_the_scope_is_read(self, key: str) -> None:
        config = StorageProxyConfig.model_validate({**_REQUIRED, key: "storage-proxy.node-1"})
        assert config.bgtask_key_scope == "storage-proxy.node-1"

    @pytest.mark.parametrize("key_scope", ["", "a:b", "a*"])
    def test_an_invalid_scope_is_refused(self, key_scope: str) -> None:
        with pytest.raises(BackendAISchemaValidationFailed):
            StorageProxyConfig.model_validate({**_REQUIRED, "bgtask-key-scope": key_scope})

    @pytest.mark.parametrize("key_scope", [None, "storage-proxy.node-1"])
    async def test_the_background_task_client_gets_the_scope(self, key_scope: str | None) -> None:
        overrides = {} if key_scope is None else {"bgtask-key-scope": key_scope}
        config = StorageProxyConfig.model_validate({**_REQUIRED, **overrides})
        assert (await _created_with(config))["key_scope"] == key_scope
