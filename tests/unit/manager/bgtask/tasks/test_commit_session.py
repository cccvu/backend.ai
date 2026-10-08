from __future__ import annotations

import uuid
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.backend.common.data.session.types import CustomizedImageVisibilityScope
from ai.backend.common.docker import ImageRef
from ai.backend.common.events.event_types.bgtask.broadcast import (
    BgtaskDoneEvent,
    BgtaskFailedEvent,
    BgtaskUpdatedEvent,
)
from ai.backend.common.exception import BgtaskFailedError
from ai.backend.common.types import AgentId, SessionId
from ai.backend.manager.bgtask.tasks.commit_session import (
    CommitSessionHandler,
    CommitSessionManifest,
    CommitSessionResult,
)
from ai.backend.manager.bgtask.types import ManagerBgtaskName
from ai.backend.manager.errors.kernel import SessionNotFound


@pytest.fixture
def sample_manifest() -> CommitSessionManifest:
    return CommitSessionManifest(
        session_id=SessionId(uuid.uuid4()),
        registry_hostname="registry.example.com",
        registry_project="test-project",
        image_name="test-image",
        image_visibility=CustomizedImageVisibilityScope.USER,
        image_owner_id="user-123",
        user_email="test@example.com",
    )


class TestCommitSessionHandler:
    """Tests for CommitSessionHandler and CommitSessionManifest."""

    @pytest.fixture
    def sample_session_id(self) -> SessionId:
        """Sample session ID for testing."""
        return SessionId(uuid.uuid4())

    def test_handler_name(self) -> None:
        """Test handler returns correct name."""
        assert CommitSessionHandler.name() == ManagerBgtaskName.COMMIT_SESSION

    def test_handler_manifest_type(self) -> None:
        """Test handler returns correct manifest type."""
        assert CommitSessionHandler.manifest_type() == CommitSessionManifest

    def test_manifest_creation(self, sample_session_id: SessionId) -> None:
        """Test manifest can be created with required fields."""
        manifest = CommitSessionManifest(
            session_id=sample_session_id,
            registry_hostname="registry.example.com",
            registry_project="test-project",
            image_name="test-image",
            image_visibility=CustomizedImageVisibilityScope.USER,
            image_owner_id="user-123",
            user_email="test@example.com",
        )
        assert manifest.session_id == sample_session_id
        assert manifest.registry_hostname == "registry.example.com"
        assert manifest.registry_project == "test-project"
        assert manifest.image_name == "test-image"
        assert manifest.image_visibility == CustomizedImageVisibilityScope.USER
        assert manifest.image_owner_id == "user-123"
        assert manifest.user_email == "test@example.com"

    def test_manifest_serialization(self, sample_manifest: CommitSessionManifest) -> None:
        """Test manifest can be serialized and deserialized."""
        # Serialize to dict
        data = sample_manifest.model_dump(mode="json")
        assert data["session_id"] == str(sample_manifest.session_id)
        assert data["registry_hostname"] == sample_manifest.registry_hostname
        assert data["registry_project"] == sample_manifest.registry_project
        assert data["image_name"] == sample_manifest.image_name

        # Deserialize back
        restored = CommitSessionManifest.model_validate(data)
        assert restored.session_id == sample_manifest.session_id
        assert restored.registry_hostname == sample_manifest.registry_hostname
        assert restored.image_visibility == sample_manifest.image_visibility

    def test_result_default_values(self) -> None:
        """Test result default values are None."""
        result = CommitSessionResult()
        assert result.image_id is None

    def test_result_success_serialization(self) -> None:
        """Test result can be serialized and deserialized with success case."""
        image_id = uuid.uuid4()
        result = CommitSessionResult(image_id=image_id)

        # Serialize to dict
        data = result.model_dump(mode="json")
        assert data["image_id"] == str(image_id)

        # Deserialize back
        restored = CommitSessionResult.model_validate(data)
        assert restored.image_id == image_id


class TestCommitSessionExecute:
    """Regression tests for CommitSessionHandler.execute() (PR #12168)."""

    def _make_handler(self, session_repository: AsyncMock) -> CommitSessionHandler:
        return CommitSessionHandler(
            session_repository=session_repository,
            image_repository=AsyncMock(),
            agent_registry=AsyncMock(),
            event_fetcher=MagicMock(),
        )

    async def test_failure_raises_instead_of_returning_result(
        self, sample_manifest: CommitSessionManifest
    ) -> None:
        # Failures must propagate as exceptions (-> bgtask_failed) instead of
        # being swallowed into a success-typed CommitSessionResult.
        session_repository = AsyncMock()
        session_repository.get_session_by_id.return_value = None
        handler = self._make_handler(session_repository)

        with pytest.raises(SessionNotFound):
            await handler.execute(sample_manifest)

    async def test_base_image_resolved_including_deleted(
        self, sample_manifest: CommitSessionManifest
    ) -> None:
        # A running session's base image may have been deleted, so it must be
        # resolved with alive_only=False.
        session = MagicMock()
        session.main_kernel.image = "registry.example.com/base:latest"
        session.main_kernel.architecture = "x86_64"

        session_repository = AsyncMock()
        session_repository.get_session_by_id.return_value = session
        session_repository.get_container_registry.return_value = MagicMock()
        # Stop right after resolve_image to keep the test focused.
        session_repository.resolve_image.side_effect = RuntimeError("stop here")
        handler = self._make_handler(session_repository)

        with pytest.raises(RuntimeError):
            await handler.execute(sample_manifest)

        _, kwargs = session_repository.resolve_image.call_args
        assert kwargs["alive_only"] is False


class TestWaitForAgentBgtask:
    """The commit waiter reads only the cache scope of the session's agent."""

    def _make_handler(
        self,
        event_fetcher: MagicMock,
        *,
        session_repository: AsyncMock | None = None,
        agent_registry: AsyncMock | None = None,
        timeout: float = 5.0,
    ) -> CommitSessionHandler:
        return CommitSessionHandler(
            session_repository=session_repository or AsyncMock(),
            image_repository=AsyncMock(),
            agent_registry=agent_registry or AsyncMock(),
            event_fetcher=event_fetcher,
            agent_bgtask_timeout=timeout,
            agent_bgtask_poll_interval=0.01,
        )

    def _fetcher(self, *events: Any) -> MagicMock:
        event_fetcher = MagicMock()
        event_fetcher.fetch_cached_event = AsyncMock(side_effect=[*events])
        return event_fetcher

    async def test_reads_the_agent_scope_until_done(self) -> None:
        task_id = uuid.uuid4()
        other_task_id = uuid.uuid4()
        event_fetcher = self._fetcher(
            None,
            BgtaskUpdatedEvent(task_id=task_id, current_progress=0, total_progress=0),
            # An event for another task is ignored.
            BgtaskFailedEvent(task_id=other_task_id, message="other"),
            BgtaskDoneEvent(task_id=task_id, message="ok"),
        )
        handler = self._make_handler(event_fetcher)

        await handler._wait_for_agent_bgtask(task_id, "Commit", AgentId("agent-1"))

        cache_ids = {call.args[0] for call in event_fetcher.fetch_cached_event.await_args_list}
        assert cache_ids == {f"bgtask.agent.agent-1.{task_id}"}
        assert event_fetcher.fetch_cached_event.await_count == 4

    async def test_accepts_a_task_id_string(self) -> None:
        # The agent RPC returns the task ID as a string.
        task_id = uuid.uuid4()
        event_fetcher = self._fetcher(BgtaskDoneEvent(task_id=task_id, message="ok"))
        handler = self._make_handler(event_fetcher)

        await handler._wait_for_agent_bgtask(
            cast(uuid.UUID, str(task_id)), "Commit", AgentId("agent-1")
        )

    async def test_failure_is_raised(self) -> None:
        task_id = uuid.uuid4()
        event_fetcher = self._fetcher(BgtaskFailedEvent(task_id=task_id, message="boom"))
        handler = self._make_handler(event_fetcher)

        with pytest.raises(BgtaskFailedError):
            await handler._wait_for_agent_bgtask(task_id, "Commit", AgentId("agent-1"))

    async def test_times_out(self) -> None:
        task_id = uuid.uuid4()
        event_fetcher = MagicMock()
        event_fetcher.fetch_cached_event = AsyncMock(
            return_value=BgtaskUpdatedEvent(task_id=task_id, current_progress=0, total_progress=0)
        )
        handler = self._make_handler(event_fetcher, timeout=0.1)

        with pytest.raises(BgtaskFailedError):
            await handler._wait_for_agent_bgtask(task_id, "Push", AgentId("agent-1"))

    async def test_fetch_errors_do_not_end_the_wait(self) -> None:
        task_id = uuid.uuid4()
        event_fetcher = self._fetcher(
            RuntimeError("connection lost"), BgtaskDoneEvent(task_id=task_id, message="ok")
        )
        handler = self._make_handler(event_fetcher)

        await handler._wait_for_agent_bgtask(task_id, "Commit", AgentId("agent-1"))

    def test_rejects_non_positive_bounds(self) -> None:
        with pytest.raises(ValueError):
            CommitSessionHandler(
                session_repository=AsyncMock(),
                image_repository=AsyncMock(),
                agent_registry=AsyncMock(),
                event_fetcher=MagicMock(),
                agent_bgtask_timeout=0,
            )

    async def test_execute_waits_on_the_main_kernel_agent(
        self, sample_manifest: CommitSessionManifest
    ) -> None:
        task_id = uuid.uuid4()
        session = MagicMock()
        session.main_kernel.agent = "agent-1"
        session.main_kernel.image = "registry.example.com/test-project/base:latest"
        session.main_kernel.architecture = "x86_64"
        image_row = MagicMock()
        image_row.image_ref = ImageRef.from_image_str(
            "registry.example.com/test-project/base:latest",
            None,
            "registry.example.com",
            architecture="x86_64",
            is_local=True,
        )
        session_repository = AsyncMock()
        session_repository.get_session_by_id.return_value = session
        session_repository.get_container_registry.return_value = MagicMock()
        session_repository.resolve_image.return_value = image_row
        session_repository.get_existing_customized_image.return_value = None
        agent_registry = AsyncMock()
        agent_registry.commit_session.return_value = {"bgtask_id": str(task_id)}
        event_fetcher = self._fetcher(BgtaskFailedEvent(task_id=task_id, message="boom"))
        handler = self._make_handler(
            event_fetcher,
            session_repository=session_repository,
            agent_registry=agent_registry,
        )

        with pytest.raises(BgtaskFailedError):
            await handler.execute(sample_manifest)

        event_fetcher.fetch_cached_event.assert_awaited_once_with(f"bgtask.agent.agent-1.{task_id}")

    async def test_execute_requires_an_agent(self, sample_manifest: CommitSessionManifest) -> None:
        session = MagicMock()
        session.main_kernel.agent = None
        session_repository = AsyncMock()
        session_repository.get_session_by_id.return_value = session
        session_repository.get_container_registry.return_value = MagicMock()
        agent_registry = AsyncMock()
        handler = self._make_handler(
            MagicMock(), session_repository=session_repository, agent_registry=agent_registry
        )

        with pytest.raises(BgtaskFailedError):
            await handler.execute(sample_manifest)
        agent_registry.commit_session.assert_not_called()
