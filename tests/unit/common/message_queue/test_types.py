from uuid import UUID

import pytest

from ai.backend.common.contexts.user import current_user, triggered_user
from ai.backend.common.data.user.types import UserData, UserRole
from ai.backend.common.json import dump_json, load_json
from ai.backend.common.message_queue.types import (
    BroadcastMessage,
    MessageMetadata,
    MQMessage,
    resolve_ack_stream_key,
)


def _make_user(user_id: str, is_superadmin: bool = False) -> UserData:
    return UserData(
        user_id=UUID(user_id),
        is_authorized=True,
        is_admin=is_superadmin,
        is_superadmin=is_superadmin,
        role=UserRole.SUPERADMIN if is_superadmin else UserRole.USER,
        domain_name="default",
    )


class TestMessageMetadata:
    def test_serialize_with_all_fields(self) -> None:
        user = UserData(
            user_id=UUID("12345678-1234-5678-1234-567812345678"),
            is_authorized=True,
            is_admin=False,
            is_superadmin=False,
            role=UserRole.USER,
            domain_name="default",
        )
        metadata = MessageMetadata(request_id="req-123", user=user)

        serialized = metadata.serialize()
        assert isinstance(serialized, bytes)

        # Verify the serialized data contains expected fields
        deserialized_dict = load_json(serialized)
        assert deserialized_dict["request_id"] == "req-123"
        assert "user" in deserialized_dict
        assert deserialized_dict["user"]["user_id"] == "12345678-1234-5678-1234-567812345678"
        assert deserialized_dict["user"]["is_authorized"] is True
        assert deserialized_dict["user"]["is_admin"] is False
        assert deserialized_dict["user"]["role"] == "user"
        assert deserialized_dict["user"]["domain_name"] == "default"

    def test_serialize_with_no_user(self) -> None:
        metadata = MessageMetadata(request_id="req-456", user=None)

        serialized = metadata.serialize()
        assert isinstance(serialized, bytes)

        deserialized_dict = load_json(serialized)
        assert deserialized_dict["request_id"] == "req-456"
        assert deserialized_dict["user"] is None

    def test_serialize_with_no_request_id(self) -> None:
        user = UserData(
            user_id=UUID("87654321-4321-8765-4321-876543218765"),
            is_authorized=True,
            is_admin=True,
            is_superadmin=False,
            role=UserRole.ADMIN,
            domain_name="test-domain",
        )
        metadata = MessageMetadata(request_id=None, user=user)

        serialized = metadata.serialize()
        assert isinstance(serialized, bytes)

        deserialized_dict = load_json(serialized)
        assert deserialized_dict["request_id"] is None
        assert "user" in deserialized_dict

    def test_serialize_with_no_fields(self) -> None:
        metadata = MessageMetadata()

        serialized = metadata.serialize()
        assert isinstance(serialized, bytes)

        deserialized_dict = load_json(serialized)
        assert deserialized_dict["request_id"] is None
        assert deserialized_dict["user"] is None

    def test_deserialize_with_all_fields(self) -> None:
        data = {
            "request_id": "req-789",
            "user": {
                "user_id": "11111111-2222-3333-4444-555555555555",
                "is_authorized": True,
                "is_admin": False,
                "is_superadmin": False,
                "role": "user",
                "domain_name": "org1",
            },
        }
        serialized = dump_json(data)

        metadata = MessageMetadata.deserialize(serialized)
        assert metadata.request_id == "req-789"
        assert isinstance(metadata.user, UserData)
        assert str(metadata.user.user_id) == "11111111-2222-3333-4444-555555555555"
        assert metadata.user.is_authorized is True
        assert metadata.user.is_admin is False
        assert metadata.user.is_superadmin is False
        assert metadata.user.role == UserRole.USER
        assert metadata.user.domain_name == "org1"

    def test_deserialize_from_string(self) -> None:
        data = {
            "request_id": "req-string",
            "user": {
                "user_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "is_authorized": False,
                "is_admin": True,
                "is_superadmin": True,
                "role": "superadmin",
                "domain_name": "system",
            },
        }
        serialized_str = dump_json(data).decode("utf-8")

        metadata = MessageMetadata.deserialize(serialized_str)
        assert metadata.request_id == "req-string"
        assert isinstance(metadata.user, UserData)
        assert str(metadata.user.user_id) == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert metadata.user.is_authorized is False
        assert metadata.user.is_admin is True
        assert metadata.user.is_superadmin is True

    def test_deserialize_with_legacy_user_id_field(self) -> None:
        # Test backward compatibility - remove user_id if present
        data = {
            "request_id": "req-legacy",
            "user_id": "should-be-removed",
            "user": {
                "user_id": "99999999-8888-7777-6666-555544443333",
                "is_authorized": True,
                "is_admin": False,
                "is_superadmin": False,
                "role": "user",
                "domain_name": "default",
            },
        }
        serialized = dump_json(data)

        metadata = MessageMetadata.deserialize(serialized)
        assert metadata.user is not None
        assert metadata.request_id == "req-legacy"
        assert hasattr(metadata, "user_id") is False  # user_id field should be removed
        assert str(metadata.user.user_id) == "99999999-8888-7777-6666-555544443333"

    def test_deserialize_with_invalid_user_data(self) -> None:
        # Test when user is not a dict
        data = {"request_id": "req-invalid", "user": "invalid-user-data"}
        serialized = dump_json(data)

        metadata = MessageMetadata.deserialize(serialized)
        assert metadata.request_id == "req-invalid"
        assert metadata.user is None

    def test_deserialize_with_no_user(self) -> None:
        data = {"request_id": "req-no-user"}
        serialized = dump_json(data)

        metadata = MessageMetadata.deserialize(serialized)
        assert metadata.request_id == "req-no-user"
        assert metadata.user is None

    def test_deserialize_empty_data(self) -> None:
        data: dict[str, str] = {}
        serialized = dump_json(data)

        metadata = MessageMetadata.deserialize(serialized)
        assert metadata.request_id is None
        assert metadata.user is None

    def test_serialize_deserialize_roundtrip(self) -> None:
        # Test complete roundtrip
        user = UserData(
            user_id=UUID("fedcba98-7654-3210-fedc-ba9876543210"),
            is_authorized=True,
            is_admin=False,
            is_superadmin=False,
            role=UserRole.USER,
            domain_name="enterprise",
        )
        original = MessageMetadata(request_id="roundtrip-test", user=user)

        serialized = original.serialize()
        deserialized = MessageMetadata.deserialize(serialized)
        assert deserialized.user is not None
        assert original.user is not None
        assert deserialized.request_id == original.request_id
        assert str(deserialized.user.user_id) == str(original.user.user_id)
        assert deserialized.user.is_authorized == original.user.is_authorized
        assert deserialized.user.is_admin == original.user.is_admin
        assert deserialized.user.is_superadmin == original.user.is_superadmin
        assert deserialized.user.role == original.user.role
        assert deserialized.user.domain_name == original.user.domain_name

    def test_roundtrip_preserves_triggered_user(self) -> None:
        target = _make_user("11111111-1111-1111-1111-111111111111")
        super_admin = _make_user("22222222-2222-2222-2222-222222222222", is_superadmin=True)
        original = MessageMetadata(user=target, triggered_user=super_admin)

        deserialized = MessageMetadata.deserialize(original.serialize())
        assert deserialized.triggered_user is not None
        assert str(deserialized.triggered_user.user_id) == str(super_admin.user_id)
        assert deserialized.triggered_user.is_superadmin is True

    def test_apply_context_restores_and_resets_both_users(self) -> None:
        target = _make_user("11111111-1111-1111-1111-111111111111")
        super_admin = _make_user("22222222-2222-2222-2222-222222222222", is_superadmin=True)
        metadata = MessageMetadata(user=target, triggered_user=super_admin)

        with metadata.apply_context():
            effective = current_user()
            trigger = triggered_user()
            assert effective is not None and trigger is not None
            assert str(effective.user_id) == str(target.user_id)
            assert str(trigger.user_id) == str(super_admin.user_id)
        # Reset after the block — also covers the system-event case (both None).
        assert current_user() is None
        assert triggered_user() is None


class TestMQMessageRetry:
    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_retry_increments_count_within_limit(self, count: int) -> None:
        msg = MQMessage(b"1-0", {b"_retry_count": str(count).encode()})
        assert msg.retry() is True
        assert msg.payload[b"_retry_count"] == str(count + 1).encode()

    def test_retry_without_count_starts_at_one(self) -> None:
        msg = MQMessage(b"1-0", {b"name": b"x"})
        assert msg.retry() is True
        assert msg.payload[b"_retry_count"] == b"1"

    def test_retry_refused_past_limit(self) -> None:
        msg = MQMessage(b"1-0", {b"_retry_count": b"4"})
        assert msg.retry() is False
        assert msg.payload[b"_retry_count"] == b"4"

    @pytest.mark.parametrize(
        "value",
        [b"x", b"-1", b"", b"\xff", b"1.5", b"9" * 5000, b"99999999999999999999"],
    )
    def test_malformed_count_discards_without_raising(self, value: bytes) -> None:
        msg = MQMessage(b"1-0", {b"_retry_count": value})
        assert msg.retry() is False
        assert msg.payload[b"_retry_count"] == value


class TestMessageOrigin:
    def test_stream_key_and_channel_default_to_none(self) -> None:
        assert MQMessage(b"1-0", {}).stream_key is None
        assert BroadcastMessage({}).channel is None

    def test_stream_key_and_channel_are_kept(self) -> None:
        assert MQMessage(b"1-0", {}, stream_key="events:a").stream_key == "events:a"
        assert BroadcastMessage({}, channel="events_all:a").channel == "events_all:a"


class TestResolveAckStreamKey:
    def test_given_stream_key_is_used(self) -> None:
        assert resolve_ack_stream_key("b", {"a", "b"}) == "b"

    def test_single_stream_is_used_without_stream_key(self) -> None:
        assert resolve_ack_stream_key(None, {"events"}) == "events"

    @pytest.mark.parametrize("consumed", [set(), {"a", "b"}])
    def test_missing_stream_key_is_refused_unless_one_stream(self, consumed: set[str]) -> None:
        with pytest.raises(ValueError):
            resolve_ack_stream_key(None, consumed)

    def test_stream_not_consumed_is_refused(self) -> None:
        with pytest.raises(ValueError):
            resolve_ack_stream_key("c", {"a", "b"})
