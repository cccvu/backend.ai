from __future__ import annotations

import zlib
from base64 import b64encode

import pytest

from ai.backend.common.log.types import ContainerLogData, ContainerLogError, ContainerLogType


class TestBoundedContent:
    def test_whole_zlib_chunk(self) -> None:
        data = ContainerLogData.from_log(ContainerLogType.ZLIB, b"hello\n")
        assert data.get_bounded_content(1024) == (b"hello\n", False)

    def test_exact_size_is_not_truncated(self) -> None:
        data = ContainerLogData.from_log(ContainerLogType.ZLIB, b"x" * 100)
        assert data.get_bounded_content(100) == (b"x" * 100, False)

    def test_decompression_stops_at_the_limit(self) -> None:
        # About 10 KiB of input that expands to 10 MiB.
        data = ContainerLogData.from_log(ContainerLogType.ZLIB, b"\0" * (10 * 1024 * 1024))
        content, truncated = data.get_bounded_content(4096)
        assert content == b"\0" * 4096
        assert truncated

    def test_zero_limit_returns_nothing(self) -> None:
        data = ContainerLogData.from_log(ContainerLogType.ZLIB, b"hello")
        assert data.get_bounded_content(0) == (b"", True)

    def test_plaintext_is_cut_at_the_limit(self) -> None:
        data = ContainerLogData.from_log(ContainerLogType.PLAINTEXT, b"abcdef")
        assert data.get_bounded_content(4) == (b"abcd", True)
        assert data.get_bounded_content(6) == (b"abcdef", False)

    @pytest.mark.parametrize(
        "content",
        [
            b64encode(b"\0\0\0").decode(),  # not a zlib stream
            b64encode(zlib.compress(b"hello world")[:-6]).decode(),  # incomplete stream
            "not base64!",
        ],
    )
    def test_malformed_content_raises(self, content: str) -> None:
        data = ContainerLogData(compress_type=ContainerLogType.ZLIB, content=content)
        with pytest.raises(ContainerLogError):
            data.get_bounded_content(1024)


class TestDeserialize:
    def test_round_trip(self) -> None:
        data = ContainerLogData.from_log(ContainerLogType.ZLIB, b"hello")
        assert ContainerLogData.deserialize(data.serialize(), max_size=1024) == data

    def test_oversized_data_is_rejected_before_parsing(self) -> None:
        data = ContainerLogData.from_log(ContainerLogType.PLAINTEXT, b"x" * 1024).serialize()
        with pytest.raises(ContainerLogError):
            ContainerLogData.deserialize(data, max_size=len(data) - 1)
        assert ContainerLogData.deserialize(data, max_size=len(data)).get_content() == b"x" * 1024
