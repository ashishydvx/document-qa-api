
import pytest

from app.chunking import split_text_into_chunks


def test_text_is_split_into_overlapping_chunks():
    text = "one two three four five six seven"

    chunks = split_text_into_chunks(
        text,
        chunk_size=4,
        overlap=1,
    )

    assert chunks == [
        "one two three four",
        "four five six seven",
    ]


def test_empty_text_returns_no_chunks():
    assert split_text_into_chunks("") == []


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        split_text_into_chunks("hello world", chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text_into_chunks(
            "hello world",
            chunk_size=5,
            overlap=5,
        )
