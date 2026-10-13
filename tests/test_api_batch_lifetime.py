"""Batch resources belong to the worker after submission, even on cancellation."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from breeze_infer import api


@pytest.mark.parametrize("worker_fails", [False, True])
def test_cancelled_batch_keeps_lock_and_reference_until_worker_exits(
    monkeypatch, tmp_path, worker_fails
):
    started, finish = threading.Event(), threading.Event()
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    monkeypatch.setattr(api, "_request_lock", threading.Lock())

    async def save(upload):
        return reference

    def generate(*args, **kwargs):
        started.set()
        assert finish.wait(5), "test did not release worker"
        assert api._request_lock.locked()
        assert reference.exists()
        if worker_fails:
            raise RuntimeError("worker failed after caller cancellation")
        return [b"pcm"]

    monkeypatch.setattr(api, "_save_upload", save)
    monkeypatch.setattr(api, "_generate_batch_pcm", generate)

    async def request():
        return await api.speech_batch(
            texts='["Hello"]', instruction=None, cfg_scale=1,
            ref_audio=SimpleNamespace(filename="reference.wav"),
            ref_text="reference", seed=1, max_new_tokens=10,
        )

    async def run():
        task = asyncio.create_task(request())
        try:
            async with asyncio.timeout(5):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert api._request_lock.locked()
            assert reference.exists()
            with pytest.raises(api.HTTPException) as busy:
                await request()
            assert busy.value.status_code == 409
        finally:
            finish.set()
        async with asyncio.timeout(5):
            while api._request_lock.locked():
                await asyncio.sleep(0.001)
        assert not reference.exists()

    asyncio.run(run())


def test_cancelled_batch_upload_releases_resources(monkeypatch):
    monkeypatch.setattr(api, "_request_lock", threading.Lock())

    async def save(upload):
        raise asyncio.CancelledError()

    monkeypatch.setattr(api, "_save_upload", save)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await api.speech_batch(
                texts='["Hello"]', instruction=None, cfg_scale=1,
                ref_audio=SimpleNamespace(filename="reference.wav"),
                ref_text="reference", seed=1, max_new_tokens=10,
            )
        assert not api._request_lock.locked()

    asyncio.run(run())
