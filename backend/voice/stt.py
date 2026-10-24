"""Local speech-to-text API backed by faster-whisper.

This module deliberately knows nothing about microphones, HTTP, or FastAPI.  It
accepts audio in a format supported by ``faster_whisper.WhisperModel.transcribe``
and returns application-friendly Python values.  Audio capture belongs to the
caller (see ``microphone_test.py`` for a temporary manual test client).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from os import PathLike, fspath
from threading import Lock
from typing import Any, BinaryIO, Callable, Mapping, Optional, Tuple, Union


AudioInput = Union[str, PathLike[str], BinaryIO, Any]
ModelFactory = Callable[..., Any]


@dataclass(frozen=True)
class TranscriptionSegment:
    """One timestamped piece of recognized speech."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """The complete result returned by :class:`SpeechToText`."""

    text: str
    duration: Optional[float]
    segments: Tuple[TranscriptionSegment, ...]


class SpeechToText:
    """Reusable, lazily loaded faster-whisper transcription service.

    Parameters mirror the most useful ``WhisperModel`` constructor options.
    The model is not loaded until the first transcription, which keeps imports
    and application startup cheap.  ``model_factory`` is primarily useful for
    tests or for supplying a compatible model implementation.
    """

    def __init__(
        self,
        model_size: str = "medium",
        *,
        device: str = "auto",
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        model_factory: Optional[ModelFactory] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root
        self.local_files_only = local_files_only
        self._model_factory = model_factory
        self._model: Optional[Any] = None
        self._model_lock = Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            factory = self._model_factory
            if factory is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "faster-whisper is required for transcription. "
                        "Install the project dependencies with "
                        "`python -m pip install -r backend/requirements.txt`."
                    ) from exc
                factory = WhisperModel

            model_options = {
                "device": self.device,
                "compute_type": self.compute_type,
                "cpu_threads": self.cpu_threads,
                "num_workers": self.num_workers,
                "download_root": self.download_root,
                "local_files_only": self.local_files_only,
            }
            self._model = factory(self.model_size, **model_options)
            return self._model

    def transcribe(
        self,
        audio: AudioInput,
        *,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio source.

        ``audio`` may be a file path, a binary file-like object, or a mono
        NumPy float32 waveform, matching faster-whisper's native API.  Extra
        faster-whisper arguments can be supplied through ``options``.
        """

        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be either 'transcribe' or 'translate'")
        if beam_size < 1:
            raise ValueError("beam_size must be at least 1")

        transcribe_options = dict(options or {})
        if "language" in transcribe_options:
            raise ValueError("language is fixed to English")

        reserved = {"task", "beam_size", "vad_filter", "initial_prompt"}
        overlap = reserved.intersection(transcribe_options)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"options duplicates explicit argument(s): {names}")

        audio_source = fspath(audio) if isinstance(audio, PathLike) else audio
        segments_iter, info = self._get_model().transcribe(
            audio_source,
            language="en",
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
            **transcribe_options,
        )

        segments = tuple(
            TranscriptionSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=str(segment.text).strip(),
            )
            for segment in segments_iter
        )
        text = " ".join(segment.text for segment in segments if segment.text).strip()

        return TranscriptionResult(
            text=text,
            duration=_optional_float(getattr(info, "duration", None)),
            segments=segments,
        )

    async def transcribe_async(
        self,
        audio: AudioInput,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Run blocking model inference without blocking an async event loop."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.transcribe(audio, **kwargs))


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)
