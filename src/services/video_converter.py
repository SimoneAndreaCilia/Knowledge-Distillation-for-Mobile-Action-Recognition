# -*- coding: utf-8 -*-
"""VideoConverter — converts non-web-playable video formats to H.264 MP4.

HTML5 ``<video>`` elements require specific codec/container combinations:
  - ``.mp4`` with H.264 (AVC)
  - ``.webm`` with VP8/VP9

The HMDB-51 dataset ships AVI files (MPEG-4 Part 2 / DivX), which are
**not playable** in any modern browser.  This service transcodes them to
H.264 MP4 using the ``ffmpeg`` binary bundled with ``imageio-ffmpeg``
and caches the result on disk so each video is converted at most once.

Usage::

    converter = VideoConverter()
    mp4_path  = converter.ensure_web_playable("/path/to/video.avi")
"""

import hashlib
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_WEB_PLAYABLE = frozenset({".mp4", ".webm", ".ogg"})
_DEFAULT_MAX_AGE_DAYS = 7


def _get_ffmpeg_exe() -> str:
    """Locate the ffmpeg executable.

    Tries ``imageio-ffmpeg`` first (pip-installable, ships a static binary),
    then falls back to the system ``ffmpeg``.
    """
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    import shutil  # noqa: PLC0415

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError(
        "ffmpeg not found. Install it with:  pip install imageio-ffmpeg"
    )


class VideoConverter:
    """Converts videos to browser-playable H.264 MP4 using ffmpeg.

    Args:
        cache_dir:    Where to store converted files.  Defaults to a
                      subdirectory of the system temp folder.
        max_age_days: Cached files older than this are removed at startup.
                      Set to ``0`` to disable cleanup.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_age_days: int = _DEFAULT_MAX_AGE_DAYS,
    ) -> None:
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "kd_video_cache"
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg = _get_ffmpeg_exe()
        logger.info("VideoConverter: cache=%s, ffmpeg=%s", self._cache_dir, self._ffmpeg)
        if max_age_days > 0:
            self._cleanup_stale(max_age_days)

    # ------------------------------------------------------------------
    # Cache maintenance
    # ------------------------------------------------------------------

    def _cleanup_stale(self, max_age_days: int) -> None:
        """Remove cached files older than *max_age_days*."""
        cutoff = time.time() - (max_age_days * 86_400)
        removed = 0
        freed_bytes = 0
        for f in self._cache_dir.glob("*.mp4"):
            try:
                if f.stat().st_mtime < cutoff:
                    size = f.stat().st_size
                    f.unlink()
                    removed += 1
                    freed_bytes += size
            except OSError:
                pass
        if removed:
            logger.info(
                "VideoConverter: cleaned %d stale file(s), freed %.1f MB",
                removed,
                freed_bytes / (1024 * 1024),
            )

    def ensure_web_playable(self, video_path: str) -> str:
        """Return a path to a browser-playable version of *video_path*.

        If the source is already a playable format it is returned as-is.
        Otherwise it is transcoded to H.264 MP4 and cached.
        """
        src = Path(video_path)

        # Already web-playable — skip conversion
        if src.suffix.lower() in _WEB_PLAYABLE:
            return video_path

        # Deterministic cache key from the absolute source path
        cache_key = hashlib.md5(str(src.resolve()).encode()).hexdigest()
        cached = self._cache_dir / f"{cache_key}.mp4"

        if cached.exists():
            logger.debug("VideoConverter: cache hit for %s", src.name)
            return str(cached)

        logger.info("VideoConverter: converting %s → H.264 MP4", src.name)
        self._transcode(src, cached)
        return str(cached)

    def _transcode(self, src: Path, dst: Path) -> None:
        """Transcode *src* to H.264 MP4 at *dst* using ffmpeg."""
        cmd = [
            self._ffmpeg,
            "-y",                   # overwrite
            "-i", str(src),         # input
            "-c:v", "libx264",      # H.264 codec
            "-preset", "ultrafast", # speed over compression
            "-crf", "23",           # reasonable quality
            "-pix_fmt", "yuv420p",  # maximum browser compat
            "-an",                  # no audio (HMDB-51 has none)
            "-loglevel", "error",
            str(dst),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("ffmpeg failed: %s", exc.stderr)
            raise RuntimeError(f"Video conversion failed: {exc.stderr}") from exc

        size_kb = dst.stat().st_size / 1024
        logger.info("VideoConverter: wrote %s (%.1f KB)", dst.name, size_kb)
