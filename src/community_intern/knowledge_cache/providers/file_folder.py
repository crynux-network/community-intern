from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from community_intern.knowledge_cache.models import CacheRecord, CacheState, FileMetadata, SourceType
from community_intern.knowledge_cache.utils import format_rfc3339, hash_text

logger = logging.getLogger(__name__)


class FileFolderProvider:
    def __init__(self, *, sources_dir: str) -> None:
        self._sources_dir = Path(sources_dir)
        self._file_sources: Dict[str, Path] = {}

    async def discover(self, *, now: datetime) -> Dict[str, SourceType]:
        _ = now
        sources: Dict[str, SourceType] = {}
        self._file_sources = {}

        if not self._sources_dir.exists():
            logger.debug("FileFolderProvider discover: sources_dir missing. path=%s", self._sources_dir)
            return sources

        logger.debug("FileFolderProvider discover: start. sources_dir=%s", self._sources_dir)
        scanned = 0
        for file_path in self._sources_dir.rglob("*"):
            scanned += 1
            if scanned % 2000 == 0:
                logger.debug("FileFolderProvider discover: scanning. scanned=%s discovered=%s", scanned, len(sources))
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue
            try:
                rel_path = file_path.relative_to(self._sources_dir).as_posix()
            except ValueError:
                continue
            sources[rel_path] = "file"
            self._file_sources[rel_path] = file_path

        logger.debug(
            "FileFolderProvider discover: completed. scanned=%s discovered=%s sources_dir=%s",
            scanned,
            len(sources),
            self._sources_dir,
        )
        return sources

    async def init_record(self, *, source_id: str, now: datetime) -> CacheRecord | None:
        file_path = self._file_sources.get(source_id)
        if not file_path:
            return None

        logger.debug("FileFolderProvider init_record: start. source_id=%s path=%s", source_id, file_path)
        try:
            stat = file_path.stat()
        except OSError as e:
            logger.warning("Failed to stat file source. path=%s error=%s", file_path, e)
            return None

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF8 file source. path=%s", file_path)
            return None
        except OSError as e:
            logger.warning("Failed to read file source. path=%s error=%s", file_path, e)
            return None

        content_hash = hash_text(text)
        logger.debug(
            "FileFolderProvider init_record: completed. source_id=%s size_bytes=%s text_chars=%s",
            source_id,
            stat.st_size,
            len(text),
        )
        return CacheRecord(
            source_type="file",
            content_hash=content_hash,
            summary_text="",
            last_indexed_at=format_rfc3339(now),
            summary_pending=True,
            file=FileMetadata(rel_path=source_id, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        )

    async def refresh(self, *, cache: CacheState, now: datetime) -> bool:
        changed = False
        logger.debug("FileFolderProvider refresh: start. known_files=%s", len(self._file_sources))
        for rel_path, file_path in self._file_sources.items():
            record = cache.sources.get(rel_path)
            if record is None:
                continue
            if record.source_type != "file":
                continue

            try:
                stat = file_path.stat()
            except OSError as e:
                logger.warning("Failed to stat file source. path=%s error=%s", file_path, e)
                continue

            file_meta = record.file
            if not file_meta:
                file_meta = FileMetadata(rel_path=rel_path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

            if file_meta.size_bytes == stat.st_size and file_meta.mtime_ns == stat.st_mtime_ns:
                continue

            logger.debug(
                "FileFolderProvider refresh: changed file detected. rel_path=%s old_size=%s new_size=%s",
                rel_path,
                file_meta.size_bytes,
                stat.st_size,
            )
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("Skipping non-UTF8 file source. path=%s", file_path)
                continue
            except OSError as e:
                logger.warning("Failed to read file source. path=%s error=%s", file_path, e)
                continue

            content_hash = hash_text(text)
            record.file = FileMetadata(rel_path=rel_path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if content_hash != record.content_hash or record.summary_pending:
                record.content_hash = content_hash
                record.summary_pending = True
            changed = True

        logger.debug("FileFolderProvider refresh: completed. changed=%s", changed)
        return changed

    async def load_text(self, *, source_id: str) -> str | None:
        file_path = self._file_sources.get(source_id)
        if not file_path:
            return None
        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.exception("Failed to load file source text. source_id=%s path=%s", source_id, file_path)
            return None

