"""CSV upload ingestion and processing service."""
from __future__ import annotations

import io
from pathlib import Path

import polars as pl
from fastapi import UploadFile

from settings import settings
from utils.errors import CSVFormatError, FileSizeError, ProcessingError
from utils.tempfiles import TempFileManager


class CsvIngestService:
    """Handles CSV uploads with memory or temp-file strategy."""

    CHUNK_SIZE = 1024 * 1024

    @staticmethod
    async def process_upload(file: UploadFile) -> dict:
        if not file.filename:
            raise CSVFormatError("Missing filename")
        if not file.filename.lower().endswith(".csv"):
            raise CSVFormatError("Only .csv files are supported")

        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        memory_threshold_bytes = settings.in_memory_threshold_mb * 1024 * 1024

        total_bytes = 0
        memory_chunks: list[bytes] = []
        storage_mode = "memory"

        temp_path: Path | None = None
        temp_file_handle = None

        try:
            while True:
                chunk = await file.read(CsvIngestService.CHUNK_SIZE)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise FileSizeError(
                        "Uploaded file exceeds maximum size",
                        details={
                            "max_upload_size_mb": settings.max_upload_size_mb,
                            "size_bytes": total_bytes,
                        },
                    )

                if storage_mode == "memory" and total_bytes > memory_threshold_bytes:
                    storage_mode = "temp_file"
                    temp_path = TempFileManager.get_temp_path(
                        prefix="dc_upload_", suffix=".csv")
                    temp_file_handle = temp_path.open("wb")
                    for previous_chunk in memory_chunks:
                        temp_file_handle.write(previous_chunk)
                    memory_chunks.clear()

                if storage_mode == "memory":
                    memory_chunks.append(chunk)
                else:
                    temp_file_handle.write(chunk)

            if total_bytes == 0:
                raise CSVFormatError("Uploaded CSV is empty")

            if storage_mode == "memory":
                dataframe = pl.read_csv(io.BytesIO(b"".join(memory_chunks)))
            else:
                temp_file_handle.flush()
                temp_file_handle.close()
                temp_file_handle = None
                dataframe = pl.read_csv(str(temp_path))

            return {
                "status": "ok",
                "filename": file.filename,
                "size_bytes": total_bytes,
                "storage_mode": storage_mode,
                "rows": dataframe.height,
                "columns": dataframe.width,
                "column_names": dataframe.columns,
            }
        except FileSizeError:
            raise
        except pl.exceptions.NoDataError as exc:
            raise CSVFormatError("Uploaded CSV has no readable data") from exc
        except pl.exceptions.ComputeError as exc:
            raise CSVFormatError("Invalid CSV format") from exc
        except CSVFormatError:
            raise
        except Exception as exc:
            raise ProcessingError("Failed to process CSV", details={
                                  "reason": str(exc)}) from exc
        finally:
            if temp_file_handle is not None and not temp_file_handle.closed:
                temp_file_handle.close()
            if temp_path is not None:
                TempFileManager.cleanup_file(temp_path)
            await file.close()
