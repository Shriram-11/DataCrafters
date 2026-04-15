"""Analysis V2 orchestration service."""
from __future__ import annotations

from fastapi import UploadFile

from analysis_modules.csv_loader import load_csv_upload
from analysis_modules.exploration_module import (
    build_correlation,
    build_recommendations,
    build_univariate,
)
from analysis_modules.profile_module import build_profile
from analysis_modules.quality_module import build_quality


class AnalysisServiceV2:
    """Single orchestration layer for all analysis v2 endpoints."""

    @staticmethod
    async def profile(file: UploadFile) -> dict:
        df = await load_csv_upload(file)
        return {
            "success": True,
            "message": "Profile generated successfully",
            "data": build_profile(df),
        }

    @staticmethod
    async def quality(file: UploadFile) -> dict:
        df = await load_csv_upload(file)
        return {
            "success": True,
            "message": "Quality checks completed successfully",
            "data": build_quality(df),
        }

    @staticmethod
    async def correlation(file: UploadFile) -> dict:
        df = await load_csv_upload(file)
        data = build_correlation(df)
        message = "Correlation computed successfully"
        if not data["columns"]:
            message = "No numeric columns available for correlation"
        return {
            "success": True,
            "message": message,
            "data": data,
        }

    @staticmethod
    async def univariate(file: UploadFile) -> dict:
        df = await load_csv_upload(file)
        return {
            "success": True,
            "message": "Univariate statistics computed successfully",
            "data": build_univariate(df),
        }

    @staticmethod
    async def recommendations(file: UploadFile) -> dict:
        df = await load_csv_upload(file)
        return {
            "success": True,
            "message": "Recommendations generated successfully",
            "data": build_recommendations(df),
        }
