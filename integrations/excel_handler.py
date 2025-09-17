"""
Excel Handler Module

Handles integration with Microsoft Excel Online via Microsoft Graph API.
Performs workbook updates, data batch insertion, and manages authentication.

Author: GG
Date: 2025-09-16
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class ExcelHandler:
    """
    Provides methods to authenticate and update Excel spreadsheets remotely.
    """

    def __init__(self, client_id: str, client_secret: str, tenant_id: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.token: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """
        Initialize authentication session.
        """
        self.session = aiohttp.ClientSession()
        self.token = await self._get_access_token()
        logger.info("ExcelHandler initialized and authenticated.")

    async def close(self):
        """
        Close the client session.
        """
        if self.session:
            await self.session.close()

    async def _get_access_token(self) -> str:
        """
        Obtain OAuth2 access token for Microsoft Graph API.

        Returns:
            Access token string.
        """
        token_url = (
            f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        )
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Failed to get token: {body}")
                    raise RuntimeError(f"Failed to get token: {resp.status}")
                json_resp = await resp.json()
                return json_resp["access_token"]

    async def update_spreadsheet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Excel spreadsheet with provided data.

        Args:
            data: Dictionary representing data to insert/update.

        Returns:
            API response as dict.
        """
        if not self.token or not self.session:
            raise RuntimeError("ExcelHandler not initialized or not authenticated.")

        workbook_id = data.get("workbook_id")
        worksheet = data.get("worksheet", "Sheet1")
        values = data.get("values")

        if not workbook_id or not values:
            raise ValueError("Missing workbook_id or values in request.")

        url = f"https://graph.microsoft.com/v1.0/me/drive/items/{workbook_id}/workbook/worksheets/{worksheet}/range(address='A1')"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        payload = {"values": values}

        async with self.session.patch(url, json=payload, headers=headers) as resp:
            resp_json = await resp.json()
            if resp.status >= 400:
                logger.error(f"Excel update failed: {resp_json}")
                raise RuntimeError(f"Excel API error: {resp_json}")
            logger.info(f"Excel spreadsheet updated successfully.")
            return resp_json

    async def health_check(self) -> Dict[str, Any]:
        """
        Check API connectivity and authentication.

        Returns:
            Health check status.
        """
        try:
            token = await self._get_access_token()
            return {"status": "healthy", "token_valid": True}
        except Exception as e:
            logger.error(f"ExcelHandler health check failed: {e}")
            return {"status": "degraded", "error": str(e)}
