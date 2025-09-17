"""
WooCommerce Handler Module

Manages WooCommerce REST API integration to create product posts,
upload images, and manage product metadata drafts.

Author: GG
Date: 2025-09-16
"""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class WooCommerceHandler:
    """
    Handles communication with WooCommerce API endpoints.
    """

    def __init__(self, store_url: str, consumer_key: str, consumer_secret: str):
        self.store_url = store_url.rstrip("/")
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.api_version = "wc/v3"
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_params = {
            "consumer_key": self.consumer_key,
            "consumer_secret": self.consumer_secret,
        }

    async def initialize(self):
        """
        Initialize aiohttp session.
        """
        self.session = aiohttp.ClientSession()
        logger.info("WooCommerceHandler session initialized.")

    async def close(self):
        """
        Close aiohttp session.
        """
        if self.session:
            await self.session.close()
            logger.info("WooCommerceHandler session closed.")

    async def create_product_draft(
        self, product_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a product draft in WooCommerce store.

        Args:
            product_data: JSON-serializable product info per WooCommerce API.

        Returns:
            WooCommerce API response as dict.
        """
        url = f"{self.store_url}/wp-json/{self.api_version}/products"

        headers = {"Content-Type": "application/json"}

        params = self.auth_params

        async with self.session.post(
            url, json=product_data, params=params, headers=headers
        ) as resp:
            resp_json = await resp.json()

            if resp.status >= 400:
                logger.error(f"WooCommerce create product failed: {resp_json}")
                raise RuntimeError(f"WooCommerce API error: {resp_json}")

            logger.info(f"Product draft created with ID {resp_json.get('id')}.")
            return resp_json

    async def health_check(self) -> Dict[str, Any]:
        """
        Check WooCommerce API connectivity.

        Returns:
            Health status dictionary.
        """
        url = f"{self.store_url}/wp-json/{self.api_version}/"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=self.auth_params) as resp:
                    if resp.status == 200:
                        return {"status": "healthy"}
                    return {"status": "degraded", "http_status": resp.status}
        except Exception as e:
            logger.error(f"WooCommerceHandler health check failed: {e}")
            return {"status": "degraded", "error": str(e)}
