"""API utilities for the application."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from diskcache import Cache
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to console
    ],
)


class API:
    """Fundamentals API class for the application."""

    def __init__(self, base_url: str, api_key: str):
        """Initialize the API class."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def GET(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ):
        """Get data from the API."""
        headers = {
            "apiKey": self.api_key,
            "Accept": "application/json",
        }

        # Add any additional headers
        if additional_headers:
            headers.update(additional_headers)

        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            try:
                response.raise_for_status()
                # Check if response has content before trying to parse JSON
                if not response.text.strip():
                    logger.warning(f"API returned empty response for {url}")
                    return None
                return response.json()
            except Exception as e:
                logger.error(f"API request failed: {e}\nResponse: {response.text}")
                raise Exception(f"API request failed: {e}\nResponse: {response.text}")

    async def GET_FILE(self, endpoint: str, params: Optional[Dict[str, Any]] = None):
        """Get file from the API."""
        headers = {
            "apiKey": self.api_key,
            "Accept": "application/json",
        }
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            try:
                response.raise_for_status()
                return response
            except Exception as e:
                logger.error(f"API request failed: {e}\nResponse: {response.text}")
                raise Exception(f"API request failed: {e}\nResponse: {response.text}")


# Persistent cache with 6 hours TTL
EXPIRE_TIME = 21600
EXPIRE_TIME_24HR = 86400  # 24 hours

active_equity_and_sub_listing_cache = Cache(
    ".cache/active_equity_and_sub_listing", expire=EXPIRE_TIME
)

# Persistent cache for meta data related to file (24 hours TTL)
meta_data_file_cache = Cache(".cache/meta_data_file", expire=EXPIRE_TIME_24HR)


class DefineEdgeFundamentalsAPI(API):
    """Define Edge Fundamentals API class for the application."""

    def __init__(self):
        """Initialize the Define Edge Fundamentals API class."""
        load_dotenv()
        super().__init__(
            base_url=os.environ.get("DEFINE_EDGE_FUNDAMENTAL_APIS_BASE_URL"),
            api_key=os.environ.get("DEFINE_EDGE_API_KEY"),
        )

    async def find_active_equity_and_sub_listing(self):
        """Find active equity and sub listings."""
        cache_key = "active_equity_and_sub_listing"

        # Check the cache first
        cached_data = await asyncio.to_thread(
            active_equity_and_sub_listing_cache.get, cache_key
        )
        if cached_data is not None:
            logger.info("Retrieved active equity and sub listing data from cache")
            return cached_data

        endpoint = "/v1/data-feed/findActiveEquityAndSublisting"
        data = await self.GET(endpoint)

        # Cache the data for 6 hours
        await asyncio.to_thread(
            active_equity_and_sub_listing_cache.set, cache_key, data, expire=EXPIRE_TIME
        )

        return data

    async def get_meta_data_related_to_file(self, start_date: str, end_date: str):
        """Get meta data related to file.

        Args:
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
        """
        cache_key = f"meta_data_file_{start_date}_{end_date}"

        # Check the cache first
        cached_data = await asyncio.to_thread(meta_data_file_cache.get, cache_key)
        if cached_data is not None:
            logger.info(
                f"Retrieved meta data file for {start_date} to {end_date} from cache"
            )
            return cached_data

        endpoint = "/v1/data-feed/getMetaDataRelatedToFile"
        params = {
            "startDate": start_date,
            "endDate": end_date,
        }
        data = await self.GET(endpoint, params)

        # Cache the data for 24 hours
        await asyncio.to_thread(
            meta_data_file_cache.set, cache_key, data, expire=EXPIRE_TIME_24HR
        )

        return data

    async def get_file_by_category_and_attachment_name(
        self, sub_cat_name: str, attachment_ame: str
    ):
        """Get file by category and attachment name."""
        endpoint = "/v1/data-feed/getFileByCategoryAndAttachmentName"
        params = {
            "subcatname": sub_cat_name,
            "attachmentName": attachment_ame,
        }
        return await self.GET_FILE(endpoint, params)

    async def get_predefined_groups(self) -> Dict[str, List[str]]:
        """Get existing predefined groups with caching.

        Returns:
            Dict[str, List[str]]: Dictionary containing group categories and their groups.
            Example: {"Nifty Indices": ["Nifty 50", "Nifty IT", "Nifty Bank"], ...}
        """
        endpoint = "/v1/data-feed/groups/0"
        return await self.GET(endpoint)


