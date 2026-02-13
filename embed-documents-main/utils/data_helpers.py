"""Data retrieval and mapping utilities.

This module contains helper functions for retrieving and mapping financial data
from the DefineEdge APIs. These utilities provide convenient access to commonly
used data structures like fincode-symbol mappings, stock listings, etc.

Organization Pattern:
- API client classes live in `api_utils.py`
- MF scheme helpers remain in `api_utils.py` (get_all_schemes, get_all_schemes_df, etc.)
- Stock data retrieval/mapping utilities that USE those APIs live here
- Pure lookup/search functions with fuzzy matching live in `find_*.py` files

Usage Patterns:

1. Stock Data - Automatic initialization at server startup (via webapp.py lifespan):
    ```python
    from utils.data_helpers import fincode_to_symbol, symbol_to_fincode

    # Happens automatically when you run: langgraph dev
    # No manual initialization needed!

    # Works in both sync and async functions - no await needed!
    symbol = fincode_to_symbol(100034)  # "BAJFINANCE"
    fincode = symbol_to_fincode("BAJFINANCE")  # 100034

    # Or get full mappings
    fincode_map = get_fincode_map()  # Dict[int, str]
    symbol_map = get_symbol_map()    # Dict[str, int]
    stocks_df = get_stocks_dataframe()  # pd.DataFrame
    ```

2. Metadata Helpers - Automatic initialization at server startup (via webapp.py lifespan):
    ```python
    from utils.data_helpers import (
        get_metadata_mapping,
        get_metadata_item_by_attachment_name,
        get_metadata_dataframe
    )

    # Happens automatically when you run: langgraph dev
    # No manual initialization needed!

    # Works in both sync and async functions - no await needed!
    mapping = get_metadata_mapping()
    item = mapping.get("some_attachment.pdf")

    # Get a specific item by attachment name
    item = get_metadata_item_by_attachment_name("some_attachment.pdf")

    # Get as DataFrame for filtering/analysis
    df = get_metadata_dataframe()
    tcs_docs = df[df['fincode'] == 100034]
    ```
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.api_utils import DefineEdgeFundamentalsAPI

logger = logging.getLogger(__name__)


# ============================================================================
# Module-Level Cache Variables (for synchronous access)
# ============================================================================

# Stock data cache - populated by initialize_stock_data()
_FINCODE_TO_SYMBOL_MAP: Optional[Dict[int, str]] = None
_SYMBOL_TO_FINCODE_MAP: Optional[Dict[str, int]] = None
_STOCKS_DF: Optional[pd.DataFrame] = None

# Metadata cache - populated by initialize_metadata_data()
_METADATA_MAPPING: Optional[Dict[str, Dict[str, Any]]] = None
_METADATA_DF: Optional[pd.DataFrame] = None

# Locks for thread-safe initialization
_STOCK_INITIALIZATION_LOCK = asyncio.Lock()
_METADATA_INITIALIZATION_LOCK = asyncio.Lock()

# ============================================================================
# Initialization Functions
# ============================================================================


async def initialize_stock_data() -> None:
    """Initialize stock data mappings at server startup.

    This function loads all stock data once and caches it in module-level variables,
    allowing for synchronous access via fincode_to_symbol() and symbol_to_fincode().

    Should be called during application startup for best performance.

    Example:
        >>> # In your main.py or application startup
        >>> await initialize_stock_data()
        >>> logger.info("Stock data initialized")
    """
    global _FINCODE_TO_SYMBOL_MAP, _SYMBOL_TO_FINCODE_MAP, _STOCKS_DF

    async with _STOCK_INITIALIZATION_LOCK:
        # Only initialize if not already done
        if _FINCODE_TO_SYMBOL_MAP is not None:
            logger.info("Stock data already initialized, skipping")
            return

        logger.info("Initializing stock data mappings...")

        listing_data = await DefineEdgeFundamentalsAPI().find_active_equity_and_sub_listing()

        # Handle both list and dict responses
        if isinstance(listing_data, dict):
            data: List[Dict[str, Any]] = listing_data.get("data", [])  # type: ignore[assignment]
        else:
            data = listing_data or []

        if not data:
            logger.warning("No stock data available during initialization")
            _FINCODE_TO_SYMBOL_MAP = {}
            _SYMBOL_TO_FINCODE_MAP = {}
            _STOCKS_DF = pd.DataFrame()
            return

        # Build both mappings
        fincode_map: Dict[int, str] = {}
        symbol_map: Dict[str, int] = {}

        for item in data:
            fincode = item.get("fincode")
            symbol = item.get("symbol") or item.get("bseScripId")

            if fincode is not None and symbol:
                fincode_map[fincode] = symbol
                symbol_map[symbol.upper()] = fincode

        # Build DataFrame
        df = pd.DataFrame(data)
        if "fincode" in df.columns:
            df = df.set_index("fincode")

        # Set module-level variables
        _FINCODE_TO_SYMBOL_MAP = fincode_map
        _SYMBOL_TO_FINCODE_MAP = symbol_map
        _STOCKS_DF = df

        logger.info(
            f"Stock data initialized: {len(fincode_map)} stocks, "
            f"{len(symbol_map)} symbols"
        )


async def initialize_metadata_data() -> None:
    """Initialize metadata mappings at server startup.

    This function loads all metadata once (20 years of data) and caches it in
    module-level variables, allowing for synchronous access via get_metadata_mapping()
    and related functions.

    Should be called during application startup for best performance.

    Example:
        >>> # In your main.py or application startup
        >>> await initialize_metadata_data()
        >>> logger.info("Metadata initialized")
    """
    global _METADATA_MAPPING, _METADATA_DF

    async with _METADATA_INITIALIZATION_LOCK:
        # Only initialize if not already done
        if _METADATA_MAPPING is not None:
            logger.info("Metadata already initialized, skipping")
            return

        logger.info("Initializing metadata mappings (20 years of data)...")

        # Get default date range (20 years ago to today)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=20 * 365)).strftime("%Y-%m-%d")

        data = await DefineEdgeFundamentalsAPI().get_meta_data_related_to_file(
            start_date=start_date, end_date=end_date
        )

        # Handle both list and dict responses
        if isinstance(data, dict):
            items: List[Dict[str, Any]] = data.get("data", [])  # type: ignore[assignment]
        else:
            items = data or []

        if not items:
            logger.warning("No metadata available during initialization")
            _METADATA_MAPPING = {}
            _METADATA_DF = pd.DataFrame()
            return

        # Build mapping from attachmentname to full item
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in items:
            attachment_name = item.get("attachmentname")
            if attachment_name:
                mapping[attachment_name] = item

        # Build DataFrame
        df = pd.DataFrame(items)
        if "attachmentname" in df.columns:
            df = df.set_index("attachmentname")

        # Set module-level variables
        _METADATA_MAPPING = mapping
        _METADATA_DF = df

        logger.info(
            f"Metadata initialized: {len(mapping)} items, "
            f"{df['fincode'].nunique() if 'fincode' in df.columns else 0} unique fincodes"
        )


def is_stock_data_initialized() -> bool:
    """Check if stock data has been initialized.

    Returns:
        bool: True if initialize_stock_data() has been called, False otherwise.
    """
    return _FINCODE_TO_SYMBOL_MAP is not None


def is_metadata_initialized() -> bool:
    """Check if metadata has been initialized.

    Returns:
        bool: True if initialize_metadata_data() has been called, False otherwise.
    """
    return _METADATA_MAPPING is not None


# ============================================================================
# Synchronous Getters (use cached data)
# ============================================================================


def fincode_to_symbol(fincode: int) -> Optional[str]:
    """Get stock symbol for a given fincode (synchronous).

    Args:
        fincode: Fincode to look up.

    Returns:
        Optional[str]: The stock symbol if found, None otherwise.

    Raises:
        RuntimeError: If stock data hasn't been initialized yet.

    Note:
        Requires initialize_stock_data() to be called first.

    Example:
        >>> symbol = fincode_to_symbol(100034)
        >>> print(symbol)  # "TCS"
    """
    if _FINCODE_TO_SYMBOL_MAP is None:
        raise RuntimeError(
            "Stock data not initialized. Call 'await initialize_stock_data()' first."
        )
    return _FINCODE_TO_SYMBOL_MAP.get(fincode)


def symbol_to_fincode(symbol: str) -> Optional[int]:
    """Get fincode for a given stock symbol (synchronous).

    Args:
        symbol: Stock symbol to look up (case-insensitive).

    Returns:
        Optional[int]: The fincode if found, None otherwise.

    Raises:
        RuntimeError: If stock data hasn't been initialized yet.

    Note:
        Requires initialize_stock_data() to be called first.

    Example:
        >>> fincode = symbol_to_fincode("TCS")
        >>> print(fincode)  # 100034
    """
    if _SYMBOL_TO_FINCODE_MAP is None:
        raise RuntimeError(
            "Stock data not initialized. Call 'await initialize_stock_data()' first."
        )
    return _SYMBOL_TO_FINCODE_MAP.get(symbol.upper())


def get_fincode_map() -> Dict[int, str]:
    """Get the complete fincode-to-symbol mapping (synchronous).

    Returns:
        Dict[int, str]: Dictionary mapping fincode to symbol.

    Raises:
        RuntimeError: If stock data hasn't been initialized yet.

    Note:
        Requires initialize_stock_data() to be called first.
        Returns a reference to the cached dictionary (not a copy).
    """
    if _FINCODE_TO_SYMBOL_MAP is None:
        raise RuntimeError(
            "Stock data not initialized. Call 'await initialize_stock_data()' first."
        )
    return _FINCODE_TO_SYMBOL_MAP


def get_symbol_map() -> Dict[str, int]:
    """Get the complete symbol-to-fincode mapping (synchronous).

    Returns:
        Dict[str, int]: Dictionary mapping symbol (uppercase) to fincode.

    Raises:
        RuntimeError: If stock data hasn't been initialized yet.

    Note:
        Requires initialize_stock_data() to be called first.
        Returns a reference to the cached dictionary (not a copy).
    """
    if _SYMBOL_TO_FINCODE_MAP is None:
        raise RuntimeError(
            "Stock data not initialized. Call 'await initialize_stock_data()' first."
        )
    return _SYMBOL_TO_FINCODE_MAP


def get_stocks_dataframe() -> pd.DataFrame:
    """Get the complete stocks DataFrame (synchronous).

    Returns:
        pd.DataFrame: DataFrame indexed by fincode with all stock details.

    Raises:
        RuntimeError: If stock data hasn't been initialized yet.

    Note:
        Requires initialize_stock_data() to be called first.
    """
    if _STOCKS_DF is None:
        raise RuntimeError(
            "Stock data not initialized. Call 'await initialize_stock_data()' first."
        )
    return _STOCKS_DF


# ============================================================================
# Metadata Synchronous Getters (use cached data)
# ============================================================================


def get_metadata_mapping() -> Dict[str, Dict[str, Any]]:
    """Get the complete metadata mapping (synchronous).

    Returns the cached mapping from attachment name to full metadata item.
    Data is loaded at server startup covering 20 years of history.

    Returns:
        Dict mapping attachmentname (str) to full metadata item (dict).
        Each item contains:
        - fincode: Financial code
        - attachmentname: Unique attachment identifier
        - newsDt: News/document date
        - subcatname: Sub-category name

    Raises:
        RuntimeError: If metadata hasn't been initialized yet.

    Note:
        Requires initialize_metadata_data() to be called first.
        Returns a reference to the cached dictionary (not a copy).

    Example:
        >>> mapping = get_metadata_mapping()
        >>> item = mapping.get("some_attachment_name.pdf")
        >>> print(item['fincode'], item['subcatname'])
    """
    if _METADATA_MAPPING is None:
        raise RuntimeError(
            "Metadata not initialized. Call 'await initialize_metadata_data()' first."
        )
    return _METADATA_MAPPING


def get_metadata_item_by_attachment_name(
    attachment_name: str,
) -> Optional[Dict[str, Any]]:
    """Get metadata item by attachment name (synchronous).

    Looks up a specific document/file metadata item using its unique attachment name
    from the cached data loaded at server startup.

    Args:
        attachment_name: The unique attachment name to look up.

    Returns:
        The metadata item dict if found, None otherwise.
        Item contains: fincode, attachmentname, newsDt, subcatname

    Raises:
        RuntimeError: If metadata hasn't been initialized yet.

    Note:
        Requires initialize_metadata_data() to be called first.

    Example:
        >>> item = get_metadata_item_by_attachment_name("report_2024.pdf")
        >>> if item:
        ...     print(f"Fincode: {item['fincode']}, Date: {item['newsDt']}")
    """
    mapping = get_metadata_mapping()
    return mapping.get(attachment_name)


def get_metadata_dataframe() -> pd.DataFrame:
    """Get the complete metadata DataFrame (synchronous).

    Returns the cached metadata as a DataFrame indexed by attachment name.
    Data is loaded at server startup covering 20 years of history.

    Returns:
        DataFrame indexed by attachmentname with columns:
        - fincode: Financial code
        - newsDt: News/document date
        - subcatname: Sub-category name

    Raises:
        RuntimeError: If metadata hasn't been initialized yet.

    Note:
        Requires initialize_metadata_data() to be called first.

    Example:
        >>> df = get_metadata_dataframe()
        >>> # Filter by fincode
        >>> tcs_docs = df[df['fincode'] == 100034]
        >>> # Filter by category
        >>> results = df[df['subcatname'] == 'Quarterly Results']
    """
    if _METADATA_DF is None:
        raise RuntimeError(
            "Metadata not initialized. Call 'await initialize_metadata_data()' first."
        )
    return _METADATA_DF
