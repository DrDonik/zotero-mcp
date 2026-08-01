"""Regression tests for the ``@mcp.tool`` decorator's return value.

fastmcp 2.x replaces the decorated function with a ``FunctionTool`` wrapper
object, while 3.x leaves the plain function in place. pyproject allows
``fastmcp>=2.14.0``, so both are installable — and several callers use the
module-level name as an ordinary function:

* :mod:`zotero_mcp.cli_standalone` (``search_mod.search_by_tag(...)`` etc.)
* :func:`zotero_mcp.tools.connectors.connector_fetch`, which calls
  ``get_item_fulltext``
* the test suite, which calls tools directly with a ``DummyContext``

:class:`zotero_mcp._app._ZoteroMCP` normalises on the 3.x behaviour. Under
2.x without it, every one of those call sites raises
``TypeError: 'FunctionTool' object is not callable``.
"""

import asyncio
import inspect

import pytest

import zotero_mcp.tools  # noqa: F401 — side-effect: registers all @mcp.tool
from zotero_mcp._app import mcp
from zotero_mcp.tools import retrieval as retrieval_module
from zotero_mcp.tools import search as search_module
from zotero_mcp.tools import write as write_module

# One representative tool per module that other code calls as a function.
SAMPLE_TOOLS = [
    (search_module, "search_items", "zotero_search_items"),
    (search_module, "advanced_search", "zotero_advanced_search"),
    (retrieval_module, "get_item_fulltext", "zotero_get_item_fulltext"),
    (write_module, "search_collections", "zotero_search_collections"),
]


def _registered_tool_names():
    """Tool names known to the app, across fastmcp versions.

    2.x exposes ``get_tools()`` (dict keyed by name), 3.x ``list_tools()``
    (list of tool objects). Both are coroutines.
    """
    if hasattr(mcp, "get_tools"):
        return set(asyncio.run(mcp.get_tools()))
    return {t.name for t in asyncio.run(mcp.list_tools())}


@pytest.mark.parametrize("module,attr,tool_name", SAMPLE_TOOLS)
def test_decorated_tool_is_a_plain_function(module, attr, tool_name):
    """The module attribute stays an ordinary, directly callable function."""
    obj = getattr(module, attr)
    assert inspect.isfunction(obj), (
        f"{module.__name__}.{attr} is {type(obj).__name__}, not a function — "
        f"direct calls from cli_standalone/connectors/tests will fail."
    )
    assert obj.__name__ == attr


@pytest.mark.parametrize("module,attr,tool_name", SAMPLE_TOOLS)
def test_tool_is_still_registered_with_the_app(module, attr, tool_name):
    """Unwrapping the decorator must not skip registration."""
    assert tool_name in _registered_tool_names()


def test_all_tools_registered():
    """Sanity check that every @mcp.tool in tools/ reached the registry."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "zotero_mcp" / "tools"
    decorated = sum(f.read_text().count("@mcp.tool(") for f in root.glob("*.py"))
    assert decorated > 0
    assert len(_registered_tool_names()) == decorated
