"""Module entry point: ``python -m pirag.mcp`` runs the stdio server.

Equivalent to ``python -m pirag.mcp.serve``. Keeping both forms work
matches the project's documented MCP-style local-client pattern
where servers are launched as ``python -m <package>``.
"""
from .serve import main

if __name__ == "__main__":
    main()
