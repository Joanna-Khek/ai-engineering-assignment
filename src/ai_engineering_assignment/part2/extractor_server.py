from dateutil import parser as dateutil_parser
from mcp.server.fastmcp import FastMCP

from ai_engineering_assignment.part1.query import Query
from ai_engineering_assignment.settings import MainConfig

mcp = FastMCP("Extractor")
configs = MainConfig()  # loaded once at server startup


@mcp.tool()
def extract_data(user_query: str, doc_json_path: str):
    """
    Extract a piece of information from a document that has already been converted to Docling's JSON format.
    Use this tool when the user asks a question about the content of a specific document.

    Args:
        user_query (str): A natural-language question describing what to extract from the document
        doc_json_path (str): Absolute file path to the Docling JSON export of the document to search
    """
    query = Query(doc_json_path=doc_json_path)
    return query.query(user_query=user_query)


@mcp.tool()
def parse_date(date_string: str) -> str:
    """
    Convert a date string in any common human-readable format into strict
    ISO format (YYYY-MM-DD). Use this tool whenever you have a date string.

    Args:
        date_string (str): A date in any common format, e.g. "March 3, 2024",
            "03/04/2024", "2024-03-04T10:00:00Z".

    Returns:
        The date in "YYYY-MM-DD" format, or a string starting with "Unable
        to parse" if the input could not be interpreted as a date.
    """
    try:
        parsed = dateutil_parser.parse(date_string)
        return parsed.strftime("%Y-%m-%d")
    except (
        ValueError,
        OverflowError,
        TypeError,
    ) as e:  # dateutil.parser.parse can raise these errors
        return f"Unable to parse '{date_string}': {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
