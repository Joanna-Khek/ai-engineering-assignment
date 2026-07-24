PICTURE_DESCRIPTION_PROMPT = """Describe this chart or figure in detail. Include:
- The chart type (bar, line, pie, scatter, etc.)
- The title, if present
- The x-axis and y-axis labels and their units
- All category or series/legend names
- Approximate or exact values for each data point or bar
- Any visible trend, comparison, or notable pattern (e.g., highest/lowest values, growth, decline)

Be specific and quantitative rather than general. Do not skip numeric values even if you have to estimate them.
"""


SYSTEM_PROMPT_TEMPLATE = """You are a careful document extraction assistant.

You are working with the document located at: {doc_json_path}

This document in in Docling's JSON format. Extract the requested field from it.
Include the 'self_refs' — a list of every item (text or table) in the JSON where the
value was found (e.g. '#/tables/0' or '#/texts/42').
If the value came from a table cell, use the self_ref of that table.

"""
