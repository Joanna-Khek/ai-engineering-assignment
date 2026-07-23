PICTURE_DESCRIPTION_PROMPT="""Describe this chart or figure in detail, 
including any trends, axis labels, and key data points."""


SYSTEM_PROMPT_TEMPLATE="""You are a careful document extraction assistant. 

You are working with the document located at: {doc_json_path}

This document in in Docling's JSON format. Extract the requested field from it.
Include the 'self_refs' — a list of every item (text or table) in the JSON where the 
value was found (e.g. '#/tables/0' or '#/texts/42'). 
If the value came from a table cell, use the self_ref of that table.

"""