import os
import json
import uuid
from pathlib import Path
from loguru import logger

from PIL import Image
from IPython.display import display

from collections import defaultdict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from docling_core.types.doc.common.reference import RefItem
from docling_core.types.doc.document import DoclingDocument

from ai_engineering_assignment.part1.prompts import SYSTEM_PROMPT_TEMPLATE
from ai_engineering_assignment.settings import MainConfig
from ai_engineering_assignment.part1.schema.query import ExtractedField
from ai_engineering_assignment.part1.visual_grounding import draw_multiple_bboxes


class Query:
    def __init__(self, doc_json_path: str):
        self.configs = MainConfig()
        self.doc_json = DoclingDocument.load_from_json(doc_json_path)

        # Initialise output directory
        self.output_dir = Path(self.configs.ROOT) / "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialise Model
        self.model = ChatAnthropic(
            model_name=self.configs.app.llm.model,
            api_key=self.configs.ANTHROPIC_API_KEY,
        )  # type: ignore[call-arg]

        self.structured_model = self.model.with_structured_output(ExtractedField)

        # Because the full version is over 1M tokens, we create a lean version,
        # which extracts only the key fields important for the LLM.
        self.lean_doc_json = json.dumps(self.build_lean_payload(doc=self.doc_json))

        logger.debug(
            f"After removing unnecessary tokens from raw json, ~{len(self.lean_doc_json) // 4} tokens"
        )  # rough estimate

    @staticmethod
    def build_lean_payload(doc):
        items = []

        for t in doc.texts:
            items.append(
                {
                    "self_ref": t.self_ref,
                    "label": str(t.label),
                    "text": t.text,
                }
            )

        for tbl in doc.tables:
            items.append(
                {
                    "self_ref": tbl.self_ref,
                    "label": "table",
                    "markdown": tbl.export_to_markdown(doc=doc),  # pass doc explicitly
                }
            )

        return items

    def _generate_image_data(self, result: ExtractedField):
        """Resolve references and prepare visual grounding data."""

        resolved_items = []
        for ref in result.self_refs:
            try:
                item = RefItem(cref=ref).resolve(self.doc_json)
                resolved_items.append((ref, item))
            except Exception as e:
                logger.warning(f"Error resolving {ref}: {type(e).__name__}: {e}")

        if not resolved_items:
            logger.debug("No resolved items to show.")
            return [], {}

        # Group items by page
        by_page = defaultdict(list)
        refs_by_page = defaultdict(list)

        for ref, item in resolved_items:
            page_no = item.prov[0].page_no
            by_page[page_no].append(item)
            refs_by_page[page_no].append(ref)

        run_id = uuid.uuid4()

        saved_paths = []
        self_ref_image_paths = {}

        for page_no, items in sorted(by_page.items()):
            # Needed by draw_multiple_bboxes()
            ref_lookup = {
                item.self_ref: ref
                for ref, item in resolved_items
                if item.prov[0].page_no == page_no
            }

            for ref in refs_by_page[page_no]:
                img = draw_multiple_bboxes(
                    doc=self.doc_json,
                    items=items,
                    ref_lookup=ref_lookup,
                    selected_ref=ref,
                )

                filename = (
                    f"page_{page_no}_"
                    f"{ref.replace('/', '_').replace('#', '')}_"
                    f"{run_id}.png"
                )

                path = os.path.join(self.output_dir, filename)

                img.save(path)

                saved_paths.append(path)
                self_ref_image_paths[ref] = path

        return (
            saved_paths,
            self_ref_image_paths,
        )

    # def _generate_images(self, result: ExtractedField):
    #     image_paths, _, _ = self._generate_image_data(result)
    #     return image_paths

    def query(self, user_query: str, show_images: bool = False) -> dict:
        messages = [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_TEMPLATE.format(
                            doc_json_path=self.lean_doc_json
                        ),
                        "cache_control": {
                            "type": "ephemeral"
                        },  # cache everything up to here
                    }
                ]
            ),
            HumanMessage(content=user_query),
        ]

        result = self.structured_model.invoke(messages)
        image_paths, self_ref_image_paths = self._generate_image_data(result=result)

        if show_images:  # only in notebook
            for path in image_paths:
                image = Image.open(path)

                # Scale to 50%
                image = image.resize(
                    (image.width // 2, image.height // 2),
                    Image.Resampling.LANCZOS,
                )

                display(image)

        logger.info(f"Answer: {result.model_dump()["value"]}")

        return {
            "extraction": result.model_dump(),
            "image_paths": image_paths,
            "self_ref_image_paths": self_ref_image_paths,
        }
