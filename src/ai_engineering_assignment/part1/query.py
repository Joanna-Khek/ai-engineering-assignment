import os
import json
from pathlib import Path
from loguru import logger

from PIL import Image
from IPython.display import display

from PIL import ImageDraw
from collections import defaultdict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from docling_core.types.doc.common.reference import RefItem
from docling_core.types.doc.document import DoclingDocument

from ai_engineering_assignment.part1.prompts import SYSTEM_PROMPT_TEMPLATE
from ai_engineering_assignment.settings import MainConfig
from ai_engineering_assignment.part1.schema.query import ExtractedField


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
        )

        self.structured_model = self.model.with_structured_output(ExtractedField)
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

    @staticmethod
    def draw_multiple_bboxes(doc, items, highlight_alpha=90):
        """Draw bounding boxes on the rendered image for selected chunks"""
        page_no = items[0].prov[0].page_no
        page = doc.pages[page_no]
        pil_image = page.image.pil_image.copy().convert("RGBA")

        scale_x = pil_image.width / page.size.width
        scale_y = pil_image.height / page.size.height

        # Transparent overlay for the yellow highlight fill
        overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        for item in items:
            bbox = item.prov[0].bbox.to_top_left_origin(page_height=page.size.height)
            left, top, right, bottom = (
                bbox.l * scale_x,
                bbox.t * scale_y,
                bbox.r * scale_x,
                bbox.b * scale_y,
            )

            # Semi-transparent yellow fill (highlighter effect)
            overlay_draw.rectangle(
                [(left, top), (right, bottom)], fill=(255, 255, 0, highlight_alpha)
            )

        # Composite the highlight overlay onto the base image
        pil_image = Image.alpha_composite(pil_image, overlay)

        # Draw solid red borders + labels on top, after compositing
        draw = ImageDraw.Draw(pil_image)
        for item in items:
            bbox = item.prov[0].bbox.to_top_left_origin(page_height=page.size.height)
            left, top, right, bottom = (
                bbox.l * scale_x,
                bbox.t * scale_y,
                bbox.r * scale_x,
                bbox.b * scale_y,
            )

            draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
            draw.text((left, max(top - 15, 0)), item.label, fill="red")

        return pil_image.convert(
            "RGB"
        )  # convert back for saving as JPEG/normal display

    def _generate_images(self, result: ExtractedField) -> list[str]:
        """Render the images with the bounding boxes"""
        resolved_items = []
        for ref in result.self_refs:
            try:
                item = RefItem(cref=ref).resolve(self.doc_json)
                resolved_items.append(item)
            except Exception as e:
                print(f"Error resolving {ref}: {type(e).__name__}: {e}")

        if not resolved_items:
            print("No resolved items to show.")
            return []

        # Group items by the page they appear on
        by_page = defaultdict(list)
        for item in resolved_items:
            page_no = item.prov[0].page_no
            by_page[page_no].append(item)

        saved_paths = []
        for page_no, items in sorted(by_page.items()):
            img = self.draw_multiple_bboxes(self.doc_json, items)
            path = os.path.join(self.output_dir, f"page_{page_no}.png")
            img.save(path)
            saved_paths.append(path)

        return saved_paths

    def query(self, user_query: str) -> dict:
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
        image_paths = self._generate_images(result=result)

        # Display the visual groundings
        for path in image_paths:
            display(Image.open(path))

        logger.info(f"Answer: {result.model_dump()["value"]}")

        return {
            "extraction": result.model_dump(),
            "image_paths": image_paths,
        }
