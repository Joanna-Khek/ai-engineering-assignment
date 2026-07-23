import os
from pathlib import Path
from loguru import logger

from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_core.types.doc.document import DoclingDocument

from ai_engineering_assignment.settings import MainConfig
from ai_engineering_assignment.part1 import prompts


class ExtractDocument:
    def __init__(self, source: str):
        self.source = source
        self.configs = MainConfig()

        self.output_dir = self.configs.ROOT / "data"
        os.makedirs(self.output_dir, exist_ok=True)

    def _set_up_accelerator_config(
        self, use_flash_attention: bool, device: str
    ) -> None:
        device = AcceleratorDevice.CUDA if device == "cuda" else AcceleratorDevice.CPU

        self.accelerator_options = AcceleratorOptions(
            device=device, cuda_use_flash_attention2=use_flash_attention
        )

    def _set_up_picture_description_config(
        self, model_name: str, scale: float, prompt: str, picture_threshold: float
    ) -> None:
        """
        Since the document has charts, we want to set up the picture description option in docling
        to turn the charts into text descriptions

        Args:
            model_name (str): The vision language model to be used to generate the description
            scale (float): The scale of the image to be sent to the vision language model
            prompt (str): The prompt to be sent to the vision language model to generate the description
            picture_threshold (float): Minimum picture area as fraction of page area (0.0-1.0) to trigger description. Pictures smaller than this threshold are skipped.
        """

        self.picture_description_options = PictureDescriptionVlmOptions(
            repo_id=model_name,
            prompt=prompt,
            scale=scale,
            picture_area_threshold=picture_threshold,
        )

    def _set_up_docling_configs(self):
        """Set up the necessary Docling DocumentConverter configs"""
        # 1. Accelerator Configs
        self._set_up_accelerator_config(
            use_flash_attention=self.configs.app.docling.use_flash_attention,
            device=self.configs.app.docling.device,
        )

        # 2. Picture Description Configs (for the charts)
        self._set_up_picture_description_config(
            model_name=self.configs.app.vlm.model,
            scale=self.configs.app.vlm.scale,
            prompt=prompts.PICTURE_DESCRIPTION_PROMPT,
            picture_threshold=self.configs.app.vlm.picture_area_threshold,
        )

        # PDF Pipeline
        self.pipeline_options = PdfPipelineOptions(
            accelerator_options=self.accelerator_options,
            do_picture_description=True,
            picture_description_options=self.picture_description_options,
            do_table_structure=True,
            generate_page_images=True,
            images_scale=1.0,
        )

        # Since we have many tables in the document
        self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    def _save_result(self, doc: DoclingDocument, output_dir: Path):
        """Save the extracted content into a json file locally"""
        filename = f"{doc.origin.binary_hash}.json"
        file_path = output_dir / filename
        doc.save_as_json(filename=file_path)
        logger.info(f"Output saved to {file_path}.")

    def extract_document(self) -> None:
        """Extract the contents of the document

        Args:
            output_folder (Path): The path in which the output json document is to be saved locally.

        """

        # Set up necessary configs
        self._set_up_docling_configs()

        # Extract contents
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        result = converter.convert(source=self.source)

        # Save to json
        self._save_result(doc=result.document, output_dir=self.output_dir)
        logger.info("Successfully extracted document. ")
