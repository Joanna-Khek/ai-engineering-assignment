import os
from pathlib import Path
from loguru import logger

from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)
from docling.datamodel.chart_extraction_options import ChartExtractionModelOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_core.types.doc.document import DoclingDocument

from ai_engineering_assignment.settings import MainConfig


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

    def _set_up_chart_description_config(self):
        """
        The document contains pie chart, bar chart and line chart. We use this configuration
        to turn the charts into text descriptions.
        """
        self.chart_description_options = ChartExtractionModelOptions(chart2summary=True)

    def _set_up_table_config(self):
        # Since we have many tables, we should go for the accurate option.
        # Since document is a digital PDF, we can set cell matching to true.
        self.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        )

    def _set_up_docling_configs(self):
        """Set up the necessary Docling DocumentConverter configs"""
        # 1. Accelerator Configs
        self._set_up_accelerator_config(
            use_flash_attention=self.configs.app.docling.use_flash_attention,
            device=self.configs.app.docling.device,
        )

        # 2. Picture Description Configs
        # self._set_up_picture_description_config(
        #     model_name=self.configs.app.vlm.model,
        #     scale=self.configs.app.vlm.scale,
        #     prompt=prompts.PICTURE_DESCRIPTION_PROMPT,
        #     picture_threshold=self.configs.app.vlm.picture_area_threshold,
        # )

        # 3. Chart Description Configs
        self._set_up_chart_description_config()

        # 4. Table Configs
        self._set_up_table_config()

        # PDF Pipeline
        self.pipeline_options = PdfPipelineOptions(
            accelerator_options=self.accelerator_options,
            do_chart_extraction=True,
            chart_extraction_options=self.chart_description_options,
            do_table_structure=True,
            table_structure_options=self.table_structure_options,
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=2.0,
        )
        logger.debug(f"Pipeine: {self.pipeline_options}")

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
