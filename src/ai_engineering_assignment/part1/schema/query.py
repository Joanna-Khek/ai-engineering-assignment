from pydantic import BaseModel, Field

class ExtractedField(BaseModel):
    field_name: str = Field(description="The extracted field name")
    value: str | float | None = Field(description="The extracted field value")
    self_refs: list[str] = Field(description="The reference where the information was found")
    confidence_note: str = Field(description="Description of the extraction confidence")