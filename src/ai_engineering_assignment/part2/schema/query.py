from typing import Literal
from datetime import date
from pydantic import BaseModel, Field, model_validator

REFERENCE_DATE = date(2024, 1, 1)


class DateResult(BaseModel):
    original_text: str = Field(
        description="The original answer to the question in full text"
    )
    normalized_date: str = Field(description="The normalized date in YYYY-MM-DD format")
    status: Literal["Upcoming", "Expired", "Ongoing"]
    self_refs: list[str] = Field(
        description="The reference where the information was found"
    )

    @model_validator(mode="after")
    def compute_status(self) -> "DateResult":
        parsed = date.fromisoformat(self.normalized_date)
        if parsed > REFERENCE_DATE:
            self.status = "Upcoming"
        elif parsed < REFERENCE_DATE:
            self.status = "Expired"
        else:
            self.status = "Ongoing"

        return self
