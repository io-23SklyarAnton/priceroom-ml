from pydantic import BaseModel, Field, field_validator


class RealtyPredictionBody(BaseModel):
    district: str = Field(..., description="District name")
    rooms_count: int = Field(..., description="Number of rooms")
    total_square_meters: float = Field(..., description="Total area in square meters")
    floor: int = Field(..., description="Floor number")
    floors_count: int = Field(..., description="Total number of floors in building")

    @field_validator("rooms_count")
    def rooms_count_must_be_positive(cls, value):
        if value < 1:
            raise ValueError("Rooms count must be positive.")
        if value > 10:
            raise ValueError("Rooms count must be less than 10.")
        return value

    @field_validator("total_square_meters")
    def total_square_meters_must_be_positive(cls, value):
        if value < 1:
            raise ValueError("Total square meters must be positive.")
        if value > 1000:
            raise ValueError("Total square meters must be less than 1000.")
        return value

    @field_validator("floor")
    def floor_must_be_positive(cls, value):
        if value < 1:
            raise ValueError("Floor must be positive.")
        if value > 100:
            raise ValueError("Floor must be less than 100.")
        return value

    @field_validator("floors_count")
    def floors_count_must_be_positive(cls, value):
        if value < 1:
            raise ValueError("Floors count must be positive.")
        if value > 100:
            raise ValueError("Floors count must be less than 100.")
        return value

    class Config:
        json_schema_extra = {
            "example": {
                "district": "Оболонь",
                "rooms_count": 1,
                "total_square_meters": 35,
                "floor": 1,
                "floors_count": 1
            }
        }
