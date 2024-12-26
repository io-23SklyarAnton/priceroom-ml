from pydantic import BaseModel, field_validator


class RealtyPredictionBody(BaseModel):
    district: str
    rooms_count: int
    total_square_meters: float
    floor: int
    floors_count: int

    @field_validator("rooms_count")
    def rooms_count_must_be_positive(cls, value):
        if value < 1:
            raise ValueError("Rooms count must be positive.")
        if value > 10:
            raise ValueError("Rooms count must be less than 10.")

    @field_validator("total_square_meters")
    def total_square_meters_must_be_positive(cls, value):
        if value < 1:
            raise ValueError("Total square meters must be positive.")
        if value > 1000:
            raise ValueError("Total square meters must be less than 1000.")

    @field_validator("floor")
    def floor_must_be_positive(cls, value):
        if value < 1:
            raise ValueError("Floor must be positive.")
        if value > 100:
            raise ValueError("Floor must be less than 100.")

    @field_validator("floors_count")
    def floors_count_must_be_positive(cls, value):
        if value < 1:
            raise ValueError("Floors count must be positive.")
        if value > 100:
            raise ValueError("Floors count must be less than 100.")


class RealtyPredictionResponse(BaseModel):
    price: float
