"""
    Schema for the car price prediction request.
    Field names use underscores (Python convention); adjust the
    ColumnTransformer input step if you need to remap to the
    original hyphenated column names.
    """
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    # --- numeric features ---
    symboling: float = Field(..., description="Insurance risk rating")
    normalized_losses: float = Field(..., description="Normalized loss value")
    wheel_base: float = Field(..., description="Wheel base length")
    length: float = Field(..., description="Car length")
    width: float = Field(..., description="Car width")
    height: float = Field(..., description="Car height")
    curb_weight: float = Field(..., description="Curb weight")
    engine_size: float = Field(..., description="Engine size")
    bore: float = Field(..., description="Bore")
    stroke: float = Field(..., description="Stroke")
    compression_ratio: float = Field(..., description="Compression ratio")
    horsepower: float = Field(..., description="Horsepower")
    peak_rpm: float = Field(..., description="Peak RPM")
    city_L_100km: float = Field(...,
                                description="City fuel consumption (L/100km)")
    highway_mpg: float = Field(..., description="Highway MPG")
    num_of_doors: float = Field(..., description="Number of doors")
    num_of_cylinders: float = Field(..., description="Number of cylinders")

    # --- categorical features ---
    make: str = Field(..., description="Manufacturer")
    fuel_type: str = Field(..., description="Fuel type")
    aspiration: str = Field(..., description="Aspiration type")
    body_style: str = Field(..., description="Body style")
    drive_wheels: str = Field(..., description="Drive wheels")
    engine_location: str = Field(..., description="Engine location")
    engine_type: str = Field(..., description="Engine type")
    fuel_system: str = Field(..., description="Fuel system")
