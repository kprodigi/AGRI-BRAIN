from pydantic import BaseModel, Field
class Policy(BaseModel):
    max_temp_c: float = Field(8.0)
    min_shelf_reroute: float = Field(0.70)
    min_shelf_expedite: float = Field(0.50)
    carbon_per_km: float = Field(0.12)
    km_farm_to_dc: float = 280
    km_dc_to_retail: float = 50
    km_expedited: float = 160
    msrp: float = 1.50
