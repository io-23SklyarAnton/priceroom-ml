from sqlalchemy import (
    Column, Integer, String, Float, Boolean
)
from db.base import Base


class Realty(Base):
    __tablename__ = "realty"

    id = Column(Integer, primary_key=True, autoincrement=True)
    realty_id = Column(Integer, nullable=False, unique=True)
    city_name_uk = Column(String, nullable=True)  # city name
    floor = Column(Integer, nullable=True)  # floor
    floors_count = Column(Integer, nullable=True)  # number of floors
    realty_sale_type = Column(Integer, nullable=True)  # type of selling
    price = Column(Float, nullable=True)  # price
    street_name_uk = Column(String, nullable=True)  # street name
    realty_type_parent_name_uk = Column(String, nullable=True)
    beautiful_url = Column(String, nullable=True)
    advert_type_name_uk = Column(String, nullable=True)
    district_name_uk = Column(String, nullable=True)  # district name
    district_id = Column(Integer, nullable=True)  # district id
    is_commercial = Column(Boolean, nullable=True)  # is commercial
    currency_type_uk = Column(String, nullable=True)  # currency type
    city_id = Column(Integer, nullable=True)
    street_id = Column(Integer, nullable=True)
    district_type_id = Column(Integer, nullable=True)
    total_square_meters = Column(Float, nullable=True)  # total square meters
    realty_type_name_uk = Column(String, nullable=True)  # realty type name
    rooms_count = Column(Integer, nullable=True)  # number of rooms
    is_developer = Column(Boolean, nullable=True)  # is new building
    type = Column(String, nullable=True)
