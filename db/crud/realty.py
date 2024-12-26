from sqlalchemy.orm import Session

from db.models import Realty
from db.utils import read_session


@read_session
def get_realties(session: Session):
    return session.query(
            Realty
        ).filter(
            Realty.city_id.isnot(None),
            Realty.floor.isnot(None),
            Realty.floors_count.isnot(None),
            Realty.street_id.isnot(None),
            Realty.district_id.isnot(None),
            Realty.is_commercial.isnot(None),
            Realty.rooms_count.isnot(None),
            Realty.total_square_meters.isnot(None),
            Realty.price.isnot(None),
            Realty.currency_type_uk == "$",
            Realty.realty_sale_type == 1,
        ).all()
