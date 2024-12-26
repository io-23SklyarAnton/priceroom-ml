from typing import Callable

from sqlalchemy.orm import Session, scoped_session, sessionmaker

from db import Settings
from db.base import Base


def get_db() -> Session:
    session_local: sessionmaker = sessionmaker(
        autoflush=False,
        autocommit=False,
        bind=Settings().engine,
    )

    db: Session = session_local()
    try:
        return db
    finally:
        db.close()


def read_session(func: Callable) -> Callable:
    def inner(*args, **kwargs):

        if any(
                isinstance(arg, Session) for arg in args
        ) or 'session' in kwargs:
            return func(*args, **kwargs)
        else:
            session_factory = sessionmaker(Settings().engine)
            session = scoped_session(session_factory)

            result = func(session, *args, **kwargs)

        session.close()
        return result

    return inner


def write_session(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        try:
            for arg in args:
                if isinstance(arg, Session):
                    session = arg
                    return func(*args, **kwargs)
            if 'session' in kwargs:
                session = kwargs['session']
                return func(*args, **kwargs)
            else:
                session_factory = sessionmaker(Settings().engine)
                session = scoped_session(session_factory)

                result = func(session, *args, **kwargs)

                session.commit()

                if result and not isinstance(result, bool) and not isinstance(result, int) and not isinstance(result,
                                                                                                              list):
                    session.refresh(result)

                session.close()

            return result
        except Exception as e:
            session.rollback()
            session.close()
            raise e

    return inner


def create_tables(engine):
    from db.models import Realty  # noqa

    Base.metadata.create_all(engine)


if __name__ == '__main__':
    settings = Settings()
    engine = settings.engine
    create_tables(engine)
