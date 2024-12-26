import os

from sqlalchemy import Engine, create_engine
from dotenv import load_dotenv

load_dotenv()


class Settings:

    def __init__(self) -> None:
        self.ENGINE = os.getenv('DB_ENGINE')
        self.HOST = os.getenv('DB_HOST')
        self.PORT = os.getenv('DB_PORT')
        self.USER = os.getenv('DB_USER')
        self.PASSWORD = os.getenv('DB_PASSWORD')
        self.DATABASE = os.getenv('DB_NAME')

    @property
    def connection_string(self) -> str:
        return f"{self.ENGINE}://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}"

    @property
    def engine(self) -> Engine:
        return create_engine(self.connection_string, pool_size=100, max_overflow=100)


if __name__ == '__main__':
    settings = Settings()
    print(settings.connection_string)
    print(settings.engine)
