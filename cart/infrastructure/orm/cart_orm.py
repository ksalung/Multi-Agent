from datetime import datetime

from sqlalchemy import Column, Integer, DateTime

from config.database.session import Base


class CartORM(Base):
    __tablename__ = "cart"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)