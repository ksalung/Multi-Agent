from sqlalchemy import Column, Integer, ForeignKey, String, Float
from config.database.session import Base

class CartItemORM(Base):
    __tablename__ = "cart_item"

    id = Column(Integer, primary_key=True, index=True)
    cart_id = Column(Integer, ForeignKey("cart.id"), nullable=False)
    product_id = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
