from pydantic import BaseModel


class AddCartItemRequest(BaseModel):
    product_id: str
    name: str
    price: float
    quantity: int = 1
