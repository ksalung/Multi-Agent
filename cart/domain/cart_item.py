from datetime import datetime
from typing import Dict, Any
from cart.domain.value_object.price import Price

class CartItem:
    def __init__(self, product_id: str, name: str, price: Price, quantity: int = 1, cart_id: int | None = None):
        if not product_id:
            raise ValueError("product_id required")
        if not name:
            raise ValueError("name required")
        if quantity <= 0:
            raise ValueError("quantity must be >= 1")

        self.cart_id = cart_id
        self.product_id = str(product_id)
        self.name = name
        self.price = price
        self.quantity = quantity
        self.added_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cart_id": self.cart_id,
            "product_id": self.product_id,
            "name": self.name,
            "price": self.price.to_dict(),
            "quantity": self.quantity,
            "added_at": self.added_at.isoformat()
        }
