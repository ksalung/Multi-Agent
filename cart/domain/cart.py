from datetime import datetime
from typing import List, Optional
from cart.domain.cart_item import CartItem

class Cart:
    def __init__(
        self,
        user_id: int,
        id: Optional[int] = None,
        items: Optional[List[CartItem]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.id = id
        self.user_id = user_id
        self.items = items or []
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def add_item(self, item: CartItem):
        for existing in self.items:
            if existing.product_id == item.product_id:
                existing.quantity += item.quantity
                self.updated_at = datetime.utcnow()
                return
        self.items.append(item)
        self.updated_at = datetime.utcnow()

    def remove_item(self, product_id: str):
        self.items = [i for i in self.items if i.product_id != product_id]
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "items": [item.to_dict() for item in self.items]
        }
