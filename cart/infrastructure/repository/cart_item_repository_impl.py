from typing import List
from cart.application.port.cart_item_repository_port import CartItemRepositoryPort
from cart.domain.cart_item import CartItem
from cart.domain.value_object.price import Price

from cart.infrastructure.orm.cart_item_orm import CartItemORM
from config.database.session import SessionLocal

class CartItemRepositoryImpl(CartItemRepositoryPort):

    def __init__(self):
        self.db = SessionLocal()

    def save(self, item: CartItem) -> CartItem:
        orm = CartItemORM(
            cart_id=item.cart_id,
            product_id=item.product_id,
            name=item.name,
            quantity=item.quantity,
            price=item.price.value
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)

        item.id = orm.id
        return item

    def update(self, item: CartItem) -> CartItem:
        orm = self.db.get(CartItemORM, item.id)
        if not orm:
            raise ValueError(f"CartItem not found: id={item.id}")

        orm.quantity = item.quantity
        orm.price = item.price.value
        orm.name = item.name

        self.db.commit()
        self.db.refresh(orm)
        return item

    def find_by_cart_id(self, cart_id: int) -> List[CartItem]:
        orms = self.db.query(CartItemORM).filter(CartItemORM.cart_id == cart_id).all()

        items = []
        for o in orms:
            item = CartItem(
                cart_id=o.cart_id,
                product_id=o.product_id,
                name=o.name,
                quantity=o.quantity,
                price=Price(o.price)
            )
            item.id = o.id
            items.append(item)

        return items

    def delete(self, item_id: int):
        orm = self.db.get(CartItemORM, item_id)
        if orm:
            self.db.delete(orm)
            self.db.commit()
