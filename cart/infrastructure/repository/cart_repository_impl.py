from cart.application.port.cart_repository_port import CartRepositoryPort
from cart.domain.cart import Cart

from cart.infrastructure.orm.cart_orm import CartORM
from config.database.session import SessionLocal

class CartRepositoryImpl(CartRepositoryPort):

    def __init__(self):
        self.db = SessionLocal()

    def save(self, cart: Cart) -> Cart:
        if cart.id is None:
            orm = CartORM(account_id=cart.user_id)
            self.db.add(orm)
            self.db.commit()
            self.db.refresh(orm)

            cart.id = orm.id
            cart.created_at = orm.created_at
            cart.updated_at = orm.updated_at
        return cart

    def find_by_account_id(self, account_id: int) -> Cart | None:
        orm = self.db.query(CartORM).filter(CartORM.account_id == account_id).first()
        if orm is None:
            return None

        cart = Cart(orm.account_id)
        cart.id = orm.id
        cart.created_at = orm.created_at
        cart.updated_at = orm.updated_at
        return cart
