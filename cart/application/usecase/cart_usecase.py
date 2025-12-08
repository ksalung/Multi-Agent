from cart.adapter.input.web.request.add_cart_item_request import AddCartItemRequest
from cart.domain.cart import Cart
from cart.domain.cart_item import CartItem
from cart.domain.value_object.price import Price
from cart.infrastructure.repository.cart_repository_impl import CartRepositoryImpl
from cart.infrastructure.repository.cart_item_repository_impl import CartItemRepositoryImpl

class CartUseCaseImpl:
    def __init__(self):
        self.cart_repo = CartRepositoryImpl()
        self.cart_item_repo = CartItemRepositoryImpl()

    async def get_cart(self, user_id: int) -> Cart:
        cart_orm = self.cart_repo.find_by_account_id(user_id)

        if cart_orm is None:
            cart = Cart(user_id=user_id)
            self.cart_repo.save(cart)
            return cart

        items = self.cart_item_repo.find_by_cart_id(cart_orm.id)

        cart = Cart(
            id=cart_orm.id,
            user_id=cart_orm.user_id,
            created_at=cart_orm.created_at,
            updated_at=cart_orm.updated_at,
            items=items
        )

        return cart

    async def add_to_cart(self, user_id: int, request: AddCartItemRequest):
        cart = await self.get_cart(user_id)

        exist_item = None
        for item in cart.items:
            if item.product_id == request.product_id:
                exist_item = item
                break

        if exist_item:
            exist_item.quantity += request.quantity
            self.cart_item_repo.update(exist_item)
            return cart

        new_item = CartItem(
            cart_id=cart.id,
            product_id=request.product_id,
            name=request.name,
            price=Price(float(request.price)),
            quantity=request.quantity if hasattr(request, "quantity") else 1,
        )

        self.cart_item_repo.save(new_item)
        cart.add_item(new_item)

        return cart
