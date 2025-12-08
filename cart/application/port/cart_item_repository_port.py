from abc import ABC, abstractmethod
from typing import List

from cart.domain.cart_item import CartItem


class CartItemRepositoryPort(ABC):

    @abstractmethod
    def save(self, item: CartItem) -> CartItem:
        pass

    @abstractmethod
    def find_by_cart_id(self, cart_id: int) -> List[CartItem]:
        pass

    @abstractmethod
    def delete(self, item_id: int):
        pass
