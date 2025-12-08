from abc import ABC, abstractmethod

from cart.domain.cart import Cart


class CartRepositoryPort(ABC):

    @abstractmethod
    def save(self, cart: Cart) -> Cart:
        pass

    @abstractmethod
    def find_by_account_id(self, account_id: int) -> Cart | None:
        pass
