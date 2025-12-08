from fastapi import APIRouter, Request, Cookie, HTTPException, Depends

from cart.adapter.input.web.request.add_cart_item_request import AddCartItemRequest
from cart.application.usecase.cart_usecase import CartUseCaseImpl
from utility.session_helper import get_current_user

cart_router = APIRouter()
cart_usecase = CartUseCaseImpl()


@cart_router.get("/list")
async def get_cart(user_id: str = Depends(get_current_user)):
    cart = await cart_usecase.get_cart(user_id)
    return cart

@cart_router.post("/add")
async def add_to_cart(request: AddCartItemRequest, user_id: str = Depends(get_current_user)):
    await cart_usecase.add_to_cart(user_id, request)
    return await cart_usecase.get_cart(user_id)
