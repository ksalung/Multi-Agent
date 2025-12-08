# domain/cart/value_object/price.py
class Price:
    def __init__(self, value: float, currency: str = "KRW"):
        if value is None:
            raise ValueError("Price value cannot be None")
        try:
            v = float(value)
        except Exception:
            raise ValueError("Price value must be numeric")
        if v < 0:
            raise ValueError("Price cannot be negative")
        self.value = v
        self.currency = currency

    def to_dict(self):
        return {"value": self.value, "currency": self.currency}

    def __repr__(self):
        return f"{self.value} {self.currency}"
