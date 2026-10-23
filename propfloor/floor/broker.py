"""Broker adapter interface. Only PaperBroker is functional; live adapters are stubs for later."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Order:
    account_id: str
    symbol: str
    side: int
    lots: float
    stop_price: float | None = None
    take_profit: float | None = None
    client_id: str = ""


@dataclass
class Fill:
    order: Order
    price: float
    time: pd.Timestamp
    broker_id: str


class BrokerAdapter(ABC):
    @abstractmethod
    def place(self, order: Order, price: float, time: pd.Timestamp) -> Fill: ...

    @abstractmethod
    def close(self, broker_id: str, price: float, time: pd.Timestamp) -> Fill: ...

    @abstractmethod
    def positions(self, account_id: str) -> list[Fill]: ...


@dataclass
class PaperBroker(BrokerAdapter):
    fills: list[Fill] = field(default_factory=list)
    open: dict[str, Fill] = field(default_factory=dict)
    _n: int = 0

    def place(self, order, price, time):
        self._n += 1
        f = Fill(order, price, time, f"paper-{self._n}")
        self.fills.append(f)
        self.open[f.broker_id] = f
        return f

    def close(self, broker_id, price, time):
        o = self.open.pop(broker_id)
        f = Fill(Order(o.order.account_id, o.order.symbol, -o.order.side, o.order.lots), price, time, broker_id)
        self.fills.append(f)
        return f

    def positions(self, account_id):
        return [f for f in self.open.values() if f.order.account_id == account_id]


class MT5Broker(BrokerAdapter):  # pragma: no cover - stub
    """Future: MetaTrader5 python package (order_send / positions_get)."""

    def __init__(self, *_, **__):
        raise NotImplementedError("MT5 adapter not implemented yet - paper trading only")

    place = close = positions = None  # type: ignore[assignment]


class CTraderBroker(BrokerAdapter):  # pragma: no cover - stub
    """Future: cTrader Open API (protobuf over TCP/websocket)."""

    def __init__(self, *_, **__):
        raise NotImplementedError("cTrader adapter not implemented yet - paper trading only")

    place = close = positions = None  # type: ignore[assignment]
