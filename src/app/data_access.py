from __future__ import annotations

import copy
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


class ShoppingDataStore:
    """In-memory lookup store for the shopping mock dataset."""

    def __init__(self, json_path: Path) -> None:
        self.json_path = json_path
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.metadata: dict[str, Any] = payload.get("metadata", {})
        self.customers: list[dict[str, Any]] = payload.get("customers", [])
        self.orders: list[dict[str, Any]] = payload.get("orders", [])
        self.vouchers: list[dict[str, Any]] = payload.get("vouchers", [])

        self.customer_by_id = {
            str(customer["customer_id"]): customer
            for customer in self.customers
            if customer.get("customer_id") is not None
        }
        self.order_by_id = {
            str(order["order_id"]): order
            for order in self.orders
            if order.get("order_id") is not None
        }

        orders_by_customer_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for order in self.orders:
            customer_id = order.get("customer_id")
            if customer_id is not None:
                orders_by_customer_id[str(customer_id)].append(order)
        self.orders_by_customer_id = dict(orders_by_customer_id)
        for customer_orders in self.orders_by_customer_id.values():
            customer_orders.sort(key=_order_sort_key, reverse=True)

        vouchers_by_customer_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for voucher in self.vouchers:
            customer_id = voucher.get("customer_id")
            if customer_id is not None:
                vouchers_by_customer_id[str(customer_id)].append(voucher)
        self.vouchers_by_customer_id = dict(vouchers_by_customer_id)

        self.today = _parse_date(self.metadata.get("today")) or date.today()

    def get_customer_by_id(self, customer_id: str) -> dict[str, Any]:
        customer_id = _normalize_id(customer_id)
        customer = self.customer_by_id.get(customer_id)
        if customer is None:
            return {"status": "not_found", "customer_id": customer_id}
        return {"status": "ok", "customer": copy.deepcopy(customer)}

    def get_orders_by_customer_id(self, customer_id: str, limit: int = 10) -> dict[str, Any]:
        customer_id = _normalize_id(customer_id)
        if customer_id not in self.customer_by_id:
            return {"status": "not_found", "customer_id": customer_id}

        safe_limit = _normalize_limit(limit)
        orders = self.orders_by_customer_id.get(customer_id, [])[:safe_limit]
        return {
            "status": "ok",
            "customer_id": customer_id,
            "count": len(orders),
            "limit": safe_limit,
            "orders": copy.deepcopy(orders),
        }

    def get_order_detail_by_order_id(self, order_id: str) -> dict[str, Any]:
        order_id = _normalize_id(order_id)
        order = self.order_by_id.get(order_id)
        if order is None:
            return {"status": "not_found", "order_id": order_id}
        return {"status": "ok", "order": copy.deepcopy(order)}

    def get_vouchers_by_customer_id(
        self,
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        customer_id = _normalize_id(customer_id)
        if customer_id not in self.customer_by_id:
            return {"status": "not_found", "customer_id": customer_id}

        vouchers = self.vouchers_by_customer_id.get(customer_id, [])
        if only_active:
            vouchers = [
                voucher
                for voucher in vouchers
                if _is_usable_voucher(voucher, today=self.today)
            ]

        return {
            "status": "ok",
            "customer_id": customer_id,
            "only_active": only_active,
            "count": len(vouchers),
            "vouchers": copy.deepcopy(vouchers),
        }


def build_data_tools(store: ShoppingDataStore) -> list:
    @tool
    def get_customer_by_id(customer_id: str) -> dict[str, Any]:
        """Tra cứu thông tin khách hàng theo customer_id, ví dụ C001."""
        return store.get_customer_by_id(customer_id)

    @tool
    def get_orders_by_customer_id(
        customer_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Lấy danh sách đơn hàng gần nhất của một khách hàng theo customer_id."""
        return store.get_orders_by_customer_id(customer_id, limit=limit)

    @tool
    def get_order_detail_by_order_id(order_id: str) -> dict[str, Any]:
        """Tra cứu chi tiết một đơn hàng theo order_id, ví dụ 1971 hoặc 2058."""
        return store.get_order_detail_by_order_id(order_id)

    @tool
    def get_vouchers_by_customer_id(
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        """Lấy voucher của khách hàng; đặt only_active=True để chỉ lấy voucher còn dùng được."""
        return store.get_vouchers_by_customer_id(
            customer_id,
            only_active=only_active,
        )

    return [
        get_customer_by_id,
        get_orders_by_customer_id,
        get_order_detail_by_order_id,
        get_vouchers_by_customer_id,
    ]


def _normalize_id(value: str) -> str:
    return str(value).strip()


def _normalize_limit(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 10


def _order_sort_key(order: dict[str, Any]) -> tuple[datetime, str]:
    created_at = _parse_datetime(order.get("created_at")) or datetime.min
    return (created_at, str(order.get("order_id", "")))


def _is_usable_voucher(voucher: dict[str, Any], today: date) -> bool:
    if voucher.get("status") not in {"active", "restored"}:
        return False
    if int(voucher.get("remaining_uses") or 0) <= 0:
        return False

    start_at = _parse_date(voucher.get("start_at"))
    end_at = _parse_date(voucher.get("end_at"))
    if start_at is not None and today < start_at:
        return False
    if end_at is not None and today > end_at:
        return False
    return True


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _parse_date(value: Any) -> date | None:
    parsed_datetime = _parse_datetime(value)
    if parsed_datetime is not None:
        return parsed_datetime.date()

    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None
