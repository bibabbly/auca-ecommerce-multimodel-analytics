"""
generate_data.py - Synthetic e-commerce dataset generator (AUCA Big Data Final Project)

Based on the course-provided `dataset_generator.py` (a Colab notebook). The data model,
page-flow probabilities, conversion logic and value ranges are kept identical. Changes
made so the generator is practical for a big-data pipeline:

  1. Configurable scale via CLI presets (small / medium / full) or explicit counts.
     `full` reproduces the original targets (10k users, 5k products, 2M sessions, 500k txns).
  2. Streaming output: sessions and transactions are written to disk as they are
     generated instead of being held in memory (2M sessions ~ several GB of RAM).
  3. JSON Lines output (one JSON object per line). Spark, mongoimport and most
     loaders can split JSONL across workers; a single huge JSON array cannot be split.
     Use --format array to get the original JSON-array format instead.
  4. Reproducibility: a fixed reference end date (--end-date) replaces datetime.now(),
     and IDs are derived from the seeded RNG instead of uuid4, so the same seed always
     yields the same dataset.
  5. Products carry a `subcategory_id` (the brief's product sample and relationship
     section expect it; the original generator never assigned one).
  6. Inventory consistency fix: a converted cart is only charged when *every* line item
     is in stock (the original could decrement stock for some items and then abort).
     A session is marked "converted" only if a transaction was actually created, so
     sessions.conversion_status and transactions.session_id always agree.

Usage:
    python src/generate_data.py --preset medium
    python src/generate_data.py --preset full --out data/raw_full
    python src/generate_data.py --users 5000 --sessions 100000 --transactions 20000
"""

import argparse
import datetime as dt
import json
import os
import random
import time

import numpy as np
from faker import Faker

PRESETS = {
    #            users  products categories sessions  transactions
    "small":  dict(users=2_000, products=1_000, categories=25, sessions=20_000, transactions=5_000),
    "medium": dict(users=10_000, products=5_000, categories=25, sessions=300_000, transactions=75_000),
    "full":   dict(users=10_000, products=5_000, categories=25, sessions=2_000_000, transactions=500_000),
}
TIMESPAN_DAYS = 90
SESSION_CHUNK_SIZE = 100_000  # sessions per output file, as in the original


# --------------------------------------------------------------------------- helpers
class JsonWriter:
    """Writes records either as JSON Lines or as a single JSON array."""

    def __init__(self, path, fmt):
        self.fmt = fmt
        self.f = open(path, "w", encoding="utf-8")
        self.count = 0
        if fmt == "array":
            self.f.write("[")

    def write(self, obj):
        if self.fmt == "array":
            if self.count:
                self.f.write(",")
            self.f.write(json.dumps(obj))
        else:
            self.f.write(json.dumps(obj) + "\n")
        self.count += 1

    def close(self):
        if self.fmt == "array":
            self.f.write("]")
        self.f.close()


class InventoryManager:
    def __init__(self, products):
        self.products = {p["product_id"]: p for p in products}

    def stock(self, product_id):
        p = self.products.get(product_id)
        return p["current_stock"] if p else 0

    def try_purchase(self, items):
        """Atomically decrement stock for {product_id: qty}; all-or-nothing."""
        if any(self.stock(pid) < qty for pid, qty in items.items()):
            return False
        for pid, qty in items.items():
            self.products[pid]["current_stock"] -= qty
        return True


def rand_between(rng, start, end):
    """Uniform random datetime in [start, end]."""
    span = int((end - start).total_seconds())
    return start + dt.timedelta(seconds=rng.randint(0, max(span, 0)))  # whole seconds, like the brief's samples


def rand_hex(rng, n):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def determine_page_type(position, previous_pages):
    """Page-flow model copied unchanged from the original generator."""
    if position == 0:
        return random.choice(["home", "search", "category_listing"])
    if not previous_pages:
        return "home"
    prev = previous_pages[-1]["page_type"]
    flows = {
        "home": (["category_listing", "search", "product_detail"], [0.5, 0.3, 0.2]),
        "category_listing": (["product_detail", "category_listing", "search", "home"], [0.7, 0.1, 0.1, 0.1]),
        "search": (["product_detail", "search", "category_listing", "home"], [0.6, 0.2, 0.1, 0.1]),
        "product_detail": (["product_detail", "cart", "category_listing", "search", "home"], [0.3, 0.3, 0.2, 0.1, 0.1]),
        "cart": (["checkout", "product_detail", "category_listing", "home"], [0.6, 0.2, 0.1, 0.1]),
        "checkout": (["confirmation", "cart", "home"], [0.8, 0.1, 0.1]),
        "confirmation": (["home", "product_detail", "category_listing"], [0.6, 0.2, 0.2]),
    }
    if prev not in flows:
        return "home"
    choices, weights = flows[prev]
    return random.choices(choices, weights=weights)[0]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="medium")
    for k in ("users", "products", "categories", "sessions", "transactions"):
        ap.add_argument(f"--{k}", type=int, help=f"override preset {k} count")
    ap.add_argument("--end-date", default="2026-10-31", help="last day of the 90-day activity window (YYYY-MM-DD)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--format", choices=["jsonl", "array"], default="jsonl")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    for k in cfg:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    n_users, n_products, n_categories = cfg["users"], cfg["products"], cfg["categories"]
    n_sessions, n_transactions = cfg["sessions"], cfg["transactions"]
    max_iterations = (n_sessions + n_transactions) * 2  # fail-safe, as in the original

    random.seed(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)
    fake = Faker()
    rng = random.Random(args.seed + 1)  # separate stream for IDs/dates

    now = dt.datetime.fromisoformat(args.end_date) + dt.timedelta(hours=23, minutes=59, seconds=59)
    window_start = now - dt.timedelta(days=TIMESPAN_DAYS)
    os.makedirs(args.out, exist_ok=True)
    ext = "jsonl" if args.format == "jsonl" else "json"
    t0 = time.time()
    print(f"Config: {cfg} | window {window_start.date()} -> {now.date()} | out={args.out} ({args.format})")

    # ---- categories
    categories = []
    for c in range(n_categories):
        categories.append({
            "category_id": f"cat_{c:03d}",
            "name": fake.company(),
            "subcategories": [
                {"subcategory_id": f"sub_{c:03d}_{s:02d}", "name": fake.bs(),
                 "profit_margin": round(random.uniform(0.1, 0.4), 2)}
                for s in range(random.randint(3, 5))
            ],
        })
    category_by_id = {c["category_id"]: c for c in categories}
    print(f"Generated {len(categories)} categories")

    # ---- products
    products = []
    product_creation_start = now - dt.timedelta(days=TIMESPAN_DAYS * 2)
    for p in range(n_products):
        category = random.choice(categories)
        subcategory = random.choice(category["subcategories"])
        base_price = round(random.uniform(5, 500), 2)
        initial_date = rand_between(rng, product_creation_start,
                                    product_creation_start + dt.timedelta(days=TIMESPAN_DAYS // 3))
        price_history = [{"price": base_price, "date": initial_date.isoformat()}]
        last = initial_date
        for _ in range(random.randint(0, 2)):
            change = rand_between(rng, last, now)
            price_history.append({"price": round(base_price * random.uniform(0.8, 1.2), 2),
                                  "date": change.isoformat()})
            last = change
        price_history.sort(key=lambda x: x["date"])
        products.append({
            "product_id": f"prod_{p:05d}",
            "name": fake.catch_phrase().title(),
            "category_id": category["category_id"],
            "subcategory_id": subcategory["subcategory_id"],
            "base_price": price_history[-1]["price"],
            "current_stock": random.randint(10, 1000),
            "is_active": random.choices([True, False], weights=[0.95, 0.05])[0],
            "price_history": price_history,
            "creation_date": price_history[0]["date"],
        })
    active_in_stock = [p for p in products if p["is_active"]]
    print(f"Generated {len(products)} products")

    # ---- users
    users = []
    for u in range(n_users):
        reg = rand_between(rng, now - dt.timedelta(days=TIMESPAN_DAYS * 3), now - dt.timedelta(days=TIMESPAN_DAYS))
        users.append({
            "user_id": f"user_{u:06d}",
            "geo_data": {"city": fake.city(), "state": fake.state_abbr(), "country": fake.country_code()},
            "registration_date": reg.isoformat(),
            "last_active": rand_between(rng, reg, now).isoformat(),
        })
    print(f"Generated {len(users)} users")

    # ---- sessions & transactions (streamed)
    inventory = InventoryManager(products)
    txn_writer = JsonWriter(os.path.join(args.out, f"transactions.{ext}"), args.format)
    sess_writer, sess_file_idx = None, 0
    session_counter = transaction_counter = iteration = 0

    def write_session(s):
        nonlocal sess_writer, sess_file_idx
        if sess_writer is None or sess_writer.count >= SESSION_CHUNK_SIZE:
            if sess_writer:
                sess_writer.close()
                sess_file_idx += 1
            sess_writer = JsonWriter(os.path.join(args.out, f"sessions_{sess_file_idx}.{ext}"), args.format)
        sess_writer.write(s)

    def make_txn(user_id, session_id, timestamp, items, methods, statuses):
        subtotal = round(sum(i["subtotal"] for i in items), 2)
        discount = 0.0
        if random.random() < 0.2:
            discount = round(subtotal * random.choice([0.05, 0.1, 0.15, 0.2]), 2)
        return {
            "transaction_id": f"txn_{rand_hex(rng, 12)}",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "items": items,
            "subtotal": subtotal,
            "discount": discount,
            "total": round(subtotal - discount, 2),
            "payment_method": random.choice(methods),
            "status": random.choice(statuses),
        }

    print("Generating sessions and transactions...")
    while (session_counter < n_sessions or transaction_counter < n_transactions) and iteration < max_iterations:
        iteration += 1

        if session_counter < n_sessions:
            user = random.choice(users)
            session_id = f"sess_{rand_hex(rng, 10)}"
            session_start = rand_between(rng, window_start, now)
            duration = random.randint(30, 3600)
            page_views, viewed, cart = [], [], {}
            slots = sorted([0] + [random.randint(1, duration - 1) for _ in range(random.randint(3, 15))] + [duration])

            for i in range(len(slots) - 1):
                page_type = determine_page_type(i, page_views)
                product = category = None
                if page_type == "product_detail":
                    product = random.choice(active_in_stock) if active_in_stock else random.choice(products)
                    category = category_by_id[product["category_id"]]
                    pid = product["product_id"]
                    if pid not in viewed:
                        viewed.append(pid)  # keeps chronological order of first view
                    if random.random() < 0.3:
                        line = cart.setdefault(pid, {"quantity": 0, "price": product["base_price"]})
                        max_possible = min(3, inventory.stock(pid) - line["quantity"])
                        if max_possible > 0:
                            line["quantity"] += random.randint(1, max_possible)
                elif page_type == "category_listing":
                    category = random.choice(categories)
                page_views.append({
                    "timestamp": (session_start + dt.timedelta(seconds=slots[i])).isoformat(),
                    "page_type": page_type,
                    "product_id": product["product_id"] if product else None,
                    "category_id": category["category_id"] if category else None,
                    "view_duration": slots[i + 1] - slots[i],
                })

            cart = {k: v for k, v in cart.items() if v["quantity"] > 0}
            end_ts = (session_start + dt.timedelta(seconds=duration)).isoformat()
            converted = False
            if cart and any(p["page_type"] in ("checkout", "confirmation") for p in page_views):
                if random.random() < 0.7 and transaction_counter < n_transactions:
                    if inventory.try_purchase({pid: v["quantity"] for pid, v in cart.items()}):
                        items = [{"product_id": pid, "quantity": v["quantity"], "unit_price": v["price"],
                                  "subtotal": round(v["quantity"] * v["price"], 2)} for pid, v in cart.items()]
                        txn_writer.write(make_txn(user["user_id"], session_id, end_ts, items,
                                                  ["credit_card", "paypal", "apple_pay", "crypto"], ["completed"]))
                        transaction_counter += 1
                        converted = True

            geo = dict(user["geo_data"], ip_address=fake.ipv4())
            write_session({
                "session_id": session_id,
                "user_id": user["user_id"],
                "start_time": session_start.isoformat(),
                "end_time": end_ts,
                "duration_seconds": duration,
                "geo_data": geo,
                "device_profile": {
                    "type": random.choice(["mobile", "desktop", "tablet"]),
                    "os": random.choice(["iOS", "Android", "Windows", "macOS"]),
                    "browser": random.choice(["Chrome", "Safari", "Firefox", "Edge"]),
                },
                "viewed_products": viewed,
                "page_views": page_views,
                "cart_contents": cart,
                "conversion_status": "converted" if converted else "abandoned" if cart else "browsed",
                "referrer": random.choice(["direct", "email", "social", "search_engine", "affiliate"]),
            })
            session_counter += 1

        # Additional transactions not linked to a session (e.g. app / phone orders), as in the original
        if transaction_counter < n_transactions and random.random() < 0.2:
            user = random.choice(users)
            items = []
            for product in random.sample(products, k=min(3, len(products))):
                if product["is_active"]:
                    qty = random.randint(1, 3)
                    if inventory.try_purchase({product["product_id"]: qty}):
                        items.append({"product_id": product["product_id"], "quantity": qty,
                                      "unit_price": product["base_price"],
                                      "subtotal": round(qty * product["base_price"], 2)})
            if items:
                txn_writer.write(make_txn(user["user_id"], None, rand_between(rng, window_start, now).isoformat(),
                                          items, ["credit_card", "paypal", "bank_transfer", "gift_card"],
                                          ["completed", "processing", "shipped", "delivered"]))
                transaction_counter += 1

        if iteration % 50_000 == 0:
            print(f"  {session_counter:,}/{n_sessions:,} sessions, "
                  f"{transaction_counter:,}/{n_transactions:,} transactions ({time.time() - t0:.0f}s)")

    if sess_writer:
        sess_writer.close()
    txn_writer.close()

    # ---- static entities (products saved last so stock reflects purchases)
    for name, rows in (("users", users), ("categories", categories), ("products", list(inventory.products.values()))):
        w = JsonWriter(os.path.join(args.out, f"{name}.{ext}"), args.format)
        for r in rows:
            w.write(r)
        w.close()

    print(f"""
Dataset generation complete in {time.time() - t0:.0f}s -> {args.out}
- Users: {len(users):,}  Products: {len(products):,}  Categories: {len(categories):,}
- Sessions: {session_counter:,} (target {n_sessions:,}) in {sess_file_idx + 1} file(s)
- Transactions: {transaction_counter:,} (target {n_transactions:,})
- Remaining stock units: {sum(p['current_stock'] for p in inventory.products.values()):,}
""")


if __name__ == "__main__":
    main()
