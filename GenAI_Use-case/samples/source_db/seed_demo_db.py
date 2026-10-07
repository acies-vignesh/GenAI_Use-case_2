"""Creates the demo retail source database (SQLite): 2 years of data (2024-2025) + planted DQ issues.

Usage (from the project root):
    backend/.venv/Scripts/python samples/source_db/seed_demo_db.py [--out PATH] [--seed 42]

Outputs:
    data/source.db                          the source DB the engine connects to (rebuilt each run)
    samples/source_db/planted_issues.json   ground truth of every injected problem; used later to
                                            measure whether the recommended rules would catch them

Only the standard library is used, and a fixed seed makes the output reproducible.
"""
import argparse
import json
import random
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "source.db"
MANIFEST_PATH = Path(__file__).resolve().parent / "planted_issues.json"

START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)
N_CUSTOMERS = 2500
N_ORDERS = 24000
PRODUCTS_PER_SUBCATEGORY = 15

# --------------------------------------------------------------------------- schema
# Keys and FKs are declared (so metadata extraction can find them), but SQLite does not
# enforce FKs unless PRAGMA foreign_keys=ON - which is how the orphan rows get in.
# No CHECK constraints: like most legacy systems, the rules live in people's heads.
DDL = """
CREATE TABLE regions (
    region_id        INTEGER PRIMARY KEY,
    region_code      VARCHAR(10)  NOT NULL,
    region_name      VARCHAR(100) NOT NULL,
    region_level     VARCHAR(10)  NOT NULL,
    parent_region_id INTEGER REFERENCES regions(region_id)
);
CREATE TABLE categories (
    category_id        INTEGER PRIMARY KEY,
    category_code      VARCHAR(20)  NOT NULL,
    category_name      VARCHAR(100) NOT NULL,
    category_level     VARCHAR(12)  NOT NULL,
    parent_category_id INTEGER REFERENCES categories(category_id)
);
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    sku          VARCHAR(20)  NOT NULL UNIQUE,
    product_name VARCHAR(200) NOT NULL,
    category_id  INTEGER REFERENCES categories(category_id),
    brand        VARCHAR(100),
    unit_price   DECIMAL(12,2),
    cost_price   DECIMAL(12,2),
    status       VARCHAR(20),
    launch_date  DATE
);
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    customer_code VARCHAR(20) NOT NULL,
    first_name    VARCHAR(100),
    last_name     VARCHAR(100),
    email         VARCHAR(255),
    phone         VARCHAR(20),
    gender        VARCHAR(10),
    date_of_birth DATE,
    region_id     INTEGER REFERENCES regions(region_id),
    segment       VARCHAR(20),
    signup_date   DATE,
    is_active     BOOLEAN
);
CREATE TABLE orders (
    order_id        INTEGER PRIMARY KEY,
    order_number    VARCHAR(30) NOT NULL,
    customer_id     INTEGER REFERENCES customers(customer_id),
    order_date      DATE NOT NULL,
    ship_date       DATE,
    delivery_date   DATE,
    status          VARCHAR(20),
    channel         VARCHAR(20),
    payment_method  VARCHAR(20),
    currency        VARCHAR(3),
    discount_amount DECIMAL(12,2),
    total_amount    DECIMAL(12,2)
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER,
    unit_price    DECIMAL(12,2),
    discount_pct  DECIMAL(5,2),
    line_amount   DECIMAL(12,2)
);
"""

TABLE_COLUMNS = {
    "regions": ["region_id", "region_code", "region_name", "region_level", "parent_region_id"],
    "categories": ["category_id", "category_code", "category_name", "category_level", "parent_category_id"],
    "products": ["product_id", "sku", "product_name", "category_id", "brand", "unit_price", "cost_price",
                 "status", "launch_date"],
    "customers": ["customer_id", "customer_code", "first_name", "last_name", "email", "phone", "gender",
                  "date_of_birth", "region_id", "segment", "signup_date", "is_active"],
    "orders": ["order_id", "order_number", "customer_id", "order_date", "ship_date", "delivery_date", "status",
               "channel", "payment_method", "currency", "discount_amount", "total_amount"],
    "order_items": ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "discount_pct",
                    "line_amount"],
}

# --------------------------------------------------------------------------- reference data
# Geography hierarchy: ZONE > STATE > CITY
GEOGRAPHY = {
    ("NZ", "North Zone"): {
        ("IN-DL", "Delhi"): [("NDL", "New Delhi"), ("DWK", "Dwarka")],
        ("IN-PB", "Punjab"): [("LDH", "Ludhiana"), ("ASR", "Amritsar")],
        ("IN-UP", "Uttar Pradesh"): [("LKO", "Lucknow"), ("NOI", "Noida")],
    },
    ("SZ", "South Zone"): {
        ("IN-KA", "Karnataka"): [("BLR", "Bengaluru"), ("MYS", "Mysuru")],
        ("IN-TN", "Tamil Nadu"): [("MAA", "Chennai"), ("CJB", "Coimbatore")],
        ("IN-KL", "Kerala"): [("COK", "Kochi"), ("TRV", "Thiruvananthapuram")],
    },
    ("EZ", "East Zone"): {
        ("IN-WB", "West Bengal"): [("CCU", "Kolkata"), ("SLG", "Siliguri")],
        ("IN-OR", "Odisha"): [("BBI", "Bhubaneswar"), ("CTC", "Cuttack")],
        ("IN-BR", "Bihar"): [("PAT", "Patna"), ("GAY", "Gaya")],
    },
    ("WZ", "West Zone"): {
        ("IN-MH", "Maharashtra"): [("BOM", "Mumbai"), ("PNQ", "Pune")],
        ("IN-GJ", "Gujarat"): [("AMD", "Ahmedabad"), ("STV", "Surat")],
        ("IN-RJ", "Rajasthan"): [("JAI", "Jaipur"), ("UDR", "Udaipur")],
    },
}
METRO_CITIES = {"NDL", "BLR", "MAA", "CCU", "BOM", "PNQ"}

# Product hierarchy: DEPARTMENT > CATEGORY > SUBCATEGORY -> (min_price, max_price, brands)
CATALOG = {
    ("ELEC", "Electronics"): {
        ("ELEC-MOB", "Mobiles"): {
            ("ELEC-MOB-SMT", "Smartphones"): (8000, 90000, ["Samsung", "Xiaomi", "OnePlus", "Apple"]),
            ("ELEC-MOB-ACC", "Mobile Accessories"): (199, 2999, ["boAt", "Portronics", "Ambrane"]),
        },
        ("ELEC-CMP", "Computers"): {
            ("ELEC-CMP-LAP", "Laptops"): (30000, 150000, ["Dell", "HP", "Lenovo", "Asus"]),
            ("ELEC-CMP-MON", "Monitors"): (7000, 40000, ["LG", "Dell", "BenQ"]),
        },
        ("ELEC-AUD", "Audio"): {
            ("ELEC-AUD-HPH", "Headphones"): (499, 25000, ["Sony", "boAt", "JBL"]),
            ("ELEC-AUD-SPK", "Speakers"): (999, 30000, ["JBL", "Sony", "Marshall"]),
        },
    },
    ("HOME", "Home & Kitchen"): {
        ("HOME-KIT", "Kitchen Appliances"): {
            ("HOME-KIT-MIX", "Mixer Grinders"): (1800, 9000, ["Prestige", "Bajaj", "Philips"]),
            ("HOME-KIT-CKW", "Cookware"): (500, 6000, ["Hawkins", "Prestige", "Pigeon"]),
        },
        ("HOME-FUR", "Furniture"): {
            ("HOME-FUR-CHR", "Chairs"): (2000, 25000, ["Nilkamal", "Green Soul", "IKEA"]),
            ("HOME-FUR-TBL", "Tables"): (3000, 40000, ["Nilkamal", "IKEA", "Urban Ladder"]),
        },
    },
    ("FASH", "Fashion"): {
        ("FASH-MEN", "Men"): {
            ("FASH-MEN-SHT", "Men's Shirts"): (499, 3999, ["Allen Solly", "Peter England", "Van Heusen"]),
            ("FASH-MEN-FTW", "Men's Footwear"): (799, 8999, ["Bata", "Puma", "Woodland"]),
        },
        ("FASH-WMN", "Women"): {
            ("FASH-WMN-DRS", "Dresses"): (699, 5999, ["W", "Biba", "Zara"]),
            ("FASH-WMN-BAG", "Handbags"): (599, 9999, ["Lavie", "Caprese", "Baggit"]),
        },
    },
    ("GROC", "Grocery"): {
        ("GROC-STP", "Staples"): {
            ("GROC-STP-RIC", "Rice"): (60, 1200, ["India Gate", "Daawat", "Fortune"]),
            ("GROC-STP-PUL", "Pulses"): (80, 600, ["Tata Sampann", "Fortune", "24 Mantra"]),
        },
        ("GROC-BEV", "Beverages"): {
            ("GROC-BEV-TEA", "Tea"): (90, 900, ["Tata Tea", "Red Label", "Taj Mahal"]),
            ("GROC-BEV-COF", "Coffee"): (120, 1200, ["Nescafe", "Bru", "Davidoff"]),
        },
    },
}

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Rohan", "Karthik", "Rahul", "Vikram", "Siddharth",
               "Ananya", "Diya", "Priya", "Sneha", "Kavya", "Meera", "Isha", "Pooja", "Lakshmi", "Nandini",
               "Rajesh", "Suresh", "Ramesh", "Anil", "Deepak", "Divya", "Swati", "Neha", "Asha", "Revathi",
               "Mohammed", "Imran", "Farhan", "Ayesha", "Zara", "Harpreet", "Gurpreet", "Simran", "John", "Mary"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Iyer", "Nair", "Reddy", "Rao", "Patel", "Shah", "Mehta",
              "Kumar", "Singh", "Das", "Bose", "Banerjee", "Mukherjee", "Pillai", "Menon", "Joshi", "Kulkarni",
              "Khan", "Ali", "Fernandes", "D'Souza", "Chatterjee", "Mishra", "Pandey", "Yadav", "Gill", "Naidu"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com"]
MODEL_WORDS = ["Pro", "Lite", "Max", "Plus", "Classic", "Neo", "Prime", "Ultra"]

VALID_ORDER_STATUSES = ["PLACED", "SHIPPED", "DELIVERED", "CANCELLED", "RETURNED"]


def rand_date(rng, start, end):
    return start + timedelta(days=rng.randint(0, (end - start).days))


# --------------------------------------------------------------------------- planted-issue bookkeeping
class Planter:
    """Picks rows to corrupt (never the same row twice per table) and logs each issue to the manifest."""

    def __init__(self, rng):
        self.rng = rng
        self.issues = []
        self.used = defaultdict(set)

    def pick(self, table, rows, key, n, where=None):
        pool = [r for r in rows if r[key] not in self.used[table] and (where is None or where(r))]
        chosen = self.rng.sample(pool, n)
        self.used[table].update(r[key] for r in chosen)
        return chosen

    def log(self, table, columns, dimension, description, keys, expected_rule):
        self.issues.append({
            "issue_id": f"PI-{len(self.issues) + 1:03d}",
            "table": table,
            "columns": columns,
            "dimension": dimension,
            "description": description,
            "expected_rule": expected_rule,
            "count": len(keys),
            "affected_keys": sorted(keys),
        })


# --------------------------------------------------------------------------- builders
def build_regions():
    rows, rid = [], 0
    for (zcode, zname), states in GEOGRAPHY.items():
        rid += 1
        zone_id = rid
        rows.append(dict(region_id=zone_id, region_code=zcode, region_name=zname, region_level="ZONE",
                         parent_region_id=None))
        for (scode, sname), cities in states.items():
            rid += 1
            state_id = rid
            rows.append(dict(region_id=state_id, region_code=scode, region_name=sname, region_level="STATE",
                             parent_region_id=zone_id))
            for ccode, cname in cities:
                rid += 1
                rows.append(dict(region_id=rid, region_code=ccode, region_name=cname, region_level="CITY",
                                 parent_region_id=state_id))
    return rows


def build_categories():
    rows, leaves, cid = [], [], 0
    for (dcode, dname), cats in CATALOG.items():
        cid += 1
        dept_id = cid
        rows.append(dict(category_id=dept_id, category_code=dcode, category_name=dname,
                         category_level="DEPARTMENT", parent_category_id=None))
        for (ccode, cname), subs in cats.items():
            cid += 1
            cat_id = cid
            rows.append(dict(category_id=cat_id, category_code=ccode, category_name=cname,
                             category_level="CATEGORY", parent_category_id=dept_id))
            for (scode, sname), (lo, hi, brands) in subs.items():
                cid += 1
                rows.append(dict(category_id=cid, category_code=scode, category_name=sname,
                                 category_level="SUBCATEGORY", parent_category_id=cat_id))
                leaves.append(dict(category_id=cid, name=sname, dept=dcode, lo=lo, hi=hi, brands=brands))
    return rows, leaves


def build_products(rng, leaves):
    rows, pid = [], 0
    for leaf in leaves:
        singular = leaf["name"][:-1] if leaf["name"].endswith("s") else leaf["name"]
        for _ in range(PRODUCTS_PER_SUBCATEGORY):
            pid += 1
            brand = rng.choice(leaf["brands"])
            price = max(leaf["lo"], round(rng.uniform(leaf["lo"], leaf["hi"]), -1) - 1)
            # ~85% launched before the data window, the rest launch during it
            launch = (rand_date(rng, date(2019, 1, 1), date(2023, 12, 31)) if rng.random() < 0.85
                      else rand_date(rng, START_DATE, date(2025, 10, 1)))
            rows.append(dict(
                product_id=pid, sku=f"SKU-{pid:06d}",
                product_name=f"{brand} {singular} {rng.choice(MODEL_WORDS)} {rng.randint(100, 999)}",
                category_id=leaf["category_id"], brand=brand,
                unit_price=float(price), cost_price=round(price * rng.uniform(0.55, 0.85), 2),
                status="ACTIVE" if rng.random() < 0.9 else "DISCONTINUED",
                launch_date=launch, _dept=leaf["dept"],
            ))
    return rows


def build_customers(rng, city_ids, metro_ids):
    weights = [3 if c in metro_ids else 1 for c in city_ids]
    rows = []
    for cid in range(1, N_CUSTOMERS + 1):
        fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        handle = f"{fn}.{ln}".lower().replace("'", "")
        rows.append(dict(
            customer_id=cid, customer_code=f"CUST-{cid:06d}", first_name=fn, last_name=ln,
            email=f"{handle}{cid}@{rng.choice(EMAIL_DOMAINS)}",
            phone=f"+91{rng.choice('6789')}{rng.randint(0, 10**9 - 1):09d}",
            gender=rng.choices(["M", "F", "O"], [49, 49, 2])[0],
            date_of_birth=rand_date(rng, date(1950, 1, 1), date(2005, 12, 31)),
            region_id=rng.choices(city_ids, weights)[0],
            segment=rng.choices(["CONSUMER", "CORPORATE", "SMALL_BUSINESS"], [80, 12, 8])[0],
            signup_date=rand_date(rng, date(2021, 1, 1), date(2025, 11, 30)),
            is_active=1 if rng.random() < 0.93 else 0,
        ))
    return rows


def day_weight(d):
    w = 1.25 if d.year == 2025 else 1.0       # year-on-year growth
    if d.month in (10, 11):
        w *= 1.8                               # festive season (Dussehra / Diwali)
    if d.weekday() >= 5:
        w *= 1.15                              # weekend uplift
    return w


def build_orders_and_items(rng, customers, products):
    days = [START_DATE + timedelta(n) for n in range((END_DATE - START_DATE).days + 1)]
    order_dates = sorted(rng.choices(days, weights=[day_weight(d) for d in days], k=N_ORDERS))

    by_signup = sorted(customers, key=lambda c: c["signup_date"])
    signups = [c["signup_date"] for c in by_signup]
    by_launch = sorted(products, key=lambda p: p["launch_date"])
    launches = [p["launch_date"] for p in by_launch]

    orders, items, item_id = [], [], 0
    for oid, od in enumerate(order_dates, start=1):
        eligible = bisect_right(signups, od)
        customer = by_signup[int(eligible * rng.random() ** 1.5)]   # older customers order more often

        age = (END_DATE - od).days
        if age <= 2:
            status = "PLACED"
        elif age <= 6:
            status = "SHIPPED"
        else:
            status = rng.choices(["DELIVERED", "CANCELLED", "RETURNED"], [88, 7, 5])[0]
        ship = od + timedelta(rng.randint(1, 3)) if status in ("SHIPPED", "DELIVERED", "RETURNED") else None
        delivery = (min(ship + timedelta(rng.randint(1, 6)), END_DATE)
                    if status in ("DELIVERED", "RETURNED") else None)

        channel = rng.choices(["ONLINE", "MOBILE_APP", "STORE"], [45, 35, 20])[0]
        if channel == "STORE":
            payment = rng.choices(["CARD", "UPI", "CASH"], [40, 45, 15])[0]
        else:
            payment = rng.choices(["CARD", "UPI", "COD", "NET_BANKING", "WALLET"], [30, 35, 20, 8, 7])[0]

        available = by_launch[:bisect_right(launches, od)]
        n_items = rng.choices([1, 2, 3, 4, 5], [45, 25, 15, 10, 5])[0]
        subtotal = 0.0
        for product in rng.sample(available, n_items):
            item_id += 1
            qty = rng.randint(1, 6) if product["_dept"] == "GROC" else rng.choices([1, 2, 3], [80, 15, 5])[0]
            price = round(product["unit_price"] * (1.05 if od.year == 2025 else 1.0), 2)   # 2025 price revision
            disc = rng.choices([0, 5, 10, 15, 20], [60, 15, 12, 8, 5])[0]
            line = round(qty * price * (1 - disc / 100), 2)
            subtotal += line
            items.append(dict(order_item_id=item_id, order_id=oid, product_id=product["product_id"],
                              quantity=qty, unit_price=price, discount_pct=float(disc), line_amount=line))

        discount = (round(min(rng.choice([50, 100, 200, 250, 500]), subtotal * 0.2), 2)
                    if rng.random() < 0.15 else 0.0)
        orders.append(dict(
            order_id=oid, order_number=f"ORD-{od:%Y%m%d}-{oid:06d}", customer_id=customer["customer_id"],
            order_date=od, ship_date=ship, delivery_date=delivery, status=status, channel=channel,
            payment_method=payment, currency="INR", discount_amount=discount,
            total_amount=round(subtotal - discount, 2), _subtotal=round(subtotal, 2),
        ))
    return orders, items


# --------------------------------------------------------------------------- planted issues
def plant_hierarchy_issues(p, regions, categories):
    code = {r["region_code"]: r["region_id"] for r in regions}

    def add_region(**kw):
        kw["region_id"] = max(r["region_id"] for r in regions) + 1
        regions.append(kw)
        return kw["region_id"]

    rid = add_region(region_code="GGN", region_name="Gurugram", region_level="CITY", parent_region_id=code["NZ"])
    p.log("regions", ["region_level", "parent_region_id"], "hierarchy",
          "CITY attached directly to a ZONE, skipping the STATE level", [rid], "A CITY's parent must be a STATE")
    rid = add_region(region_code="NAG", region_name="Nagpur", region_level="CITY", parent_region_id=999)
    p.log("regions", ["parent_region_id"], "referential_integrity",
          "parent_region_id points to a region that does not exist", [rid],
          "parent_region_id must exist in regions.region_id")
    rid = add_region(region_code="hsr1", region_name="Hosur", region_level="CITY", parent_region_id=code["IN-TN"])
    p.log("regions", ["region_code"], "validity", "CITY code breaks the 3-uppercase-letter pattern", [rid],
          "CITY region_code matches ^[A-Z]{3}$")
    rid = add_region(region_code="PNQ", region_name="Pune East", region_level="CITY", parent_region_id=code["IN-MH"])
    p.log("regions", ["region_code"], "uniqueness", "Duplicate region_code (PNQ already used by Pune)", [rid],
          "region_code must be unique")

    cat = {c["category_code"]: c["category_id"] for c in categories}

    def add_category(**kw):
        kw["category_id"] = max(c["category_id"] for c in categories) + 1
        categories.append(kw)
        return kw["category_id"]

    cid = add_category(category_code="ELEC-WRB-SMW", category_name="Smartwatches", category_level="SUBCATEGORY",
                       parent_category_id=cat["ELEC"])
    p.log("categories", ["category_level", "parent_category_id"], "hierarchy",
          "SUBCATEGORY attached directly to a DEPARTMENT, skipping the CATEGORY level", [cid],
          "A SUBCATEGORY's parent must be a CATEGORY")
    cid = add_category(category_code="MISC-CLR", category_name="Clearance", category_level="CATEGORY",
                       parent_category_id=None)
    categories[-1]["parent_category_id"] = cid
    p.log("categories", ["parent_category_id"], "hierarchy", "Category is its own parent (cycle)", [cid],
          "parent_category_id must differ from category_id and the hierarchy must be acyclic")


def plant_product_issues(p, products, categories):
    non_leaf = [c["category_id"] for c in categories if c["category_level"] == "CATEGORY"]
    key = "product_id"

    rows = p.pick("products", products, key, 3)
    for r in rows:
        r["category_id"] = None
    p.log("products", ["category_id"], "completeness", "Product without a category", [r[key] for r in rows],
          "category_id is mandatory")

    rows = p.pick("products", products, key, 3)
    for r in rows:
        r["category_id"] = 9000 + r[key]
    p.log("products", ["category_id"], "referential_integrity", "category_id does not exist in categories",
          [r[key] for r in rows], "products.category_id must exist in categories.category_id")

    rows = p.pick("products", products, key, 4)
    for r in rows:
        r["category_id"] = p.rng.choice(non_leaf)
    p.log("products", ["category_id"], "hierarchy", "Product mapped to a non-leaf (CATEGORY-level) node",
          [r[key] for r in rows], "Products must be mapped to SUBCATEGORY-level categories")

    rows = p.pick("products", products, key, 6)
    for r in rows:
        r["cost_price"] = round(r["unit_price"] * p.rng.uniform(1.1, 1.5), 2)
    p.log("products", ["cost_price", "unit_price"], "business_rule", "cost_price higher than selling price",
          [r[key] for r in rows], "cost_price <= unit_price")

    rows = p.pick("products", products, key, 3)
    for r in rows:
        r["unit_price"] = p.rng.choice([0.0, -1.0])
    p.log("products", ["unit_price"], "validity", "Zero or negative selling price", [r[key] for r in rows],
          "unit_price > 0")

    rows = p.pick("products", products, key, 4)
    for r in rows:
        r["status"] = p.rng.choice(["Actv", "active", "DISC"])
    p.log("products", ["status"], "validity", "status outside the allowed set", [r[key] for r in rows],
          "status IN ('ACTIVE','DISCONTINUED')")


def plant_customer_issues(p, customers, regions):
    key = "customer_id"
    state_ids = [r["region_id"] for r in regions if r["region_level"] == "STATE"]

    rows = p.pick("customers", customers, key, 75)
    for r in rows:
        r["email"] = None
    p.log("customers", ["email"], "completeness", "Missing email (~3%)", [r[key] for r in rows],
          "email null rate below an agreed threshold (e.g. 1%)")

    rows = p.pick("customers", customers, key, 25)
    for r in rows:
        local, domain = r["email"].split("@")
        r["email"] = p.rng.choice([f"{local}{domain}", f"{local}@{domain.replace('.', '')}", f"{local} @{domain}"])
    p.log("customers", ["email"], "validity", "Malformed email address", [r[key] for r in rows],
          "email matches a valid email pattern")

    rows = p.pick("customers", customers, key, 50)
    for r in rows:
        r["phone"] = None
    p.log("customers", ["phone"], "completeness", "Missing phone (~2%)", [r[key] for r in rows],
          "phone null rate below threshold")

    rows = p.pick("customers", customers, key, 25)
    for r in rows:
        r["phone"] = str(p.rng.randint(10**7, 10**8 - 1))
    p.log("customers", ["phone"], "validity", "Phone not in +91XXXXXXXXXX format (8 digits, no country code)",
          [r[key] for r in rows], "phone matches ^\\+91[6-9][0-9]{9}$")

    rows = p.pick("customers", customers, key, 25)
    for r in rows:
        r["gender"] = p.rng.choice(["Male", "female", "X", "U"])
    p.log("customers", ["gender"], "validity", "Gender code outside the allowed set", [r[key] for r in rows],
          "gender IN ('M','F','O')")

    rows = p.pick("customers", customers, key, 8)
    for r in rows:
        r["date_of_birth"] = rand_date(p.rng, date(2012, 1, 1), date(2018, 12, 31))
    p.log("customers", ["date_of_birth"], "business_rule", "Customer younger than 18", [r[key] for r in rows],
          "Customer age at signup >= 18")

    rows = p.pick("customers", customers, key, 6)
    for r in rows:
        r["date_of_birth"] = date(1900, 1, 1)
    p.log("customers", ["date_of_birth"], "accuracy", "Placeholder date of birth 1900-01-01",
          [r[key] for r in rows], "date_of_birth not a known default/placeholder; age <= 110")

    rows = p.pick("customers", customers, key, 10)
    for r in rows:
        r["region_id"] = 9000 + r[key]
    p.log("customers", ["region_id"], "referential_integrity", "region_id does not exist in regions",
          [r[key] for r in rows], "customers.region_id must exist in regions.region_id")

    rows = p.pick("customers", customers, key, 10)
    for r in rows:
        r["region_id"] = p.rng.choice(state_ids)
    p.log("customers", ["region_id"], "hierarchy", "Customer mapped to a STATE instead of a CITY",
          [r[key] for r in rows], "customers.region_id must reference a CITY-level region")

    sources = p.pick("customers", customers, key, 12)
    dup_ids = []
    for src in sources:
        new_id = max(c["customer_id"] for c in customers) + 1
        customers.append(dict(src, customer_id=new_id, customer_code=f"CUST-{new_id:06d}",
                              signup_date=src["signup_date"] + timedelta(p.rng.randint(30, 300))))
        p.used["customers"].add(new_id)
        dup_ids.append(new_id)
    p.log("customers", ["email", "first_name", "last_name"], "uniqueness",
          "Duplicate customer (same name, email, phone) registered under a new customer_id", dup_ids,
          "email must be unique across customers")


def plant_order_issues(p, orders, customers):
    key = "order_id"
    max_customer = max(c["customer_id"] for c in customers)

    rows = p.pick("orders", orders, key, 20)
    for i, r in enumerate(rows, start=1):
        r["customer_id"] = max_customer + 1000 + i
    p.log("orders", ["customer_id"], "referential_integrity", "Order placed by a customer that does not exist",
          [r[key] for r in rows], "orders.customer_id must exist in customers.customer_id")

    rows = p.pick("orders", orders, key, 40)
    for r in rows:
        r["status"] = p.rng.choice(["SHIPED", "unknown", "delivered", "Dlvrd"])
    p.log("orders", ["status"], "validity", "Order status outside the allowed set", [r[key] for r in rows],
          f"status IN {tuple(VALID_ORDER_STATUSES)}")

    rows = p.pick("orders", orders, key, 25, lambda r: r["status"] == "DELIVERED")
    for r in rows:
        r["delivery_date"] = None
    p.log("orders", ["status", "delivery_date"], "consistency", "DELIVERED order without a delivery_date",
          [r[key] for r in rows], "status = 'DELIVERED' implies delivery_date IS NOT NULL")

    rows = p.pick("orders", orders, key, 30, lambda r: r["ship_date"] is not None)
    for r in rows:
        r["ship_date"] = r["order_date"] - timedelta(p.rng.randint(1, 5))
    p.log("orders", ["order_date", "ship_date"], "consistency", "Shipped before it was ordered",
          [r[key] for r in rows], "ship_date >= order_date")

    rows = p.pick("orders", orders, key, 15, lambda r: r["delivery_date"] is not None)
    for r in rows:
        r["delivery_date"] = r["ship_date"] - timedelta(p.rng.randint(1, 3))
    p.log("orders", ["ship_date", "delivery_date"], "consistency", "Delivered before it was shipped",
          [r[key] for r in rows], "delivery_date >= ship_date")

    rows = p.pick("orders", orders, key, 5)
    for r in rows:
        r.update(order_date=date(2027, p.rng.randint(1, 12), p.rng.randint(1, 28)), status="PLACED",
                 ship_date=None, delivery_date=None)
    p.log("orders", ["order_date"], "timeliness", "Order date in the future", [r[key] for r in rows],
          "order_date <= current date")

    rows = p.pick("orders", orders, key, 15)
    for r in rows:
        r["currency"] = p.rng.choice(["US$", "inr", None])
    p.log("orders", ["currency"], "validity", "Invalid or missing currency code", [r[key] for r in rows],
          "currency is a valid ISO-4217 code (here always 'INR')")

    rows = p.pick("orders", orders, key, 20)
    for r in rows:
        r["channel"] = {"ONLINE": "online", "MOBILE_APP": "app", "STORE": "store"}[r["channel"]]
    p.log("orders", ["channel"], "validity", "Channel code not standardised (lowercase / alias)",
          [r[key] for r in rows], "channel IN ('ONLINE','MOBILE_APP','STORE')")

    rows = p.pick("orders", orders, key, 10, lambda r: r["channel"] == "STORE")
    for r in rows:
        r["payment_method"] = "COD"
    p.log("orders", ["channel", "payment_method"], "business_rule", "Cash-on-delivery used for an in-store order",
          [r[key] for r in rows], "payment_method = 'COD' only when channel <> 'STORE'")

    rows = p.pick("orders", orders, key, 10)
    for r in rows:
        r["order_number"] = p.rng.choice(orders)["order_number"]
    p.log("orders", ["order_number"], "uniqueness", "order_number reused by another order",
          [r[key] for r in rows], "order_number must be unique")

    rows = p.pick("orders", orders, key, 40)
    for r in rows:
        r["total_amount"] = round(r["total_amount"] + p.rng.uniform(50, 2000), 2)
    p.log("orders", ["total_amount", "discount_amount"], "consistency",
          "total_amount does not equal sum(order_items.line_amount) - discount_amount", [r[key] for r in rows],
          "total_amount = SUM(line_amount) - discount_amount")

    rows = p.pick("orders", orders, key, 5)
    for r in rows:
        r["discount_amount"] = -100.0
        r["total_amount"] = round(r["_subtotal"] + 100, 2)
    p.log("orders", ["discount_amount"], "validity", "Negative discount amount", [r[key] for r in rows],
          "discount_amount >= 0")

    rows = p.pick("orders", orders, key, 5)
    for r in rows:
        r["discount_amount"] = round(r["_subtotal"] + 100, 2)
        r["total_amount"] = -100.0
    p.log("orders", ["discount_amount", "total_amount"], "business_rule",
          "Discount larger than the order subtotal (negative total)", [r[key] for r in rows],
          "discount_amount <= subtotal and total_amount >= 0")


def plant_item_issues(p, items, orders, products):
    key = "order_item_id"
    max_product = max(pr["product_id"] for pr in products)

    # Orders with no lines: append header-only orders
    empty_ids = []
    for _ in range(10):
        src = p.rng.choice(orders)
        new_id = max(o["order_id"] for o in orders) + 1
        orders.append(dict(src, order_id=new_id, order_number=f"ORD-{src['order_date']:%Y%m%d}-{new_id:06d}"))
        empty_ids.append(new_id)
    p.log("orders", ["order_id"], "referential_integrity", "Order header without any order_items",
          empty_ids, "Every order has at least one order_item")

    # Lines pointing at orders that do not exist: append orphan lines
    orphan_ids, valid_order_max = [], max(o["order_id"] for o in orders)
    for i in range(15):
        src = p.rng.choice(items)
        new_id = max(it["order_item_id"] for it in items) + 1
        items.append(dict(src, order_item_id=new_id, order_id=valid_order_max + 500 + i))
        p.used["order_items"].add(new_id)
        orphan_ids.append(new_id)
    p.log("order_items", ["order_id"], "referential_integrity", "Order line whose order does not exist",
          orphan_ids, "order_items.order_id must exist in orders.order_id")

    rows = p.pick("order_items", items, key, 15)
    for i, r in enumerate(rows, start=1):
        r["product_id"] = max_product + 100 + i
    p.log("order_items", ["product_id"], "referential_integrity", "Order line for a product that does not exist",
          [r[key] for r in rows], "order_items.product_id must exist in products.product_id")

    rows = p.pick("order_items", items, key, 20)
    for r in rows:
        r["quantity"] = p.rng.choice([0, -1, -2])
    p.log("order_items", ["quantity"], "validity", "Zero or negative quantity", [r[key] for r in rows],
          "quantity > 0")

    rows = p.pick("order_items", items, key, 5)
    for r in rows:
        r["quantity"] = 5000
    p.log("order_items", ["quantity"], "accuracy", "Extreme quantity outlier (5000 units)",
          [r[key] for r in rows], "quantity within an expected range (e.g. <= 100) or flagged as outlier")

    rows = p.pick("order_items", items, key, 10)
    for r in rows:
        r["discount_pct"] = p.rng.choice([-5.0, 120.0, 150.0])
    p.log("order_items", ["discount_pct"], "validity", "Discount percentage outside 0-100",
          [r[key] for r in rows], "discount_pct BETWEEN 0 AND 100")

    rows = p.pick("order_items", items, key, 30)
    for r in rows:
        r["line_amount"] = round(r["line_amount"] * p.rng.uniform(1.2, 2.0), 2)
    p.log("order_items", ["line_amount", "quantity", "unit_price", "discount_pct"], "consistency",
          "line_amount does not equal quantity * unit_price * (1 - discount_pct/100)", [r[key] for r in rows],
          "line_amount = ROUND(quantity * unit_price * (1 - discount_pct/100), 2)")

    rows = p.pick("order_items", items, key, 10)
    for r in rows:
        r["unit_price"] = None
    p.log("order_items", ["unit_price"], "completeness", "Missing unit_price on an order line",
          [r[key] for r in rows], "unit_price is mandatory")


def plant_signup_issue(p, customers, orders):
    first_order = {}
    for o in orders:
        cid = o["customer_id"]
        if cid not in first_order or o["order_date"] < first_order[cid]:
            first_order[cid] = o["order_date"]
    rows = p.pick("customers", customers, "customer_id", 15,
                  lambda c: c["customer_id"] in first_order and first_order[c["customer_id"]] < END_DATE)
    for r in rows:
        r["signup_date"] = first_order[r["customer_id"]] + timedelta(p.rng.randint(10, 90))
    p.log("customers", ["signup_date"], "consistency", "Customer signed up after their first order",
          [r["customer_id"] for r in rows], "signup_date <= MIN(orders.order_date) for the customer")


# --------------------------------------------------------------------------- write
def write_db(out_path, tables):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    con = sqlite3.connect(out_path)
    con.execute("PRAGMA foreign_keys = OFF")
    con.executescript(DDL)
    for table, rows in tables.items():
        cols = TABLE_COLUMNS[table]
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"
        con.executemany(sql, [
            {c: (r[c].isoformat() if isinstance(r[c], date) else r[c]) for c in cols} for r in rows
        ])
    con.commit()
    con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    planter = Planter(rng)

    regions = build_regions()
    categories, leaves = build_categories()
    products = build_products(rng, leaves)
    city_ids = [r["region_id"] for r in regions if r["region_level"] == "CITY"]
    metro_ids = {r["region_id"] for r in regions if r["region_code"] in METRO_CITIES}
    customers = build_customers(rng, city_ids, metro_ids)
    orders, items = build_orders_and_items(rng, customers, products)

    plant_hierarchy_issues(planter, regions, categories)
    plant_product_issues(planter, products, categories)
    plant_customer_issues(planter, customers, regions)
    plant_order_issues(planter, orders, customers)
    plant_item_issues(planter, items, orders, products)
    plant_signup_issue(planter, customers, orders)

    tables = {"regions": regions, "categories": categories, "products": products,
              "customers": customers, "orders": orders, "order_items": items}
    write_db(args.out, tables)

    MANIFEST_PATH.write_text(json.dumps({
        "database": "demo_retail",
        "seed": args.seed,
        "data_window": [START_DATE.isoformat(), END_DATE.isoformat()],
        "row_counts": {t: len(r) for t, r in tables.items()},
        "notes": "Item-level corruptions (quantity, line_amount, unit_price) also surface as order-total "
                 "mismatches on their parent orders.",
        "issues": planter.issues,
    }, indent=2), encoding="utf-8")

    print(f"Database : {args.out}")
    for t, r in tables.items():
        print(f"  {t:<12} {len(r):>7,} rows")
    print(f"Manifest : {MANIFEST_PATH}  ({len(planter.issues)} planted issues, "
          f"{sum(i['count'] for i in planter.issues):,} affected rows)")


if __name__ == "__main__":
    main()
