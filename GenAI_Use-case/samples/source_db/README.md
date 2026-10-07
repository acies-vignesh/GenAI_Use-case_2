# Demo source database: `demo_retail`

A synthetic Indian retail dataset covering 2024-01-01 to 2025-12-31. It contains deliberately planted data quality problems and is used to develop and evaluate the engine end to end.

```bash
# from the project root
backend/.venv/Scripts/python samples/source_db/seed_demo_db.py          # -> data/source.db
backend/.venv/Scripts/python samples/source_db/seed_demo_db.py --seed 7 # different but reproducible data
```

## Model

```
regions (ZONE > STATE > CITY, self-referencing)      categories (DEPARTMENT > CATEGORY > SUBCATEGORY, self-referencing)
   ^ region_id                                            ^ category_id
customers  --< orders --< order_items >-- products -------+
```

| Table | Rows | Notes |
|---|---|---|
| regions | 44 | 4 zones, 12 states, 24 cities, plus 4 planted bad rows |
| categories | 33 | 4 departments, 8 categories, 16 subcategories, plus 2 planted bad rows |
| products | 270 | 15 per subcategory; prices are revised +5% in 2025 |
| customers | ~2.5k | Metro cities weighted 3x |
| orders | ~24k | +25% growth in 2025, Oct/Nov festive peak, weekend uplift |
| order_items | ~49k | 1 to 5 lines per order |

Dates are stored as ISO text (`YYYY-MM-DD`), as SQLite has no native date type. Primary and foreign keys are declared but **not enforced**. That gap is how the orphan rows got in, as often happens in real systems.

## Planted issues → `planted_issues.json`

There are 44 issue types (about 645 affected rows). Each one records its table, columns, DQ dimension, affected keys, and the rule that would catch it. They cover these dimensions:

- **completeness:** missing email, phone, category, unit price
- **validity:** malformed email and phone; out-of-set status, gender, channel and currency codes; negative values
- **uniqueness:** duplicate customers, order numbers, region codes
- **referential integrity:** orphan orders, order lines, products, regions; orders with no lines
- **hierarchy:** level skips, a self-referencing cycle, a product on a non-leaf category, a customer on a STATE instead of a CITY
- **consistency:** ship before order, delivery before ship, DELIVERED with no date, total ≠ Σ lines − discount, signup after first order
- **business rules:** cost > price, under-18 customers, COD on in-store orders, discount > subtotal
- **accuracy / timeliness:** 1900-01-01 placeholder date of birth, quantity outliers, future order dates

This file is the **answer key**. The engine must never see it. It is used only to score the engine's recommendations: which planted issues the approved rules would have caught.
