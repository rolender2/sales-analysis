# Moving Beyond "The What" to "The Why": A Guide to Diagnostic AI

## The Data Gap
Most sales dashboards answer **Descriptive** questions:
- *What* were sales last month? (Aggregations)
- *Where* did they happen? (Geospatial)
- *Which* products sold best? (Ranking)

To answer **Diagnostic** questions (*Why* did sales drop?), AI agents need **Context**. Transactional data (`sales` collection) alone is insufficient because the *cause* of a sales fluctuation usually lies *outside* the transaction itself.

## Requirements for Diagnostic AI

To enable an AI agent to explain causality, we must bridge **Internal Transactional Data** with **External Contextual Data**.

### 1. The Three Pillars of Context
We need to ingest data sources that represent common "Causes":

| Context Layer | Data Source | Example Question Answered |
|---|---|---|
| **Operational** | Marketing Campaigns (`marketing_campaigns`) | "Did the Q4 spike correlate with the $50k Instagram ad spend?" |
| **Environmental** | Economic Indicators (`economic_indicators`) | "Did sales drop due to the national inflation rise?" |
| **Competitive** | Competitor Price Index (`competitor_data`) | "Did we lose market share because Competitor X slashed prices?" |

### 2. The Linking Strategy (Data Engineering)
Just having the data isn't enough; the AI needs a common "Key" to join them.

**Strategy A: Dimensional Joining (Loose Coupling)**
*Recommended for this project*. We rely on shared dimensions: Time and Space.
- **Join Key 1**: `Date` / `Order Date`
- **Join Key 2**: `Region` / `State`
- **Join Key 3**: `Category`

The Agent's Logic:
> "Find all Marketing Campaigns where `target_region` == 'West' AND `date_range` overlaps with '2023-11-01'."

**Strategy B: Key Injection (Tight Coupling)**
Updating the `sales` collection to explicitly include foreign keys.
- Add `campaign_id` to every sales record.
- *Pros*: Easier for simple SQL queries.
- *Cons*: Unrealistic. In the real world, "Ad Impressions" usually live in Facebook Ads Manager, not attached to every line item in ERP.

### 3. Our Implementation Plan

We will implement **Strategy A (Dimensional Joining)** but ensure our data cleanliness allows it:

1.  **Seed Context Data**: Create 2 new MongoDB collections (`marketing_campaigns`, `economic_indicators`) with `Region` and `Date` fields that *perfectly match* the `sales` data format.
2.  **Agent Instruction**: Teach the Data Analyst to "Cross-Reference" collections using these dimensions.
3.  **Data Consistency**: We must ensure `Region` in `sales` ("West") matches `target_region` in `marketing_campaigns` ("West", not "US-West").

## The "Bad Data" Test
To prove the system is robust, we will introduce a "Chaos Monkey" script. This simulates real-world data decay, forcing the **Data Cleaner** to:
- Detect Broken Links (e.g., A sales record with Region="Wst" that fails to join with Marketing data).
- Flag Impossible Values (Negative Profit).
