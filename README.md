# Supply Chain Analysis: Late Delivery Drivers and Prediction

An analysis of ~180K orders from the **DataCo Smart Supply Chain** dataset. The project measures how often orders arrive late, how late deliveries relate to profit, which factors drive them, and whether late delivery can be **predicted** at order time with a machine-learning model.

## Business Questions
- How many orders are delivered late, and by how many days?
- How profitable is the business, and do delays reduce profit?
- Which factors (region, shipping mode, customer segment, department, payment type) are linked to late delivery?
- Do delays follow a pattern by month, day of week or hour of day?
- Can we predict which orders are at risk of arriving late?

## Dataset
- **Source:** DataCo Smart Supply Chain for Big Data Analysis (Kaggle). 
- **Size:** 180,519 rows and 53 columns, covering orders from 2015 to early 2018
- **Includes:** order and shipping dates, scheduled vs actual shipping days, shipping mode, delivery status, customer segment, market and region, product category, sales and profit per order
- **Column descriptions:** see [`DescriptionDataCoSupplyChain.csv`](DescriptionDataCoSupplyChain.csv)


## Tools
Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · imbalanced-learn (SMOTE) · Jupyter Notebook

## Workflow
1. **Load and inspect** the data (shape, missing values, duplicates, value counts).
2. **Clean:** removed 7,754 cancelled-shipping orders (180,519 → 172,765 rows), converted order and shipping dates to datetime, and found no duplicate rows.
3. **Feature engineering:** order processing time, `Delay` (actual minus scheduled shipping days), a delayed flag, a profit/loss/break-even flag, and order month, day and hour.
4. **Analyse:** business KPIs, delay and profit by delay days, and delay % by region, customer segment, shipping mode, order status, payment type and department.
5. **Model:** a Random Forest classifier predicts `Late_delivery_risk`, using frequency-encoded categorical features and SMOTE to balance the classes.

## Key Results
| Metric | Value |
|---|---|
| Orders analysed | 172,765 |
| Late deliveries | 98,977 (57.3%) |
| On-time or early deliveries | 42.7% |
| 90th-percentile delay (late orders) | 3 days |
| Profitable / loss-making / break-even orders | 80.7% / 18.7% / 0.6% |
| Gross profit from profitable orders | ~$7.5M |
| Net profit (all orders) | ~$3.8M |

## Key Insights
- **More than half of all orders arrive late (57.3%),** but most delays are short. About 59% of late orders are just 1 day late, and 90% are 3 days late or less.
- **Shipping mode is the strongest driver of lateness.** The delay rate is **100% for First Class**, **79.8% for Second Class** and **39.8% for Standard Class**. The faster the promised service, the more often it is missed.
- **Region, customer segment, payment type and department make little difference.** Delay rates stay in a narrow band of roughly 53–59%. *Central Africa* is the highest region at 58.7%.
- **Time patterns are weak.** Delay % varies by only about 2 percentage points across months and weekdays, and about 5 across hours of the day. The highest points are August–September and 8 PM.
- **Delays don't change profit per order much.** Average profit per order is roughly $20–23 whether an order is early, on time or late.
- **About 1 in 5 orders loses money (18.7%).** Gross profit on profitable orders is ~$7.5M, but net profit is ~$3.8M, which means losses of roughly $3.7M. This is a bigger lever than delivery delays.
- **The model gives moderate predictive power.** Random Forest on the held-out 20% of orders scores:

  | Metric | Value |
  |---|---|
  | Accuracy | 0.74 |
  | Precision (late class) | 0.79 |
  | Recall (late class) | 0.75 |
  | F1 (late class) | 0.77 |

## Visualisations
| | |
|---|---|
| ![Profitability distribution](images/profitability_distribution.png) | ![Delay distribution and profit](images/delay_distribution_and_profit.png) |
| ![Delay by category](images/delay_by_category.png) | ![Central Africa drivers](images/central_africa_drivers.png) |
| ![Delay by month, day and hour](images/delay_by_month_day_hour.png) | |

## Limitations
- The model uses only order-time information (scheduled days, shipping mode, category, segment, department, region, month and hour). Accuracy of 0.74 is a useful baseline, not a production-ready predictor.
- Shipping mode explains most of the lateness, so the model largely learns this one relationship.
- The delay % charts by category and time use the order-to-shipping gap versus the schedule (94,523 orders flagged), while the headline KPI uses actual shipping days versus scheduled days (98,977 orders). The two give slightly different percentages (about 55% vs 57%).

## Project Structure
```
supply-chain-analysis/
├── SupplyChainAnalysis.ipynb          # analysis and model
├── DescriptionDataCoSupplyChain.csv   # column descriptions
├── images/                            # charts
├── .gitignore
└── README.md
```

## How to Run
1. Clone the repo and install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```
2. Download `DataCoSupplyChainDataset.csv` from Kaggle and place it in the project folder.
3. Open `SupplyChainAnalysis.ipynb` and run all cells.

## Author
**Sweta Mishra**: B.Tech in Computer Science, KIIT University
GitHub: [swetaamishra16](https://github.com/swetaamishra16)
