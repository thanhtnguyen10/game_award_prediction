# Steam Games: Award Winner Prediction 🏆

This project implements a machine learning pipeline following the **CRISP-DM** methodology to predict whether a Steam game will become an "Award Winner" (`is_award_winner`). By analyzing technical specifications and community reception, this tool serves as a "Talent Scout" for publishers and developers to guide marketing and investment strategies.

## 📌 Business Understanding
The primary objective is to identify prestige titles in a market where "winners" represent an extremely rare, elite tier—less than 1% of the dataset. For game publishers, understanding which pre-launch features correlate with prestige can guide development investments and marketing strategies.

## 📊 Dataset Description
The dataset consists of **9,874 records** with **26 features**, aggregated from the Steam Storefront and the SteamSpy API.

### Data Dictionary & Feature Definitions
| Feature | Data Type | Description |
| :--- | :--- | :--- |
| `AppID` | `int64` | Unique numeric identifier for the game on Steam. |
| `Title` | `object` | The official name of the game. |
| `Developer` | `object` | The studio or individual creator. |
| `URL` | `object` | Direct link to the Steam store page. |
| `game_age_days` | `float64` | Days since release (Reference: March 22, 2026). |
| `target_total_reviews` | `int64` | Total count of user reviews. |
| `Positive_Review_Pct` | `int64` | Percentage of positive reviews (0–100). |
| `Price_NTD` | `int64` | Base price in New Taiwan Dollars. |
| `is_free_to_play` | `bool` | True if the game is free to download. |
| `is_award_winner` | `bool` | **Target Variable.** True if the game won a Steam Award. |
| `is_mature` | `bool` | True if the game has an 18+ content rating. |
| `Min_RAM_GB` | `int64` | Minimum RAM required in Gigabytes. |
| `Storage_GB` | `int64` | Required disk space in Gigabytes. |
| `count_languages` | `int64` | Total number of supported interface languages. |
| `has_audio_english` | `bool` | True if full English audio is provided. |
| `count_os_supported` | `int64` | Total number of OS (Windows, Mac, Linux) supported. |
| `is_on_linux` | `bool` | True if the game supports Linux. |
| `is_on_mac` | `bool` | True if the game supports macOS. |
| `count_dlcs` | `int64` | Total number of downloadable content packs. |
| `count_tags` | `int64` | Number of user-defined genre/category tags. |
| `feat_multiplayer` | `bool` | True if the game supports multiplayer. |
| `feat_workshop` | `bool` | True if Steam Workshop (modding) is enabled. |
| `feat_achievements` | `bool` | True if Steam Achievements are available. |
| `feat_trading_cards` | `bool` | True if the game has Steam Trading Cards. |
| `feat_in_app_purchases`| `bool` | True if the game includes microtransactions. |
| `feat_remote_play` | `bool` | True if Remote Play features are enabled. |

## 🛠️ Technical Implementation
### 1. Data Preparation
* **Cleaning:** Dropped rows missing `game_age_days` and removed non-predictive text columns (`Title`, `Developer`, `URL`, `AppID`).
* **Resampling:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance, increasing training winners from 47 to 782.
* **Scaling:** Normalized numerical features via `StandardScaler`.

### 2. Modeling
* **Logistic Regression:** A baseline linear model with balanced class weights.
* **Random Forest:** An ensemble method optimized via `GridSearchCV` (Max Depth: 15, Estimators: 200).
* **Neural Network (MLP):** A 3-layer deep architecture (128-64-32) with early stopping and L2 regularization.

## 📈 Experimental Results
By applying **Youden’s J-Statistic** to calibrate decision thresholds, we moved away from the "Accuracy Paradox" (where 99% accuracy is misleading) to achieve actionable results.

| Model | AUROC | Recall (Optimized) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9741** | **1.00** | 0.11 |
| **Random Forest** | 0.9767 | 0.25 | 0.18 |
| **Neural Network** | 0.9661 | 1.00 | 0.10 |

> **Conclusion:** The **Optimized Logistic Regression** is the champion model for a "Talent Scout" use case, successfully catching 100% of true winners in the test set by utilizing a calibrated threshold of 0.38.

* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `imblearn`, `seaborn`, `matplotlib`
* **Framework:** CRISP-DM
