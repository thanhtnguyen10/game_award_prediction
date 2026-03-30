# Steam Games: Award Winner Prediction 🏆

[cite_start]This project implements a machine learning pipeline following the **CRISP-DM** methodology to predict whether a Steam game will become an "Award Winner" (`is_award_winner`)[cite: 3, 5]. [cite_start]By analyzing technical specifications and community reception, this tool serves as a "Talent Scout" for publishers and developers to guide marketing and investment strategies[cite: 6].

## 📌 Business Understanding
[cite_start]The primary objective is to identify prestige titles in a market where "winners" represent an extremely rare, elite tier—less than 1% of the dataset[cite: 5, 197]. 

## 📊 Dataset Description
[cite_start]The dataset consists of **9,874 records** with **26 features**[cite: 18]. The data was aggregated from the Steam Storefront and the SteamSpy API.

### Data Dictionary & Feature Definitions
| Feature | Data Type | Description |
| :--- | :--- | :--- |
| `AppID` | `int64` | [cite_start]Unique numeric identifier for the game on Steam[cite: 103]. |
| `Title` | `object` | [cite_start]The official name of the game[cite: 22]. |
| `Developer` | `object` | [cite_start]The studio or individual creator[cite: 26]. |
| `URL` | `object` | [cite_start]Direct link to the Steam store page[cite: 106]. |
| `game_age_days` | `float64` | [cite_start]Days since release (Reference: March 22, 2026)[cite: 38]. |
| `target_total_reviews` | `int64` | [cite_start]Total count of user reviews[cite: 30]. |
| `Positive_Review_Pct` | `int64` | [cite_start]Percentage of positive reviews (0–100)[cite: 34]. |
| `Price_NTD` | `int64` | [cite_start]Base price in New Taiwan Dollars[cite: 70]. |
| `is_free_to_play` | `bool` | [cite_start]True if the game is free to download[cite: 67]. |
| `is_award_winner` | `bool` | [cite_start]**Target Variable.** True if the game won a Steam Award[cite: 97]. |
| `is_mature` | `bool` | [cite_start]True if the game has an 18+ content rating[cite: 100]. |
| `Min_RAM_GB` | `int64` | [cite_start]Minimum RAM required in Gigabytes[cite: 61]. |
| `Storage_GB` | `int64` | [cite_start]Required disk space in Gigabytes[cite: 64]. |
| `count_languages` | `int64` | [cite_start]Total number of supported interface languages[cite: 42]. |
| `has_audio_english` | `bool` | [cite_start]True if full English audio is provided[cite: 46]. |
| `count_os_supported` | `int64` | [cite_start]Total number of OS (Windows, Mac, Linux) supported[cite: 50]. |
| `is_on_linux` | `bool` | [cite_start]True if the game supports Linux[cite: 54]. |
| `is_on_mac` | `bool` | [cite_start]True if the game supports macOS[cite: 58]. |
| `count_dlcs` | `int64` | [cite_start]Total number of downloadable content packs[cite: 91]. |
| `count_tags` | `int64` | [cite_start]Number of user-defined genre/category tags[cite: 94]. |
| `feat_multiplayer` | `bool` | [cite_start]True if the game supports multiplayer[cite: 73]. |
| `feat_workshop` | `bool` | [cite_start]True if Steam Workshop (modding) is enabled[cite: 76]. |
| `feat_achievements` | `bool` | [cite_start]True if Steam Achievements are available[cite: 85]. |
| `feat_trading_cards` | `bool` | [cite_start]True if the game has Steam Trading Cards[cite: 82]. |
| `feat_in_app_purchases`| `bool` | [cite_start]True if the game includes microtransactions[cite: 79]. |
| `feat_remote_play` | `bool` | [cite_start]True if Remote Play features are enabled[cite: 88]. |

## 🛠️ Technical Implementation
### 1. Data Preparation
* [cite_start]**Cleaning:** Dropped rows missing `game_age_days` and removed non-predictive text columns (`Title`, `URL`, etc.)[cite: 205, 210].
* [cite_start]**Resampling:** Used **SMOTE** to handle class imbalance, increasing training winners from 47 to 782[cite: 203, 237].
* [cite_start]**Scaling:** Normalized numerical features via `StandardScaler`[cite: 202, 224].

### 2. Modeling
* [cite_start]**Logistic Regression:** A baseline linear model with balanced class weights[cite: 366, 370].
* [cite_start]**Random Forest:** An ensemble method optimized via `GridSearchCV`[cite: 472, 486].
* [cite_start]**Neural Network (MLP):** A 3-layer deep architecture with early stopping[cite: 600, 613].

## 📈 Experimental Results
[cite_start]By applying **Youden’s J-Statistic** to calibrate decision thresholds, we moved away from the "Accuracy Paradox" to achieve actionable results[cite: 445, 506].

| Model | AUROC | Recall (Optimized) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9741** | **1.00** | 0.11 |
| **Random Forest** | 0.9767 | 0.25 | 0.18 |
| **Neural Network** | 0.9661 | 1.00 | 0.10 |

> [cite_start]**Conclusion:** The **Optimized Logistic Regression** is the champion model for a "Talent Scout" use case, successfully catching 100% of true winners in the test set[cite: 586, 804].

## 🚀 Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `imblearn`, `seaborn`, `matplotlib`
* **Framework:** CRISP-DM
