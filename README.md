# Student Performance Prediction

Predicting whether a secondary-school student will **pass or fail** their final maths exam, using classic machine-learning classifiers on demographic, family, social, and school-related features.

The whole project lives in one Jupyter notebook: data cleaning, exploratory analysis, feature engineering and reduction, hyperparameter tuning, and a side-by-side comparison of five models (Random Forest, XGBoost, SVM, Neural Network, Logistic Regression).

## Dataset

- **Source:** [Student Performance on Kaggle](https://www.kaggle.com/datasets/whenamancodes/student-performance/data) (`Maths.csv`), which appears to be a copy of the UCI *Student Performance* data (Cortez & Silva, 2008).
- **Size:** 397 records and 33 columns; 395 records remain after removing duplicates.
- **Features:** school, sex, age, address, family size, parents' education and jobs, travel and study time, past failures, school/family support, extra paid classes, activities, internet access, romantic relationship, family relationship, free time, going out, alcohol consumption, health, absences, and the first and second period grades (`G1`, `G2`).
- **Target:** the final grade `G3` (0–20) is converted to a binary label. A student with `G3 >= 8` (40%) is **Pass**; otherwise **Fail**. Fail is the positive class (`1`), so precision and recall below refer to detecting failing students.
- **Class balance:** 325 Pass vs 70 Fail (about 82% / 18%).

The dataset is downloaded automatically by the notebook through `kagglehub`, so it is not stored in this repository.

## Approach

1. **Cleaning:** check for missing values and drop duplicate rows.
2. **Labelling:** discretize `G3` into Pass/Fail, then drop `G3` from the features.
3. **EDA:** descriptive statistics, class balance, histograms and boxplots for numeric features, count plots and fail rates for categorical features, pair plot, correlation heat map, and cluster map.
4. **Encoding:** one-hot encoding for nominal features (`Mjob`, `Fjob`, `reason`, `guardian`) and 0/1 mapping for binary features.
5. **Feature reduction:**
   - drop features whose absolute correlation with the target is below 0.05
   - drop `G1` because it is highly correlated with `G2`
   - apply a variance threshold of 0.01 (no further features were removed)
   - this leaves 22 features
6. **Scaling:** standardize the numeric features with `StandardScaler`.
7. **Split:** 80/20 stratified train/test split (`random_state=42`), giving 316 training and 79 test samples.
8. **Modelling:** train a baseline and a tuned version of each model, using 5-fold cross-validation for tuning.
9. **Evaluation:** accuracy, precision, recall, F1, train vs test accuracy, learning curves, ROC/AUC, confusion matrices, and training time.

## Models and Tuning

| Model                | Tuning method                                                                   |
|----------------------|---------------------------------------------------------------------------------|
| Random Forest        | Randomized search (50 iterations, scored on F1)                                 |
| XGBoost              | Randomized search (50 iterations, scored on F1)                                 |
| SVM                  | Grid search over C / kernel / gamma, kernel comparison, then a poly-kernel grid |
| Neural Network (MLP) | Grid search (768 candidates), architecture comparison, then a narrower grid     |
| Logistic Regression  | Randomized search (10 iterations, scored on F1)                                 |

## Results

Test set of 79 students (14 failing). Metrics are for the Fail class. Each row is the configuration used in the notebook's final comparison.

| Model                      | Accuracy | Precision | Recall | F1    | AUC   | Train time (s) |
|----------------------------|----------|-----------|--------|-------|-------|----------------|
| Random Forest (50 trees)   | 0.937    | 0.846     | 0.786  | 0.815 | 0.936 | 0.061          |
| XGBoost (default)          | 0.924    | 0.750     | 0.857  | 0.800 | 0.982 | 0.037          |
| SVM (poly kernel, default) | 0.937    | 0.909     | 0.714  | 0.800 | 0.971 | 0.007          |
| Neural Network (50-50-50)  | 0.937    | 0.800     | 0.857  | 0.828 | 0.976 | 0.615          |
| Logistic Regression        | 0.937    | 0.846     | 0.786  | 0.815 | 0.979 | 0.022          |

Key observations:

- Four of the five models land on the same test accuracy (74 of 79 correct), so accuracy alone does not separate them.
- Hyperparameter tuning did not improve test performance over the simpler configurations in this run.
- Logistic Regression and SVM are the cheapest to train; the neural network is the slowest and shows no clear advantage.
- Random Forest, XGBoost, and the neural network reach 100% training accuracy, which indicates overfitting on this small dataset.

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook student_performance_prediction.ipynb
```

Notes for running the notebook:

- The first data cell downloads the dataset with `kagglehub`. If it asks for credentials, set up a [Kaggle API token](https://www.kaggle.com/docs/api).
- The file is named `Maths.csv` but the notebook reads it with `pd.read_excel`, so `openpyxl` is required.
- A cell that runs `!ls` and `!file` only works on Linux/macOS and can be skipped on Windows. The Google Colab drive lines are commented out.

## Repository Structure

```
.
├── student_performance_prediction.ipynb   # Full analysis and modelling
├── requirements.txt                       # Python dependencies
└── README.md
```

## Limitations

- **Small test set.** With only 79 test samples (14 positives), one extra correct prediction changes accuracy by about 1.3 points, so differences between models are not statistically meaningful.
- **Data leakage risks.** Feature scaling and the correlation-based feature selection are done on the full dataset before the train/test split, and some choices (SVM kernel, neural-network architecture) were picked by comparing scores on the test set. Reported scores may be slightly optimistic. Moving these steps inside a cross-validated pipeline would fix this.
- **The second-period grade is a feature.** `G2` is kept and is strongly related to the final grade, so the models probably rely on it heavily. Predictions are therefore closer to "will this student fail given their second-term grade" than an early-warning model based on background information alone.
- **Single dataset, single subject.** Results are for maths students at two Portuguese schools and may not generalize to other subjects or populations.

## Acknowledgements

Dataset: P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, 2008.
