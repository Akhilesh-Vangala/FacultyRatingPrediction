# Professor Ratings Analysis: Gender Bias and Predictive Modeling

**Project affiliated with New York University Center for Data Science**


A comprehensive data science project analyzing Rate My Professor data to investigate gender bias in professor ratings and build predictive models for rating and difficulty prediction.

## Overview

This project conducts a thorough statistical analysis of professor ratings data, examining gender differences in academic evaluations and developing machine learning models to predict professor ratings and course difficulty. The analysis employs rigorous statistical methods including hypothesis testing, effect size calculations, and multiple machine learning approaches to provide insights into rating patterns and build robust predictive models.

## Research Questions Addressed

### Gender Bias Analysis

The project investigates several key aspects of gender differences in professor ratings:

- **Mean Rating Differences**: Statistical comparison of average ratings between male and female professors using Welch's t-test, accounting for potential variance differences
- **Variance Analysis**: Examination of spread differences in ratings using Levene's test to assess whether rating distributions differ by gender
- **Effect Size Quantification**: Calculation of Cohen's d with 95% bootstrap confidence intervals to measure the practical significance of observed differences
- **Tag Association Patterns**: Analysis of gender differences across 20 professor characteristic tags to identify which attributes are associated with gender
- **Difficulty Rating Differences**: Investigation of whether perceived course difficulty varies by professor gender
- **Difficulty Effect Sizes**: Quantification of effect sizes for difficulty differences with confidence intervals

### Predictive Modeling

The project develops and compares multiple machine learning models:

- **Rating Prediction from Numerical Features**: Regression models predicting average rating using numerical predictors (difficulty, number of ratings, would-take-again percentage, etc.). Multiple approaches are evaluated including Ordinary Least Squares, Ridge Regression, Lasso, ElasticNet, Random Forest, Gradient Boosting, and XGBoost with feature interaction engineering and hyperparameter tuning.

- **Rating Prediction from Tag Features**: Regression models using normalized tag features to predict ratings, with comprehensive model comparison and feature importance analysis.

- **Difficulty Prediction**: Regression models predicting average difficulty from tag characteristics, with best model selection based on R² and RMSE metrics.

- **Pepper Status Classification**: Binary classification model to identify highly-rated professors ("Pepper" status) using Random Forest with class balancing, evaluated using ROC-AUC and comprehensive classification metrics.

### Extended Analysis

- **Geographic and Disciplinary Patterns**: Analysis of ratings by state and major, including statistical tests for geographic and disciplinary differences
- **Comprehensive Visualizations**: Publication-quality charts and graphs illustrating all findings

## Project Structure

```
.
├── professor_ratings_analysis.py  # Main analysis script
├── requirements.txt                # Python dependencies
├── README.md                        # This file
├── data/                            # Data directory
│   ├── rmpCapstoneNum.csv          # Numerical features
│   ├── rmpCapstoneQual.csv         # Qualitative features (Major, University, State)
│   └── rmpCapstoneTags.csv         # Tag features (20 professor characteristics)
└── .gitignore                       # Git ignore rules
```

## Data Description

The dataset contains professor ratings from Rate My Professor with the following features:

### Numerical Features
- `AvgRating`: Average rating (1-5 scale)
- `AvgDifficulty`: Average difficulty rating
- `NumRatings`: Number of ratings received
- `Pepper`: Binary indicator for highly-rated professors
- `WouldTakeAgain`: Proportion who would take again
- `OnlineRatings`: Number of online ratings
- `Male`, `Female`: Binary gender indicators

### Qualitative Features
- `Major`: Academic discipline
- `University`: Institution name
- `State`: US state or province

### Tag Features (20 characteristics)
ToughGrader, GoodFeedback, Respected, LotsToRead, ParticipationMatters, DontSkipClass, LotsOfHomework, Inspirational, PopQuizzes, Accessible, SoManyPapers, ClearGrading, Hilarious, TestHeavy, GradedByFewThings, AmazingLectures, Caring, ExtraCredit, GroupProjects, LectureHeavy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Akhilesh-Vangala/Faculty-Rating-Analysis.git
cd Faculty-Rating-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete analysis:

```bash
python professor_ratings_analysis.py
```

The script will:
1. Load and preprocess the data
2. Perform all statistical analyses
3. Train and evaluate multiple models
4. Generate visualizations (saved as PNG files)
5. Print comprehensive results to console

## Methodology

### Statistical Analysis
- **Hypothesis Testing**: Welch's t-test for unequal variances, Levene's test for variance equality
- **Effect Sizes**: Cohen's d with bootstrap confidence intervals
- **Significance Level**: α = 0.005 for all tests
- **Bootstrap Methods**: 10,000 bootstrap iterations for confidence interval estimation

### Machine Learning
- **Train/Test Split**: 80/20 with stratification for classification tasks
- **Cross-Validation**: 5-fold CV for hyperparameter tuning
- **Regularization**: Ridge, Lasso, and ElasticNet with grid search optimization
- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost with tuned hyperparameters
- **Evaluation Metrics**: R² and RMSE for regression; ROC-AUC, precision, recall, and F1-score for classification
- **Feature Engineering**: Interaction terms, tag normalization, and feature scaling

### Data Preprocessing
- Missing value handling: median imputation for numerical features
- Tag normalization: normalized by (NumRatings × 3) to account for variable rating counts
- Feature scaling: StandardScaler for all regression models
- Minimum ratings threshold: ≥2 ratings for meaningful analysis
- Gender filtering: Analysis restricted to professors with known gender information

## Results Summary

The analysis includes comprehensive model comparisons showing:
- Best performing models for each prediction task with detailed performance metrics
- Feature importance rankings for all models
- Model improvement percentages over baseline approaches
- Statistical significance of gender differences with effect sizes
- Comprehensive visualizations of all findings

All results are printed to console and visualizations are saved as high-resolution PNG files (300 DPI) suitable for publication.

## Visualizations

The script generates multiple publication-quality visualizations:
- Distribution comparisons (KDE plots, box plots, violin plots)
- Effect size visualizations with confidence intervals
- Model performance comparisons (bar charts, scatter plots)
- Feature importance plots
- ROC curves and confusion matrices
- Geographic and disciplinary analysis charts
- Predicted vs. actual value scatter plots with regression lines

## Technical Details

- **Language**: Python 3.7+
- **Key Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Random Seed**: 16467256 (for reproducibility)
- **Figure DPI**: 300 (publication quality)
- **Bootstrap Iterations**: 10,000 for confidence intervals

## Key Contributions

This project demonstrates:
- Rigorous statistical analysis of gender bias in academic evaluations
- Comprehensive comparison of multiple machine learning approaches
- Best practices in data preprocessing and feature engineering
- Publication-quality visualizations and reporting
- Reproducible research with fixed random seeds and detailed methodology

## License

This project is for academic purposes.

## Acknowledgments

Data source: Rate My Professor dataset
