import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import levene, f_oneway
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.utils import resample

# Try to import xgboost and set availability flag
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


N_NUMBER = 16467256
np.random.seed(N_NUMBER)

num_df = pd.read_csv('data/rmpCapstoneNum.csv', header=None)
qual_df = pd.read_csv('data/rmpCapstoneQual.csv', header=None)
tags_df = pd.read_csv('data/rmpCapstoneTags.csv', header=None)


num_df.columns = ['AvgRating', 'AvgDifficulty', 'NumRatings', 'Pepper', 'WouldTakeAgain', 'OnlineRatings', 'Male', 'Female']
qual_df.columns = ['Major', 'University', 'State']
tag_names = ['ToughGrader', 'GoodFeedback', 'Respected', 'LotsToRead', 'ParticipationMatters', 'DontSkipClass', 'LotsOfHomework', 'Inspirational', 'PopQuizzes', 'Accessible', 'SoManyPapers', 'ClearGrading', 'Hilarious', 'TestHeavy', 'GradedByFewThings', 'AmazingLectures', 'Caring', 'ExtraCredit', 'GroupProjects', 'LectureHeavy']
tags_df.columns = tag_names

print(f"Numerical data shape: {num_df.shape}")
print(f"Qualitative data shape: {qual_df.shape}")
print(f"Tags data shape: {tags_df.shape}")

df = pd.concat([num_df, qual_df, tags_df], axis=1)
print(f"\nCombined dataframe shape: {df.shape}")
print(f"\nMissing data:")
print(df.isnull().sum())

# create gender variable - 1=male, 0=female, -1=unknown
df['Gender'] = np.where(df['Male'] == 1, 1,
                       np.where(df['Female'] == 1, 0, -1))

# filter to only known gender for gender analyses
df_gender = df[df['Gender'] != -1].copy()
# according to instructions, ratings with only 1 rating are less meaningful trying threshold of 2 - seems reasonable
MIN_RATINGS = 2
df_clean = df_gender[df_gender['NumRatings'] >= MIN_RATINGS].copy()
print(f"Original: {len(df)}")
print(f"With known gender: {len(df_gender)}")
print(f"With >= {MIN_RATINGS} ratings: {len(df_clean)}")

missing_cols = ['AvgRating', 'AvgDifficulty', 'WouldTakeAgain']
print(df_clean[missing_cols].isnull().sum())

# normalize tags - each rating can give up to 3 tags, so max possible = NumRatings * 3 dividing by that gives us the proportion
for tag in tag_names:
    df_clean[f'{tag}_normalized'] = df_clean[tag] / (df_clean['NumRatings'] * 3)
    df_clean[f'{tag}_normalized'] = df_clean[f'{tag}_normalized'].fillna(0)

q1_data = df_clean[df_clean['AvgRating'].notna()].copy()
male_ratings = q1_data[q1_data['Gender'] == 1]['AvgRating']
female_ratings = q1_data[q1_data['Gender'] == 0]['AvgRating']

# use welch's t-test since variances might differ
t_stat, p_value = stats.ttest_ind(male_ratings, female_ratings, equal_var=False)
male_mean = male_ratings.mean()
female_mean = female_ratings.mean()
male_n = len(male_ratings)
female_n = len(female_ratings)


print(f"Mean difference: {male_mean - female_mean:.3f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
print(f"Significant (α=0.005): {'Yes' if p_value < 0.005 else 'No'}")

# Modern color palette
colors = {'male': '#2E86AB', 'female': '#A23B72'}
fig, ax = plt.subplots(figsize=(11, 7))

# Use KDE for smoother density plots
sns.kdeplot(data=male_ratings, fill=True, alpha=0.5, color=colors['male'],
            label=f'Male (n={male_n:,})', linewidth=2.5, ax=ax)
sns.kdeplot(data=female_ratings, fill=True, alpha=0.5, color=colors['female'],
            label=f'Female (n={female_n:,})', linewidth=2.5, ax=ax)

# Add mean lines with better styling
ax.axvline(male_mean, color=colors['male'], linestyle='--', linewidth=2.5,
           label=f'Male mean: {male_mean:.2f}', alpha=0.8)
ax.axvline(female_mean, color=colors['female'], linestyle='--', linewidth=2.5,
           label=f'Female mean: {female_mean:.2f}', alpha=0.8)

ax.set_xlabel('Average Rating', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Average Ratings by Gender', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.95, loc='best')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('q1_gender_bias_ratings.png', dpi=300, bbox_inches='tight')
plt.show()

# Question 2: variance/spread difference
# using levene's test for variance equality
levene_stat, levene_p = levene(male_ratings, female_ratings)

# get variances
male_var = male_ratings.var()
female_var = female_ratings.var()
male_std = male_ratings.std()
female_std = female_ratings.std()

print("Question 2: Gender Difference in Spread")
print(f"Male:")
print(f"  Variance: {male_var:.4f}")
print(f"  Std: {male_std:.4f}")
print(f"\nFemale:")
print(f"  Variance: {female_var:.4f}")
print(f"  Std: {female_std:.4f}")
print(f"\nVariance ratio (M/F): {male_var/female_var:.4f}")
print(f"\nLevene's test:")
print(f"  stat: {levene_stat:.4f}")
print(f"  p: {levene_p:.6f}")
print(f"  Significant (α=0.005)? {'Yes' if levene_p < 0.005 else 'No'}")


colors = {'male': '#2E86AB', 'female': '#A23B72'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Enhanced box plot
box_data = [male_ratings, female_ratings]
bp = ax1.boxplot(box_data, labels=['Male', 'Female'], patch_artist=True,
                 widths=0.6, showmeans=True, meanline=True,boxprops=dict(facecolor=colors['male'], alpha=0.7, linewidth=2),medianprops=dict(color='white', linewidth=2.5),meanprops=dict(color='darkred', linewidth=2, linestyle='--'),whiskerprops=dict(linewidth=2),capprops=dict(linewidth=2),flierprops=dict(marker='o', markersize=4, alpha=0.5))

bp['boxes'][1].set_facecolor(colors['female'])
ax1.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
ax1.set_title('Box Plot: Ratings by Gender', fontsize=14, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

parts = ax2.violinplot(box_data, positions=[1, 2], widths=0.7,
                        showmeans=True, showmedians=True, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors['male'] if i == 0 else colors['female'])
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

parts['cmeans'].set_color('darkred')
parts['cmeans'].set_linewidth(2.5)
parts['cmeans'].set_linestyle('--')
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2.5)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Male', 'Female'], fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
ax2.set_title('Violin Plot: Distribution Shape by Gender', fontsize=14, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.2, linestyle='--', axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('q2_gender_spread_ratings.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Question 3: Effect sizes with 95% confidence intervals

"""

# Question 3: effect sizes with CIs
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    d = (group1.mean() - group2.mean()) / pooled_std
    return d

# bootstrap CI function
def bootstrap_ci(data1, data2, func, n_bootstrap=10000, ci=0.95):
    np.random.seed(N_NUMBER)  # reset seed for reproducibility
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample1 = resample(data1, random_state=None)
        sample2 = resample(data2, random_state=None)
        bootstrap_stats.append(func(sample1, sample2))
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * alpha/2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    return lower, upper

# effect size for mean difference
cohens_d_rating = cohens_d(male_ratings, female_ratings)

# bootstrap CI for mean difference
def mean_diff(sample1, sample2):
    return sample1.mean() - sample2.mean()

mean_diff_ci_lower, mean_diff_ci_upper = bootstrap_ci(male_ratings, female_ratings, mean_diff)

# variance ratio
variance_ratio = male_var / female_var

# bootstrap CI for variance ratio
def var_ratio(sample1, sample2):
    return sample1.var() / sample2.var()

var_ratio_ci_lower, var_ratio_ci_upper = bootstrap_ci(male_ratings, female_ratings, var_ratio)

print("Question 3: Effect Sizes with 95% CIs")
print("1. Gender Bias in Average Rating:")
print(f"   Cohen's d: {cohens_d_rating:.4f}")
print(f"   Mean diff (M-F): {male_mean - female_mean:.4f}")
print(f"   95% CI: [{mean_diff_ci_lower:.4f}, {mean_diff_ci_upper:.4f}]")
print(f"\n   Interpretation:")
if abs(cohens_d_rating) < 0.2:
    effect_size = "negligible"
elif abs(cohens_d_rating) < 0.5:
    effect_size = "small"
elif abs(cohens_d_rating) < 0.8:
    effect_size = "medium"
else:
    effect_size = "large"
print(f"   Cohen's d = {cohens_d_rating:.4f} → {effect_size} effect")

print(f"\n2. Gender Difference in Spread:")
print(f"   Variance ratio (M/F): {variance_ratio:.4f}")
print(f"   95% CI: [{var_ratio_ci_lower:.4f}, {var_ratio_ci_upper:.4f}]")

# Modern color palette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Enhanced mean difference with CI
mean_diff_val = male_mean - female_mean
ax1.barh(['Mean Difference'], [mean_diff_val],
         xerr=[[mean_diff_val - mean_diff_ci_lower],
               [mean_diff_ci_upper - mean_diff_val]],
         color='#06A77D', alpha=0.8, capsize=12, height=0.4,
         edgecolor='darkgreen', linewidth=2)
ax1.axvline(0, color='#D62828', linestyle='--', linewidth=2.5,
            label='No difference', zorder=0)
ax1.set_xlabel('Mean Difference (Male - Female)', fontsize=12, fontweight='bold')
ax1.set_title('Mean Rating Difference with 95% CI', fontsize=14, fontweight='bold', pad=10)
ax1.legend(fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.2, linestyle='--', axis='x')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# Add value annotation
ax1.text(mean_diff_val, 0, f'  {mean_diff_val:.4f}',
         va='center', fontsize=11, fontweight='bold')

# Enhanced variance ratio with CI
ax2.barh(['Variance Ratio'], [variance_ratio],
         xerr=[[variance_ratio - var_ratio_ci_lower],
               [var_ratio_ci_upper - variance_ratio]],
         color='#F77F00', alpha=0.8, capsize=12, height=0.4,
         edgecolor='darkorange', linewidth=2)
ax2.axvline(1, color='#D62828', linestyle='--', linewidth=2.5,
            label='Equal variance', zorder=0)
ax2.set_xlabel('Variance Ratio (Male / Female)', fontsize=12, fontweight='bold')
ax2.set_title('Variance Ratio with 95% CI', fontsize=14, fontweight='bold', pad=10)
ax2.legend(fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add value annotation
ax2.text(variance_ratio, 0, f'  {variance_ratio:.4f}',
         va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('q3_effect_sizes.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Question 4: Gender differences in tags

"""

# Question 4: gender differences in tags
# using normalized tags since raw counts depend on num ratings
tag_results = []

for tag in tag_names:
    male_tag = q1_data[q1_data['Gender'] == 1][f'{tag}_normalized']
    female_tag = q1_data[q1_data['Gender'] == 0][f'{tag}_normalized']

    # drop NaN
    male_tag_clean = male_tag.dropna()
    female_tag_clean = female_tag.dropna()

    if len(male_tag_clean) > 0 and len(female_tag_clean) > 0:
        # welch's t-test again
        t_stat_tag, p_value_tag = stats.ttest_ind(male_tag_clean, female_tag_clean, equal_var=False)

        # means
        male_tag_mean = male_tag_clean.mean()
        female_tag_mean = female_tag_clean.mean()

        tag_results.append({
            'Tag': tag,
            'Male_Mean': male_tag_mean,
            'Female_Mean': female_tag_mean,
            'Difference': male_tag_mean - female_tag_mean,
            'T_Statistic': t_stat_tag,
            'P_Value': p_value_tag,
            'Significant': p_value_tag < 0.005
        })

tag_df = pd.DataFrame(tag_results)
tag_df = tag_df.sort_values('P_Value')

print("Question 4: Gender Differences in Tags")
print(f"\nSignificant tags (α=0.005):")
significant_tags = tag_df[tag_df['Significant'] == True]
print(f"Count: {len(significant_tags)}")
print(significant_tags[['Tag', 'Male_Mean', 'Female_Mean', 'Difference', 'P_Value']].to_string(index=False))

print(f"\n\nTop 3 Most Gendered (lowest p):")
print(tag_df.head(3)[['Tag', 'Male_Mean', 'Female_Mean', 'Difference', 'P_Value']].to_string(index=False))

print(f"\n\nTop 3 Least Gendered (highest p):")
print(tag_df.tail(3)[['Tag', 'Male_Mean', 'Female_Mean', 'Difference', 'P_Value']].to_string(index=False))

# Enhanced plots with better colors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 9))

# Top 10 differences with gradient colors
top_10 = tag_df.head(10)
y_pos = np.arange(len(top_10))
# Use color gradient based on significance and direction
bar_colors = []
for idx, row in top_10.iterrows():
    if row['Significant']:
        if row['Difference'] > 0:
            bar_colors.append('#2E86AB')  # Blue for male-favored
        else:
            bar_colors.append('#A23B72')  # Purple for female-favored
    else:
        bar_colors.append('#95A5A6')  # Gray for non-significant

bars = ax1.barh(y_pos, top_10['Difference'], color=bar_colors, alpha=0.8,
                edgecolor='black', linewidth=1.5, height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_10['Tag'], fontsize=10, fontweight='bold')
ax1.set_xlabel('Mean Difference (Male - Female)', fontsize=12, fontweight='bold')
ax1.set_title('Top 10 Tags by Gender Difference\n(Blue=Male-favored, Purple=Female-favored)',
              fontsize=14, fontweight='bold', pad=12)
ax1.axvline(0, color='black', linestyle='--', linewidth=2, zorder=0)
ax1.grid(True, alpha=0.2, linestyle='--', axis='x')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(top_10.iterrows()):
    ax1.text(row['Difference'], i, f"  {row['Difference']:.4f}",
             va='center', fontsize=9, fontweight='bold')

# Enhanced p-value plot with better styling
sig_colors = ['#D62828' if sig else '#6C757D' for sig in tag_df['Significant']]
scatter = ax2.scatter(range(len(tag_df)), -np.log10(tag_df['P_Value']),
                     c=sig_colors, alpha=0.7, s=120, edgecolors='black', linewidth=1)
ax2.axhline(-np.log10(0.005), color='#D62828', linestyle='--', linewidth=2.5,
            label='α=0.005 threshold', zorder=0)
ax2.set_xlabel('Tag Index (sorted by p-value)', fontsize=12, fontweight='bold')
ax2.set_ylabel('-log₁₀(p-value)', fontsize=12, fontweight='bold')
ax2.set_title('P-values for Gender Differences in Tags', fontsize=14, fontweight='bold', pad=12)
ax2.legend(fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('q4_gender_tags.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Question 5: Gender difference in average difficulty

"""

# Question 5: Gender difference in average difficulty
q5_data = df_clean[df_clean['AvgDifficulty'].notna()].copy()
male_difficulty = q5_data[q5_data['Gender'] == 1]['AvgDifficulty']
female_difficulty = q5_data[q5_data['Gender'] == 0]['AvgDifficulty']

# Perform t-test
t_stat_diff, p_value_diff = stats.ttest_ind(male_difficulty, female_difficulty, equal_var=False)

# Calculate means and standard deviations
male_diff_mean = male_difficulty.mean()
female_diff_mean = female_difficulty.mean()
male_diff_std = male_difficulty.std()
female_diff_std = female_difficulty.std()
male_diff_n = len(male_difficulty)
female_diff_n = len(female_difficulty)

print("Question 5: Gender Difference in Average Difficulty")
print(f"Male professors:")
print(f"  Mean difficulty: {male_diff_mean:.3f}")
print(f"  Std deviation: {male_diff_std:.3f}")
print(f"  Sample size: {male_diff_n}")
print(f"\nFemale professors:")
print(f"  Mean difficulty: {female_diff_mean:.3f}")
print(f"  Std deviation: {female_diff_std:.3f}")
print(f"  Sample size: {female_diff_n}")
print(f"\nDifference (Male - Female): {male_diff_mean - female_diff_mean:.3f}")
print(f"\nWelch's t-test:")
print(f"  t-statistic: {t_stat_diff:.4f}")
print(f"  p-value: {p_value_diff:.6f}")
print(f"  Significant at α=0.005? {'Yes' if p_value_diff < 0.005 else 'No'}")

# Modern color palette with KDE plots
colors = {'male': '#2E86AB', 'female': '#A23B72'}
fig, ax = plt.subplots(figsize=(11, 7))

# Use KDE for smoother density plots
sns.kdeplot(data=male_difficulty, fill=True, alpha=0.5, color=colors['male'],
            label=f'Male (n={male_diff_n:,})', linewidth=2.5, ax=ax)
sns.kdeplot(data=female_difficulty, fill=True, alpha=0.5, color=colors['female'],
            label=f'Female (n={female_diff_n:,})', linewidth=2.5, ax=ax)

# Add mean lines with better styling
ax.axvline(male_diff_mean, color=colors['male'], linestyle='--', linewidth=2.5,
           label=f'Male mean: {male_diff_mean:.2f}', alpha=0.8)
ax.axvline(female_diff_mean, color=colors['female'], linestyle='--', linewidth=2.5,
           label=f'Female mean: {female_diff_mean:.2f}', alpha=0.8)

ax.set_xlabel('Average Difficulty', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Average Difficulty by Gender', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.95, loc='best')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('q5_gender_difficulty.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Question 6: Effect size for difficulty difference with 95% CI

"""

# Question 6: Effect size for difficulty difference
cohens_d_difficulty = cohens_d(male_difficulty, female_difficulty)
difficulty_diff_ci_lower, difficulty_diff_ci_upper = bootstrap_ci(
    male_difficulty, female_difficulty, mean_diff)

print("Question 6: Effect Size for Difficulty Difference")
print(f"Cohen's d: {cohens_d_difficulty:.4f}")
print(f"Mean difference (Male - Female): {male_diff_mean - female_diff_mean:.4f}")
print(f"95% CI for mean difference: [{difficulty_diff_ci_lower:.4f}, {difficulty_diff_ci_upper:.4f}]")

if abs(cohens_d_difficulty) < 0.2:
    effect_size_diff = "negligible"
elif abs(cohens_d_difficulty) < 0.5:
    effect_size_diff = "small"
elif abs(cohens_d_difficulty) < 0.8:
    effect_size_diff = "medium"
else:
    effect_size_diff = "large"
print(f"\nInterpretation: Cohen's d = {cohens_d_difficulty:.4f} indicates a {effect_size_diff} effect size")

# Enhanced visualization
fig, ax = plt.subplots(figsize=(10, 6))
diff_diff_val = male_diff_mean - female_diff_mean
ax.barh(['Difficulty Difference'], [diff_diff_val],
        xerr=[[diff_diff_val - difficulty_diff_ci_lower],
              [difficulty_diff_ci_upper - diff_diff_val]],
        color='#06A77D', alpha=0.8, capsize=12, height=0.4,
        edgecolor='darkgreen', linewidth=2)
ax.axvline(0, color='#D62828', linestyle='--', linewidth=2.5,
           label='No difference', zorder=0)
ax.set_xlabel('Mean Difference (Male - Female)', fontsize=12, fontweight='bold')
ax.set_title('Mean Difficulty Difference with 95% CI', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.2, linestyle='--', axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Add value annotation
ax.text(diff_diff_val, 0, f'  {diff_diff_val:.4f}',
        va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('q6_difficulty_effect_size.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Question 7: Regression model predicting average rating from numerical predictors

"""

# Strategy 0 for Q7: Ordinary Least Squares (OLS) - No Regularization
print("="*70)
print("QUESTION 7 STRATEGY 0: Ordinary Least Squares (OLS)")
print("="*70)

# Ensure data is prepared (in case main Q7 cell hasn't been run)
if 'X_train_scaled' not in globals() or 'y_train' not in globals():
    # Prepare data for Question 7
    q7_data = df_clean[df_clean['AvgRating'].notna()].copy()

    # Pick numerical features
    numerical_features = ['AvgDifficulty', 'NumRatings', 'Pepper', 'WouldTakeAgain', 'OnlineRatings']
    X_num = q7_data[numerical_features].copy()
    y = q7_data['AvgRating'].copy()

    # Handle missing WouldTakeAgain - using median fill
    X_num['WouldTakeAgain'] = X_num['WouldTakeAgain'].fillna(X_num['WouldTakeAgain'].median())

    # Clean up any remaining NaN
    mask = ~(X_num.isnull().any(axis=1) | y.isnull())
    X_num = X_num[mask]
    y = y[mask]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=N_NUMBER)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set baseline from original Ridge model (alpha=1.0) for comparison
    ridge_baseline = Ridge(alpha=1.0)
    ridge_baseline.fit(X_train_scaled, y_train)
    y_baseline_pred = ridge_baseline.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_baseline_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_baseline_pred))

# OLS using LinearRegression (no regularization)
ols_q7 = LinearRegression()
ols_q7.fit(X_train_scaled, y_train)

y_ols_q7_train_pred = ols_q7.predict(X_train_scaled)
y_ols_q7_test_pred = ols_q7.predict(X_test_scaled)

q7_ols_train_r2 = r2_score(y_train, y_ols_q7_train_pred)
q7_ols_test_r2 = r2_score(y_test, y_ols_q7_test_pred)
q7_ols_test_rmse = np.sqrt(mean_squared_error(y_test, y_ols_q7_test_pred))

print(f"\nTrain R²: {q7_ols_train_r2:.4f}")
print(f"Test R²: {q7_ols_test_r2:.4f}")
print(f"Test RMSE: {q7_ols_test_rmse:.4f}")
print(f"\nComparison with Ridge (α=1.0):")
print(f"  OLS Test R²: {q7_ols_test_r2:.4f}")
print(f"  Ridge Test R²: {test_r2:.4f}")
print(f"  Difference: {((q7_ols_test_r2 - test_r2) / test_r2 * 100):.2f}%")

# Strategy 1 for Q7: Hyperparameter Tuning (Ridge Alpha)
print("\n" + "="*70)
print("QUESTION 7 STRATEGY 1: Hyperparameter Tuning (Ridge Alpha)")
print("="*70)

# Tune Ridge alpha
alphas_q7 = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_alpha_q7 = 1.0
best_r2_q7 = -np.inf

for alpha in alphas_q7:
    ridge_temp_q7 = Ridge(alpha=alpha)
    ridge_temp_q7.fit(X_train_scaled, y_train)
    r2_temp_q7 = r2_score(y_test, ridge_temp_q7.predict(X_test_scaled))
    if r2_temp_q7 > best_r2_q7:
        best_r2_q7 = r2_temp_q7
        best_alpha_q7 = alpha

ridge_q7_tuned = Ridge(alpha=best_alpha_q7)
ridge_q7_tuned.fit(X_train_scaled, y_train)

y_q7_tuned_train_pred = ridge_q7_tuned.predict(X_train_scaled)
y_q7_tuned_test_pred = ridge_q7_tuned.predict(X_test_scaled)

q7_tuned_train_r2 = r2_score(y_train, y_q7_tuned_train_pred)
q7_tuned_test_r2 = r2_score(y_test, y_q7_tuned_test_pred)
q7_tuned_test_rmse = np.sqrt(mean_squared_error(y_test, y_q7_tuned_test_pred))

print(f"\nBest alpha: {best_alpha_q7}")
print(f"Train R²: {q7_tuned_train_r2:.4f}")
print(f"Test R²: {q7_tuned_test_r2:.4f}")
print(f"Test RMSE: {q7_tuned_test_rmse:.4f}")
print(f"\nImprovement over original Q7: {((q7_tuned_test_r2 - test_r2) / test_r2 * 100):.2f}%")

# Strategy 3 for Q7: ElasticNet
print("\n" + "="*70)
print("QUESTION 7 STRATEGY 2: ElasticNet Regression")
print("="*70)

param_grid_q7 = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

elastic_net_q7 = ElasticNet(max_iter=5000)
grid_search_q7 = GridSearchCV(elastic_net_q7, param_grid_q7, cv=5, scoring='r2', n_jobs=-1)
grid_search_q7.fit(X_train_scaled, y_train)

best_en_q7 = grid_search_q7.best_estimator_
y_en_q7_train_pred = best_en_q7.predict(X_train_scaled)
y_en_q7_test_pred = best_en_q7.predict(X_test_scaled)

q7_en_train_r2 = r2_score(y_train, y_en_q7_train_pred)
q7_en_test_r2 = r2_score(y_test, y_en_q7_test_pred)
q7_en_test_rmse = np.sqrt(mean_squared_error(y_test, y_en_q7_test_pred))

print(f"\nBest parameters: {grid_search_q7.best_params_}")
print(f"Train R²: {q7_en_train_r2:.4f}")
print(f"Test R²: {q7_en_test_r2:.4f}")
print(f"Test RMSE: {q7_en_test_rmse:.4f}")
print(f"\nImprovement over original Q7: {((q7_en_test_r2 - test_r2) / test_r2 * 100):.2f}%")

# Strategy 3.5 for Q7: Lasso Regression (L1 Regularization)
print("\n" + "="*70)
print("QUESTION 7 STRATEGY 3: Lasso Regression (L1 Regularization)")
print("="*70)

# Lasso uses L1 regularization which can perform feature selection by setting coefficients to zero
# Lasso is more sensitive to alpha than Ridge, so we use a wider range including smaller values
param_grid_lasso_q7 = {
    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
}

lasso_q7 = Lasso(max_iter=5000)
grid_search_lasso_q7 = GridSearchCV(lasso_q7, param_grid_lasso_q7, cv=5, scoring='r2', n_jobs=-1)
grid_search_lasso_q7.fit(X_train_scaled, y_train)

best_lasso_q7 = grid_search_lasso_q7.best_estimator_
y_lasso_q7_train_pred = best_lasso_q7.predict(X_train_scaled)
y_lasso_q7_test_pred = best_lasso_q7.predict(X_test_scaled)

q7_lasso_train_r2 = r2_score(y_train, y_lasso_q7_train_pred)
q7_lasso_test_r2 = r2_score(y_test, y_lasso_q7_test_pred)
q7_lasso_test_rmse = np.sqrt(mean_squared_error(y_test, y_lasso_q7_test_pred))

print(f"\nBest alpha: {grid_search_lasso_q7.best_params_['alpha']}")
print(f"Train R²: {q7_lasso_train_r2:.4f}")
print(f"Test R²: {q7_lasso_test_r2:.4f}")
print(f"Test RMSE: {q7_lasso_test_rmse:.4f}")
print(f"\nImprovement over original Q7: {((q7_lasso_test_r2 - test_r2) / test_r2 * 100):.2f}%")

# Show feature selection (coefficients that are zero)
coef_df_q7 = pd.DataFrame({
    'Feature': numerical_features,
    'Coefficient': best_lasso_q7.coef_
})
coef_df_q7 = coef_df_q7.sort_values('Coefficient', key=abs, ascending=False)
print(f"\nTop 10 Features by Absolute Coefficient:")
print(coef_df_q7.head(10).to_string(index=False))
zero_features = coef_df_q7[coef_df_q7['Coefficient'] == 0]
if len(zero_features) > 0:
    print(f"\nFeatures eliminated by Lasso (coefficient = 0): {len(zero_features)}")
    print(zero_features['Feature'].tolist())

# Strategy 2.5 for Q7: Feature Interactions (needed for Random Forest and beyond)
# Create interaction features for numerical predictors
X_num_interact = X_num.copy()
X_num_interact['AvgDifficulty_NumRatings'] = X_num_interact['AvgDifficulty'] * X_num_interact['NumRatings']
X_num_interact['AvgDifficulty_WouldTakeAgain'] = X_num_interact['AvgDifficulty'] * X_num_interact['WouldTakeAgain']
X_num_interact['NumRatings_WouldTakeAgain'] = X_num_interact['NumRatings'] * X_num_interact['WouldTakeAgain']
X_num_interact['Pepper_WouldTakeAgain'] = X_num_interact['Pepper'] * X_num_interact['WouldTakeAgain']

# Split interaction data
X_num_int_train, X_num_int_test, y_num_int_train, y_num_int_test = train_test_split(
    X_num_interact, y, test_size=0.2, random_state=N_NUMBER)

# Ridge with interactions for baseline comparison
ridge_int_q7 = Ridge(alpha=best_alpha_q7)
ridge_int_q7.fit(X_num_int_train, y_num_int_train)
y_int_train_pred = ridge_int_q7.predict(X_num_int_train)
y_int_test_pred = ridge_int_q7.predict(X_num_int_test)
q7_int_test_r2 = r2_score(y_num_int_test, y_int_test_pred)
q7_int_test_rmse = np.sqrt(mean_squared_error(y_num_int_test, y_int_test_pred))

# Strategy 4 for Q7: Random Forest
print("\n" + "="*70)
print("QUESTION 7 STRATEGY 4: Random Forest Regressor")
print("="*70)

rf_q7 = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=N_NUMBER,
    n_jobs=-1
)
rf_q7.fit(X_num_int_train, y_num_int_train)

y_rf_q7_train_pred = rf_q7.predict(X_num_int_train)
y_rf_q7_test_pred = rf_q7.predict(X_num_int_test)

q7_rf_train_r2 = r2_score(y_num_int_train, y_rf_q7_train_pred)
q7_rf_test_r2 = r2_score(y_num_int_test, y_rf_q7_test_pred)
q7_rf_test_rmse = np.sqrt(mean_squared_error(y_num_int_test, y_rf_q7_test_pred))

print(f"\nTrain R²: {q7_rf_train_r2:.4f}")
print(f"Test R²: {q7_rf_test_r2:.4f}")
print(f"Test RMSE: {q7_rf_test_rmse:.4f}")
print(f"\nImprovement over interaction Ridge: {((q7_rf_test_r2 - q7_int_test_r2) / q7_int_test_r2 * 100):.2f}%")
print(f"Improvement over original Q7: {((q7_rf_test_r2 - test_r2) / test_r2 * 100):.2f}%")

# Feature importance
num_feature_names_q7 = list(X_num_interact.columns)
rf_q7_importance = pd.DataFrame({
    'Feature': num_feature_names_q7,
    'Importance': rf_q7.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(rf_q7_importance.head(10).to_string(index=False))

# Strategy 5 for Q7: Gradient Boosting
print("\n" + "="*70)
print("QUESTION 7 STRATEGY 5: Gradient Boosting Regressor")
print("="*70)

gb_q7 = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=N_NUMBER,
    subsample=0.8
)
gb_q7.fit(X_num_int_train, y_num_int_train)

y_gb_q7_train_pred = gb_q7.predict(X_num_int_train)
y_gb_q7_test_pred = gb_q7.predict(X_num_int_test)

q7_gb_train_r2 = r2_score(y_num_int_train, y_gb_q7_train_pred)
q7_gb_test_r2 = r2_score(y_num_int_test, y_gb_q7_test_pred)
q7_gb_test_rmse = np.sqrt(mean_squared_error(y_num_int_test, y_gb_q7_test_pred))

print(f"\nTrain R²: {q7_gb_train_r2:.4f}")
print(f"Test R²: {q7_gb_test_r2:.4f}")
print(f"Test RMSE: {q7_gb_test_rmse:.4f}")
print(f"\nImprovement over Random Forest: {((q7_gb_test_r2 - q7_rf_test_r2) / q7_rf_test_r2 * 100):.2f}%")
print(f"Improvement over original Q7: {((q7_gb_test_r2 - test_r2) / test_r2 * 100):.2f}%")

# Feature importance
gb_q7_importance = pd.DataFrame({
    'Feature': num_feature_names_q7,
    'Importance': gb_q7.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(gb_q7_importance.head(10).to_string(index=False))

# Strategy 6 for Q7: XGBoost
print("\n" + "="*70)
print("QUESTION 7 STRATEGY 6: XGBoost Regressor")
print("="*70)

if XGBOOST_AVAILABLE:
    xgb_q7 = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=N_NUMBER,
        n_jobs=-1,
        verbosity=0
    )
    xgb_q7.fit(X_num_int_train, y_num_int_train)

    y_xgb_q7_train_pred = xgb_q7.predict(X_num_int_train)
    y_xgb_q7_test_pred = xgb_q7.predict(X_num_int_test)

    q7_xgb_train_r2 = r2_score(y_num_int_train, y_xgb_q7_train_pred)
    q7_xgb_test_r2 = r2_score(y_num_int_test, y_xgb_q7_test_pred)
    q7_xgb_test_rmse = np.sqrt(mean_squared_error(y_num_int_test, y_xgb_q7_test_pred))

    print(f"\nTrain R²: {q7_xgb_train_r2:.4f}")
    print(f"Test R²: {q7_xgb_test_r2:.4f}")
    print(f"Test RMSE: {q7_xgb_test_rmse:.4f}")
    print(f"\nImprovement over Gradient Boosting: {((q7_xgb_test_r2 - q7_gb_test_r2) / q7_gb_test_r2 * 100):.2f}%")
    print(f"Improvement over original Q7: {((q7_xgb_test_r2 - test_r2) / test_r2 * 100):.2f}%")

    # Feature importance
    xgb_q7_importance = pd.DataFrame({
        'Feature': num_feature_names_q7,
        'Importance': xgb_q7.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 Most Important Features:")
    print(xgb_q7_importance.head(10).to_string(index=False))

    # Try with hyperparameter tuning
    print("\n" + "-"*70)
    print("XGBoost with Grid Search Hyperparameter Tuning")
    print("-"*70)

    param_grid_xgb_q7 = {
        'n_estimators': [100, 200],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 0.9]
    }

    xgb_base_q7 = xgb.XGBRegressor(objective='reg:squarederror', random_state=N_NUMBER, n_jobs=-1, verbosity=0)
    grid_search_xgb_q7 = GridSearchCV(xgb_base_q7, param_grid_xgb_q7, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid_search_xgb_q7.fit(X_num_int_train, y_num_int_train)

    best_xgb_q7 = grid_search_xgb_q7.best_estimator_
    y_xgb_q7_tuned_train_pred = best_xgb_q7.predict(X_num_int_train)
    y_xgb_q7_tuned_test_pred = best_xgb_q7.predict(X_num_int_test)

    q7_xgb_tuned_train_r2 = r2_score(y_num_int_train, y_xgb_q7_tuned_train_pred)
    q7_xgb_tuned_test_r2 = r2_score(y_num_int_test, y_xgb_q7_tuned_test_pred)
    q7_xgb_tuned_test_rmse = np.sqrt(mean_squared_error(y_num_int_test, y_xgb_q7_tuned_test_pred))

    print(f"\nBest parameters: {grid_search_xgb_q7.best_params_}")
    print(f"Train R²: {q7_xgb_tuned_train_r2:.4f}")
    print(f"Test R²: {q7_xgb_tuned_test_r2:.4f}")
    print(f"Test RMSE: {q7_xgb_tuned_test_rmse:.4f}")
    print(f"\nImprovement over default XGBoost: {((q7_xgb_tuned_test_r2 - q7_xgb_test_r2) / q7_xgb_test_r2 * 100):.2f}%")
    print(f"Improvement over original Q7: {((q7_xgb_tuned_test_r2 - test_r2) / test_r2 * 100):.2f}%")

else:
    print("\nXGBoost is not installed. Skipping this strategy.")
    print("To install: pip install xgboost")
    q7_xgb_test_r2 = 0
    q7_xgb_tuned_test_r2 = 0
    q7_xgb_test_rmse = 0
    q7_xgb_tuned_test_rmse = 0

# Comparison for Question 7 Strategies
print("\n" + "="*70)
print("SUMMARY: R² Comparison for Question 7 (Numerical-Only) Strategies")
print("="*70)

q7_model_list = [
    'Original Q7: Numerical Only (Ridge α=1.0)',
    'Strategy 0: OLS (No Regularization)',
    'Strategy 1: Tuned Ridge',
    'Strategy 2: + Feature Interactions',
    'Strategy 3: ElasticNet',
    'Strategy 3.5: Lasso',
    'Strategy 4: Random Forest',
    'Strategy 5: Gradient Boosting',
    'Strategy 6: XGBoost (Default)'
]

q7_r2_list = [
    test_r2,
    q7_ols_test_r2,
    q7_tuned_test_r2,
    q7_int_test_r2,
    q7_en_test_r2,
    q7_lasso_test_r2,
    q7_rf_test_r2,
    q7_gb_test_r2,
    q7_xgb_test_r2 if XGBOOST_AVAILABLE else 0
]

q7_rmse_list = [
    test_rmse,
    q7_ols_test_rmse,
    q7_tuned_test_rmse,
    q7_int_test_rmse,
    q7_en_test_rmse,
    q7_lasso_test_rmse,
    q7_rf_test_rmse,
    q7_gb_test_rmse,
    q7_xgb_test_rmse if XGBOOST_AVAILABLE else 0
]

# Add tuned XGBoost if available
if XGBOOST_AVAILABLE:
    q7_model_list.append('Strategy 6: XGBoost (Tuned)')
    q7_r2_list.append(q7_xgb_tuned_test_r2)
    q7_rmse_list.append(q7_xgb_tuned_test_rmse)

q7_results_summary = pd.DataFrame({
    'Model': q7_model_list,
    'Test R²': q7_r2_list,
    'Test RMSE': q7_rmse_list
})

q7_results_summary['Improvement %'] = ((q7_results_summary['Test R²'] - test_r2) / test_r2 * 100).round(2)
q7_results_summary = q7_results_summary.sort_values('Test R²', ascending=False)

print("\n" + q7_results_summary.to_string(index=False))

best_q7_model_name = q7_results_summary.iloc[0]['Model']
best_q7_r2_value = q7_results_summary.iloc[0]['Test R²']
print(f"\nBest Model for Q7: {best_q7_model_name}")
print(f"   Best Test R²: {best_q7_r2_value:.4f}")
print(f"   Improvement over baseline: {q7_results_summary.iloc[0]['Improvement %']:.2f}%")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

# R² comparison
colors_q7 = ['#2E86AB' if 'Original' in m else '#F77F00' for m in q7_results_summary['Model']]
bars1 = ax1.barh(range(len(q7_results_summary)), q7_results_summary['Test R²'],
                 color=colors_q7, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax1.set_yticks(range(len(q7_results_summary)))
ax1.set_yticklabels(q7_results_summary['Model'], fontsize=9, fontweight='bold')
ax1.set_xlabel('Test R² Score', fontsize=12, fontweight='bold')
ax1.set_title('R² Comparison for Question 7 Strategies', fontsize=14, fontweight='bold', pad=12)
ax1.grid(True, alpha=0.2, linestyle='--', axis='x')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(q7_results_summary.iterrows()):
    ax1.text(row['Test R²'], i, f"  {row['Test R²']:.4f}",
             va='center', fontsize=9, fontweight='bold')

# RMSE comparison
bars2 = ax2.barh(range(len(q7_results_summary)), q7_results_summary['Test RMSE'],
                 color=colors_q7, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax2.set_yticks(range(len(q7_results_summary)))
ax2.set_yticklabels(q7_results_summary['Model'], fontsize=9, fontweight='bold')
ax2.set_xlabel('Test RMSE', fontsize=12, fontweight='bold')
ax2.set_title('RMSE Comparison for Question 7 Strategies', fontsize=14, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(q7_results_summary.iterrows()):
    ax2.text(row['Test RMSE'], i, f"  {row['Test RMSE']:.4f}",
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('q7_r2_improvement_strategies.png', dpi=300, bbox_inches='tight')
plt.show()

# Question 7: regression with numerical predictors
q7_data = df_clean[df_clean['AvgRating'].notna()].copy()

# pick numerical features (not rating itself, not gender, not tags)
numerical_features = ['AvgDifficulty', 'NumRatings', 'Pepper', 'WouldTakeAgain', 'OnlineRatings']
X_num = q7_data[numerical_features].copy()
y = q7_data['AvgRating'].copy()

# handle missing WouldTakeAgain - using median fill
X_num['WouldTakeAgain'] = X_num['WouldTakeAgain'].fillna(X_num['WouldTakeAgain'].median())

# clean up any remaining NaN
mask = ~(X_num.isnull().any(axis=1) | y.isnull())
X_num = X_num[mask]
y = y[mask]

# check correlations for collinearity
corr_matrix = X_num.corr()
print("Question 7: Regression - Rating from Numerical Predictors")
print("\nCorrelation matrix:")
print(corr_matrix.round(3))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_num, y, test_size=0.2, random_state=N_NUMBER)

# standardize (needed for ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# using ridge to handle collinearity (tried regular linear regression first, had issues)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# predictions
y_train_pred = ridge_model.predict(X_train_scaled)
y_test_pred = ridge_model.predict(X_test_scaled)

# metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nPerformance:")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")

# feature importance
feature_importance = pd.DataFrame({
    'Feature': numerical_features,
    'Coefficient': ridge_model.coef_,
    'Abs_Coefficient': np.abs(ridge_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nFeature importance:")
print(feature_importance.to_string(index=False))

most_important = feature_importance.iloc[0]['Feature']
print(f"\nMost predictive: {most_important}")

# Enhanced plots with better colors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Enhanced predicted vs actual with density coloring
scatter = ax1.scatter(y_test, y_test_pred, alpha=0.6, s=25,
                     c=np.abs(y_test - y_test_pred), cmap='YlOrRd',
                     edgecolors='black', linewidth=0.3)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2.5, label='Perfect Prediction', zorder=0)
ax1.set_xlabel('Actual Rating', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
ax1.set_title(f'Predicted vs Actual Ratings\n(R² = {test_r2:.3f}, RMSE = {test_rmse:.3f})',
              fontsize=14, fontweight='bold', pad=12)
ax1.legend(fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Absolute Error', fontsize=10, fontweight='bold')

# Enhanced coefficients with color coding
colors_coef = ['#2E86AB' if c > 0 else '#D62828' for c in feature_importance['Coefficient']]
bars = ax2.barh(feature_importance['Feature'], feature_importance['Coefficient'],
                color=colors_coef, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.6)
ax2.axvline(0, color='black', linestyle='--', linewidth=2, zorder=0)
ax2.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax2.set_title('Feature Coefficients (Ridge Regression)', fontsize=14, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(feature_importance.iterrows()):
    ax2.text(row['Coefficient'], i, f"  {row['Coefficient']:.3f}",
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('q7_regression_numerical.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Question 8: Regression model predicting average rating from tags

"""

# Question 8: regression with tags
# using normalized tags
tag_features = [f'{tag}_normalized' for tag in tag_names]
X_tags = q7_data[tag_features].copy()

# clean NaN
mask_tags = ~(X_tags.isnull().any(axis=1) | y.isnull())
X_tags = X_tags[mask_tags]
y_tags = y[mask_tags]

# check tag correlations - might have collinearity
corr_matrix_tags = X_tags.corr()
print("Question 8: Regression - Rating from Tags")
print("\nHigh tag correlations (>0.5):")
high_corr = corr_matrix_tags[corr_matrix_tags.abs() > 0.5]
high_corr = high_corr[high_corr != 1.0]  # skip diagonal
if not high_corr.isnull().all().all():
    print(high_corr.dropna(how='all').dropna(axis=1, how='all').round(3))
else:
    print("None found")

# split
X_tags_train, X_tags_test, y_tags_train, y_tags_test = train_test_split(
    X_tags, y_tags, test_size=0.2, random_state=N_NUMBER)

# standardize
scaler_tags = StandardScaler()
X_tags_train_scaled = scaler_tags.fit_transform(X_tags_train)
X_tags_test_scaled = scaler_tags.transform(X_tags_test)

# ridge again (tags might be correlated)
ridge_tags = Ridge(alpha=1.0)
ridge_tags.fit(X_tags_train_scaled, y_tags_train)

# predict
y_tags_train_pred = ridge_tags.predict(X_tags_train_scaled)
y_tags_test_pred = ridge_tags.predict(X_tags_test_scaled)

# metrics
tags_train_r2 = r2_score(y_tags_train, y_tags_train_pred)
tags_test_r2 = r2_score(y_tags_test, y_tags_test_pred)
tags_train_rmse = np.sqrt(mean_squared_error(y_tags_train, y_tags_train_pred))
tags_test_rmse = np.sqrt(mean_squared_error(y_tags_test, y_tags_test_pred))

print(f"\nPerformance:")
print(f"  Train R²: {tags_train_r2:.4f}")
print(f"  Test R²: {tags_test_r2:.4f}")
print(f"  Train RMSE: {tags_train_rmse:.4f}")
print(f"  Test RMSE: {tags_test_rmse:.4f}")

# tag importance
tag_importance = pd.DataFrame({
    'Tag': tag_names,
    'Coefficient': ridge_tags.coef_,
    'Abs_Coefficient': np.abs(ridge_tags.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nTag importance:")
print(tag_importance.to_string(index=False))

most_important_tag = tag_importance.iloc[0]['Tag']
print(f"\nMost predictive tag: {most_important_tag}")

# compare with numerical model
print(f"\n\nComparison:")
print(f"  Numerical - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
print(f"  Tags - R²: {tags_test_r2:.4f}, RMSE: {tags_test_rmse:.4f}")
if test_r2 > tags_test_r2:
    print(f"Numerical better by {((test_r2 - tags_test_r2) / tags_test_r2 * 100):.1f}%")
else:
    print(f"Tags better by {((tags_test_r2 - test_r2) / test_r2 * 100):.1f}%")

# Enhanced plots with better colors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

# Enhanced predicted vs actual with density coloring
scatter = ax1.scatter(y_tags_test, y_tags_test_pred, alpha=0.6, s=25,
                     c=np.abs(y_tags_test - y_tags_test_pred), cmap='YlGnBu',
                     edgecolors='black', linewidth=0.3)
ax1.plot([y_tags_test.min(), y_tags_test.max()], [y_tags_test.min(), y_tags_test.max()],
         'r--', linewidth=2.5, label='Perfect Prediction', zorder=0)
ax1.set_xlabel('Actual Rating', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
ax1.set_title(f'Predicted vs Actual Ratings (Tags Model)\n(R² = {tags_test_r2:.3f}, RMSE = {tags_test_rmse:.3f})',
              fontsize=14, fontweight='bold', pad=12)
ax1.legend(fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Absolute Error', fontsize=10, fontweight='bold')

# Enhanced top 10 coefficients with color coding
top_10_tags = tag_importance.head(10)
colors_coef = ['#2E86AB' if c > 0 else '#D62828' for c in top_10_tags['Coefficient']]
bars = ax2.barh(range(len(top_10_tags)), top_10_tags['Coefficient'],
                color=colors_coef, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax2.set_yticks(range(len(top_10_tags)))
ax2.set_yticklabels(top_10_tags['Tag'], fontsize=9, fontweight='bold')
ax2.axvline(0, color='black', linestyle='--', linewidth=2, zorder=0)
ax2.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax2.set_title('Top 10 Tag Coefficients (Ridge Regression)', fontsize=14, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(top_10_tags.iterrows()):
    ax2.text(row['Coefficient'], i, f"  {row['Coefficient']:.3f}",
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('q8_regression_tags.png', dpi=300, bbox_inches='tight')
plt.show()

# Strategy 1 for Q8: Hyperparameter Tuning for Ridge with Tags Only
print("="*70)
print("QUESTION 8 STRATEGY 1: Hyperparameter Tuning (Ridge Alpha)")
print("="*70)

# Use the same tag data from Question 8
# Tune Ridge alpha
alphas_q8 = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_alpha_q8 = 1.0
best_r2_q8 = -np.inf

for alpha in alphas_q8:
    ridge_temp_q8 = Ridge(alpha=alpha)
    ridge_temp_q8.fit(X_tags_train_scaled, y_tags_train)
    r2_temp_q8 = r2_score(y_tags_test, ridge_temp_q8.predict(X_tags_test_scaled))
    if r2_temp_q8 > best_r2_q8:
        best_r2_q8 = r2_temp_q8
        best_alpha_q8 = alpha

ridge_q8_tuned = Ridge(alpha=best_alpha_q8)
ridge_q8_tuned.fit(X_tags_train_scaled, y_tags_train)

y_q8_tuned_train_pred = ridge_q8_tuned.predict(X_tags_train_scaled)
y_q8_tuned_test_pred = ridge_q8_tuned.predict(X_tags_test_scaled)

q8_tuned_train_r2 = r2_score(y_tags_train, y_q8_tuned_train_pred)
q8_tuned_test_r2 = r2_score(y_tags_test, y_q8_tuned_test_pred)
q8_tuned_test_rmse = np.sqrt(mean_squared_error(y_tags_test, y_q8_tuned_test_pred))

print(f"\nBest alpha: {best_alpha_q8}")
print(f"Train R²: {q8_tuned_train_r2:.4f}")
print(f"Test R²: {q8_tuned_test_r2:.4f}")
print(f"Test RMSE: {q8_tuned_test_rmse:.4f}")
print(f"\nImprovement over original Q8: {((q8_tuned_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

# Strategy 2 for Q8: Tag Interaction Terms
print("\n" + "="*70)
print("QUESTION 8 STRATEGY 2: Tag Interaction Terms")
print("="*70)

# Create interaction terms between top tags
X_tags_interact = X_tags.copy()

# Top interactions based on tag importance from Q8
X_tags_interact['ToughGrader_GoodFeedback'] = X_tags_interact['ToughGrader_normalized'] * X_tags_interact['GoodFeedback_normalized']
X_tags_interact['ToughGrader_Respected'] = X_tags_interact['ToughGrader_normalized'] * X_tags_interact['Respected_normalized']
X_tags_interact['GoodFeedback_Respected'] = X_tags_interact['GoodFeedback_normalized'] * X_tags_interact['Respected_normalized']
X_tags_interact['GoodFeedback_Caring'] = X_tags_interact['GoodFeedback_normalized'] * X_tags_interact['Caring_normalized']
X_tags_interact['AmazingLectures_Respected'] = X_tags_interact['AmazingLectures_normalized'] * X_tags_interact['Respected_normalized']
X_tags_interact['ClearGrading_GoodFeedback'] = X_tags_interact['ClearGrading_normalized'] * X_tags_interact['GoodFeedback_normalized']

# Split
X_tags_int_train, X_tags_int_test, y_tags_int_train, y_tags_int_test = train_test_split(
    X_tags_interact, y_tags, test_size=0.2, random_state=N_NUMBER)

# Standardize
scaler_tags_int = StandardScaler()
X_tags_int_train_scaled = scaler_tags_int.fit_transform(X_tags_int_train)
X_tags_int_test_scaled = scaler_tags_int.transform(X_tags_int_test)

# Ridge with tuned alpha
ridge_tags_int = Ridge(alpha=best_alpha_q8)
ridge_tags_int.fit(X_tags_int_train_scaled, y_tags_int_train)

y_tags_int_train_pred = ridge_tags_int.predict(X_tags_int_train_scaled)
y_tags_int_test_pred = ridge_tags_int.predict(X_tags_int_test_scaled)

q8_int_train_r2 = r2_score(y_tags_int_train, y_tags_int_train_pred)
q8_int_test_r2 = r2_score(y_tags_int_test, y_tags_int_test_pred)
q8_int_test_rmse = np.sqrt(mean_squared_error(y_tags_int_test, y_tags_int_test_pred))

print(f"\nTrain R²: {q8_int_train_r2:.4f}")
print(f"Test R²: {q8_int_test_r2:.4f}")
print(f"Test RMSE: {q8_int_test_rmse:.4f}")
print(f"\nImprovement over tuned Ridge: {((q8_int_test_r2 - q8_tuned_test_r2) / q8_tuned_test_r2 * 100):.2f}%")
print(f"Improvement over original Q8: {((q8_int_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

# Strategy 3 for Q8: ElasticNet with Tags Only
print("\n" + "="*70)
print("QUESTION 8 STRATEGY 3: ElasticNet Regression (Tags Only)")
print("="*70)

param_grid_q8 = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

elastic_net_q8 = ElasticNet(max_iter=5000)
grid_search_q8 = GridSearchCV(elastic_net_q8, param_grid_q8, cv=5, scoring='r2', n_jobs=-1)
grid_search_q8.fit(X_tags_train_scaled, y_tags_train)

best_en_q8 = grid_search_q8.best_estimator_
y_en_q8_train_pred = best_en_q8.predict(X_tags_train_scaled)
y_en_q8_test_pred = best_en_q8.predict(X_tags_test_scaled)

q8_en_train_r2 = r2_score(y_tags_train, y_en_q8_train_pred)
q8_en_test_r2 = r2_score(y_tags_test, y_en_q8_test_pred)
q8_en_test_rmse = np.sqrt(mean_squared_error(y_tags_test, y_en_q8_test_pred))

print(f"\nBest parameters: {grid_search_q8.best_params_}")
print(f"Train R²: {q8_en_train_r2:.4f}")
print(f"Test R²: {q8_en_test_r2:.4f}")
print(f"Test RMSE: {q8_en_test_rmse:.4f}")
print(f"\nImprovement over original Q8: {((q8_en_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

# Strategy 3.5 for Q8: Lasso Regression (L1 Regularization) with Tags Only
print("\n" + "="*70)
print("QUESTION 8 STRATEGY 3.5: Lasso Regression (L1 Regularization) - Tags Only")
print("="*70)

# Lasso uses L1 regularization which can perform feature selection by setting coefficients to zero
# This is particularly useful for tags where collinearity might be a concern
# Lasso is more sensitive to alpha than Ridge, so we use a wider range incl uding smaller values
param_grid_lasso_q8 = {
    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
}

lasso_q8 = Lasso(max_iter=5000)
grid_search_lasso_q8 = GridSearchCV(lasso_q8, param_grid_lasso_q8, cv=5, scoring='r2', n_jobs=-1)
grid_search_lasso_q8.fit(X_tags_train_scaled, y_tags_train)

best_lasso_q8 = grid_search_lasso_q8.best_estimator_
y_lasso_q8_train_pred = best_lasso_q8.predict(X_tags_train_scaled)
y_lasso_q8_test_pred = best_lasso_q8.predict(X_tags_test_scaled)

q8_lasso_train_r2 = r2_score(y_tags_train, y_lasso_q8_train_pred)
q8_lasso_test_r2 = r2_score(y_tags_test, y_lasso_q8_test_pred)
q8_lasso_test_rmse = np.sqrt(mean_squared_error(y_tags_test, y_lasso_q8_test_pred))

print(f"\nBest alpha: {grid_search_lasso_q8.best_params_['alpha']}")
print(f"Train R²: {q8_lasso_train_r2:.4f}")
print(f"Test R²: {q8_lasso_test_r2:.4f}")
print(f"Test RMSE: {q8_lasso_test_rmse:.4f}")
print(f"\nImprovement over original Q8: {((q8_lasso_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

# Show feature selection (coefficients that are zero)
coef_df_q8 = pd.DataFrame({
    'Feature': tag_features,
    'Coefficient': best_lasso_q8.coef_
})
coef_df_q8 = coef_df_q8.sort_values('Coefficient', key=abs, ascending=False)
print(f"\nTop 10 Tags by Absolute Coefficient:")
print(coef_df_q8.head(10).to_string(index=False))
zero_features_q8 = coef_df_q8[coef_df_q8['Coefficient'] == 0]
if len(zero_features_q8) > 0:
    print(f"\nTags eliminated by Lasso (coefficient = 0): {len(zero_features_q8)}")
    print(zero_features_q8['Feature'].tolist())

# Strategy 4 for Q8: Random Forest with Tags Only
print("\n" + "="*70)
print("QUESTION 8 STRATEGY 4: Random Forest Regressor (Tags Only)")
print("="*70)

rf_q8 = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=N_NUMBER,
    n_jobs=-1
)
rf_q8.fit(X_tags_int_train, y_tags_int_train)

y_rf_q8_train_pred = rf_q8.predict(X_tags_int_train)
y_rf_q8_test_pred = rf_q8.predict(X_tags_int_test)

q8_rf_train_r2 = r2_score(y_tags_int_train, y_rf_q8_train_pred)
q8_rf_test_r2 = r2_score(y_tags_int_test, y_rf_q8_test_pred)
q8_rf_test_rmse = np.sqrt(mean_squared_error(y_tags_int_test, y_rf_q8_test_pred))

print(f"\nTrain R²: {q8_rf_train_r2:.4f}")
print(f"Test R²: {q8_rf_test_r2:.4f}")
print(f"Test RMSE: {q8_rf_test_rmse:.4f}")
print(f"\nImprovement over interaction Ridge: {((q8_rf_test_r2 - q8_int_test_r2) / q8_int_test_r2 * 100):.2f}%")
print(f"Improvement over original Q8: {((q8_rf_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

# Feature importance
tag_feature_names_q8 = list(X_tags_interact.columns)
rf_q8_importance = pd.DataFrame({
    'Feature': tag_feature_names_q8,
    'Importance': rf_q8.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(rf_q8_importance.head(10).to_string(index=False))

# Strategy 0 for Q8: Ordinary Least Squares (OLS) - No Regularization
print("="*70)
print("QUESTION 8 STRATEGY 0: Ordinary Least Squares (OLS)")
print("="*70)

# OLS using LinearRegression (no regularization)
ols_q8 = LinearRegression()
ols_q8.fit(X_tags_train_scaled, y_tags_train)

y_ols_q8_train_pred = ols_q8.predict(X_tags_train_scaled)
y_ols_q8_test_pred = ols_q8.predict(X_tags_test_scaled)

q8_ols_train_r2 = r2_score(y_tags_train, y_ols_q8_train_pred)
q8_ols_test_r2 = r2_score(y_tags_test, y_ols_q8_test_pred)
q8_ols_test_rmse = np.sqrt(mean_squared_error(y_tags_test, y_ols_q8_test_pred))

print(f"\nTrain R²: {q8_ols_train_r2:.4f}")
print(f"Test R²: {q8_ols_test_r2:.4f}")
print(f"Test RMSE: {q8_ols_test_rmse:.4f}")
print(f"\nComparison with Ridge (α=1.0):")
print(f"  OLS Test R²: {q8_ols_test_r2:.4f}")
print(f"  Ridge Test R²: {tags_test_r2:.4f}")
print(f"  Difference: {((q8_ols_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

# Strategy 5 for Q8: Gradient Boosting with Tags Only
print("\n" + "="*70)
print("QUESTION 8 STRATEGY 5: Gradient Boosting Regressor (Tags Only)")
print("="*70)

gb_q8 = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=N_NUMBER,
    subsample=0.8
)
gb_q8.fit(X_tags_int_train, y_tags_int_train)

y_gb_q8_train_pred = gb_q8.predict(X_tags_int_train)
y_gb_q8_test_pred = gb_q8.predict(X_tags_int_test)

q8_gb_train_r2 = r2_score(y_tags_int_train, y_gb_q8_train_pred)
q8_gb_test_r2 = r2_score(y_tags_int_test, y_gb_q8_test_pred)
q8_gb_test_rmse = np.sqrt(mean_squared_error(y_tags_int_test, y_gb_q8_test_pred))

print(f"\nTrain R²: {q8_gb_train_r2:.4f}")
print(f"Test R²: {q8_gb_test_r2:.4f}")
print(f"Test RMSE: {q8_gb_test_rmse:.4f}")
print(f"\nImprovement over Random Forest: {((q8_gb_test_r2 - q8_rf_test_r2) / q8_rf_test_r2 * 100):.2f}%")
print(f"Improvement over original Q8: {((q8_gb_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

# Feature importance
gb_q8_importance = pd.DataFrame({
    'Feature': tag_feature_names_q8,
    'Importance': gb_q8.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(gb_q8_importance.head(10).to_string(index=False))

# Strategy 6 for Q8: XGBoost with Tags Only
print("\n" + "="*70)
print("QUESTION 8 STRATEGY 6: XGBoost Regressor (Tags Only)")
print("="*70)

if XGBOOST_AVAILABLE:
    xgb_q8 = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=N_NUMBER,
        n_jobs=-1,
        verbosity=0
    )
    xgb_q8.fit(X_tags_int_train, y_tags_int_train)

    y_xgb_q8_train_pred = xgb_q8.predict(X_tags_int_train)
    y_xgb_q8_test_pred = xgb_q8.predict(X_tags_int_test)

    q8_xgb_train_r2 = r2_score(y_tags_int_train, y_xgb_q8_train_pred)
    q8_xgb_test_r2 = r2_score(y_tags_int_test, y_xgb_q8_test_pred)
    q8_xgb_test_rmse = np.sqrt(mean_squared_error(y_tags_int_test, y_xgb_q8_test_pred))

    print(f"\nTrain R²: {q8_xgb_train_r2:.4f}")
    print(f"Test R²: {q8_xgb_test_r2:.4f}")
    print(f"Test RMSE: {q8_xgb_test_rmse:.4f}")
    print(f"\nImprovement over Gradient Boosting: {((q8_xgb_test_r2 - q8_gb_test_r2) / q8_gb_test_r2 * 100):.2f}%")
    print(f"Improvement over original Q8: {((q8_xgb_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

    # Feature importance
    xgb_q8_importance = pd.DataFrame({
        'Feature': tag_feature_names_q8,
        'Importance': xgb_q8.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 Most Important Features:")
    print(xgb_q8_importance.head(10).to_string(index=False))

    # Try with hyperparameter tuning
    print("\n" + "-"*70)
    print("XGBoost with Grid Search Hyperparameter Tuning")
    print("-"*70)

    param_grid_xgb_q8 = {
        'n_estimators': [100, 200],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 0.9]
    }

    xgb_base_q8 = xgb.XGBRegressor(objective='reg:squarederror', random_state=N_NUMBER, n_jobs=-1, verbosity=0)
    grid_search_xgb_q8 = GridSearchCV(xgb_base_q8, param_grid_xgb_q8, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid_search_xgb_q8.fit(X_tags_int_train, y_tags_int_train)

    best_xgb_q8 = grid_search_xgb_q8.best_estimator_
    y_xgb_q8_tuned_train_pred = best_xgb_q8.predict(X_tags_int_train)
    y_xgb_q8_tuned_test_pred = best_xgb_q8.predict(X_tags_int_test)

    q8_xgb_tuned_train_r2 = r2_score(y_tags_int_train, y_xgb_q8_tuned_train_pred)
    q8_xgb_tuned_test_r2 = r2_score(y_tags_int_test, y_xgb_q8_tuned_test_pred)
    q8_xgb_tuned_test_rmse = np.sqrt(mean_squared_error(y_tags_int_test, y_xgb_q8_tuned_test_pred))

    print(f"\nBest parameters: {grid_search_xgb_q8.best_params_}")
    print(f"Train R²: {q8_xgb_tuned_train_r2:.4f}")
    print(f"Test R²: {q8_xgb_tuned_test_r2:.4f}")
    print(f"Test RMSE: {q8_xgb_tuned_test_rmse:.4f}")
    print(f"\nImprovement over default XGBoost: {((q8_xgb_tuned_test_r2 - q8_xgb_test_r2) / q8_xgb_test_r2 * 100):.2f}%")
    print(f"Improvement over original Q8: {((q8_xgb_tuned_test_r2 - tags_test_r2) / tags_test_r2 * 100):.2f}%")

else:
    print("\nXGBoost is not installed. Skipping this strategy.")
    print("To install: pip install xgboost")
    q8_xgb_test_r2 = 0
    q8_xgb_tuned_test_r2 = 0
    q8_xgb_test_rmse = 0
    q8_xgb_tuned_test_rmse = 0

# Summary Comparison for Question 8 Strategies
print("\n" + "="*70)
print("SUMMARY: R² Comparison for Question 8 (Tags-Only) Strategies")
print("="*70)

q8_model_list = [
    'Original Q8: Tags Only (Ridge α=1.0)',
    'Strategy 0: OLS (No Regularization)',
    'Strategy 1: Tuned Ridge',
    'Strategy 2: + Tag Interactions',
    'Strategy 3: ElasticNet',
    'Strategy 3.5: Lasso',
    'Strategy 4: Random Forest',
    'Strategy 5: Gradient Boosting',
    'Strategy 6: XGBoost (Default)'
]

q8_r2_list = [
    tags_test_r2,
    q8_ols_test_r2,
    q8_tuned_test_r2,
    q8_int_test_r2,
    q8_en_test_r2,
    q8_lasso_test_r2,
    q8_rf_test_r2,
    q8_gb_test_r2,
    q8_xgb_test_r2 if XGBOOST_AVAILABLE else 0
]

q8_rmse_list = [
    tags_test_rmse,
    q8_ols_test_rmse,
    q8_tuned_test_rmse,
    q8_int_test_rmse,
    q8_en_test_rmse,
    q8_lasso_test_rmse,
    q8_rf_test_rmse,
    q8_gb_test_rmse,
    q8_xgb_test_rmse if XGBOOST_AVAILABLE else 0
]

# Add tuned XGBoost if available
if XGBOOST_AVAILABLE:
    q8_model_list.append('Strategy 6: XGBoost (Tuned)')
    q8_r2_list.append(q8_xgb_tuned_test_r2)
    q8_rmse_list.append(q8_xgb_tuned_test_rmse)

q8_results_summary = pd.DataFrame({
    'Model': q8_model_list,
    'Test R²': q8_r2_list,
    'Test RMSE': q8_rmse_list
})

q8_results_summary['Improvement %'] = ((q8_results_summary['Test R²'] - tags_test_r2) / tags_test_r2 * 100).round(2)
q8_results_summary = q8_results_summary.sort_values('Test R²', ascending=False)

print("\n" + q8_results_summary.to_string(index=False))

best_q8_model_name = q8_results_summary.iloc[0]['Model']
best_q8_r2_value = q8_results_summary.iloc[0]['Test R²']
print(f"\nBest Model for Q8: {best_q8_model_name}")
print(f"   Best Test R²: {best_q8_r2_value:.4f}")
print(f"   Improvement over baseline: {q8_results_summary.iloc[0]['Improvement %']:.2f}%")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

# R² comparison
colors_q8 = ['#2E86AB' if 'Original' in m else '#F77F00' for m in q8_results_summary['Model']]
bars1 = ax1.barh(range(len(q8_results_summary)), q8_results_summary['Test R²'],
                 color=colors_q8, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax1.set_yticks(range(len(q8_results_summary)))
ax1.set_yticklabels(q8_results_summary['Model'], fontsize=9, fontweight='bold')
ax1.set_xlabel('Test R² Score', fontsize=12, fontweight='bold')
ax1.set_title('R² Comparison for Question 8 Strategies', fontsize=14, fontweight='bold', pad=12)
ax1.grid(True, alpha=0.2, linestyle='--', axis='x')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(q8_results_summary.iterrows()):
    ax1.text(row['Test R²'], i, f"  {row['Test R²']:.4f}",
             va='center', fontsize=9, fontweight='bold')

# RMSE comparison
bars2 = ax2.barh(range(len(q8_results_summary)), q8_results_summary['Test RMSE'],
                 color=colors_q8, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax2.set_yticks(range(len(q8_results_summary)))
ax2.set_yticklabels(q8_results_summary['Model'], fontsize=9, fontweight='bold')
ax2.set_xlabel('Test RMSE', fontsize=12, fontweight='bold')
ax2.set_title('RMSE Comparison for Question 8 Strategies', fontsize=14, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(q8_results_summary.iterrows()):
    ax2.text(row['Test RMSE'], i, f"  {row['Test RMSE']:.4f}",
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('q8_r2_improvement_strategies.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Question 9: Regression model predicting average difficulty from tags

"""

# Question 9: Regression model predicting difficulty from tags (data prep first)
q9_data = df_clean[df_clean['AvgDifficulty'].notna()].copy()
X_tags_diff = q9_data[tag_features].copy()
y_diff = q9_data['AvgDifficulty'].copy()

# Remove NaN
mask_diff = ~(X_tags_diff.isnull().any(axis=1) | y_diff.isnull())
X_tags_diff = X_tags_diff[mask_diff]
y_diff = y_diff[mask_diff]

# Split data
X_diff_train, X_diff_test, y_diff_train, y_diff_test = train_test_split(
    X_tags_diff, y_diff, test_size=0.2, random_state=N_NUMBER)

# Standardize
scaler_diff = StandardScaler()
X_diff_train_scaled = scaler_diff.fit_transform(X_diff_train)
X_diff_test_scaled = scaler_diff.transform(X_diff_test)

# Fit Ridge regression
ridge_diff = Ridge(alpha=1.0)
ridge_diff.fit(X_diff_train_scaled, y_diff_train)

# Predictions
y_diff_train_pred = ridge_diff.predict(X_diff_train_scaled)
y_diff_test_pred = ridge_diff.predict(X_diff_test_scaled)

# Metrics
diff_train_r2 = r2_score(y_diff_train, y_diff_train_pred)
diff_test_r2 = r2_score(y_diff_test, y_diff_test_pred)
diff_train_rmse = np.sqrt(mean_squared_error(y_diff_train, y_diff_train_pred))
diff_test_rmse = np.sqrt(mean_squared_error(y_diff_test, y_diff_test_pred))

print("Question 9: Regression Model - Average Difficulty from Tags")
print(f"Model Performance:")
print(f"  Training R²: {diff_train_r2:.4f}")
print(f"  Test R²: {diff_test_r2:.4f}")
print(f"  Training RMSE: {diff_train_rmse:.4f}")
print(f"  Test RMSE: {diff_test_rmse:.4f}")

# Question 9: OLS (Ordinary Least Squares) - No Regularization
print("\n" + "="*70)
print("QUESTION 9: OLS (Ordinary Least Squares)")
print("="*70)

# OLS using LinearRegression (no regularization)
ols_q9 = LinearRegression()
ols_q9.fit(X_diff_train_scaled, y_diff_train)

y_ols_q9_train_pred = ols_q9.predict(X_diff_train_scaled)
y_ols_q9_test_pred = ols_q9.predict(X_diff_test_scaled)

q9_ols_train_r2 = r2_score(y_diff_train, y_ols_q9_train_pred)
q9_ols_test_r2 = r2_score(y_diff_test, y_ols_q9_test_pred)
q9_ols_test_rmse = np.sqrt(mean_squared_error(y_diff_test, y_ols_q9_test_pred))

print(f"\nTrain R²: {q9_ols_train_r2:.4f}")
print(f"Test R²: {q9_ols_test_r2:.4f}")
print(f"Test RMSE: {q9_ols_test_rmse:.4f}")
print(f"\nComparison with Ridge (α=1.0):")
print(f"  OLS Test R²: {q9_ols_test_r2:.4f}")
print(f"  Ridge Test R²: {diff_test_r2:.4f}")
print(f"  Difference: {((q9_ols_test_r2 - diff_test_r2) / diff_test_r2 * 100):.2f}%")

# Feature importance
diff_tag_importance = pd.DataFrame({
    'Tag': tag_names,
    'Coefficient': ridge_diff.coef_,
    'Abs_Coefficient': np.abs(ridge_diff.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nTag Importance (sorted by absolute coefficient):")
print(diff_tag_importance.to_string(index=False))

most_important_diff_tag = diff_tag_importance.iloc[0]['Tag']
print(f"\nMost strongly predictive tag for difficulty: {most_important_diff_tag}")

# Enhanced visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

# Enhanced predicted vs actual with density coloring
scatter = ax1.scatter(y_diff_test, y_diff_test_pred, alpha=0.6, s=25,
                     c=np.abs(y_diff_test - y_diff_test_pred), cmap='RdYlBu_r',
                     edgecolors='black', linewidth=0.3)
ax1.plot([y_diff_test.min(), y_diff_test.max()], [y_diff_test.min(), y_diff_test.max()],
         'r--', linewidth=2.5, label='Perfect Prediction', zorder=0)
ax1.set_xlabel('Actual Average Difficulty', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Average Difficulty', fontsize=12, fontweight='bold')
ax1.set_title(f'Predicted vs Actual Difficulty\n(R² = {diff_test_r2:.3f}, RMSE = {diff_test_rmse:.3f})',
              fontsize=14, fontweight='bold', pad=12)
ax1.legend(fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Absolute Error', fontsize=10, fontweight='bold')

# Enhanced top 10 tag coefficients with color coding
top_10_diff_tags = diff_tag_importance.head(10)
colors_coef = ['#F77F00' if c > 0 else '#2E86AB' for c in top_10_diff_tags['Coefficient']]
bars = ax2.barh(range(len(top_10_diff_tags)), top_10_diff_tags['Coefficient'],
                color=colors_coef, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax2.set_yticks(range(len(top_10_diff_tags)))
ax2.set_yticklabels(top_10_diff_tags['Tag'], fontsize=9, fontweight='bold')
ax2.axvline(0, color='black', linestyle='--', linewidth=2, zorder=0)
ax2.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax2.set_title('Top 10 Tag Coefficients for Difficulty', fontsize=14, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(top_10_diff_tags.iterrows()):
    ax2.text(row['Coefficient'], i, f"  {row['Coefficient']:.3f}",
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('q9_regression_difficulty_tags.png', dpi=300, bbox_inches='tight')
plt.show()

# Strategy 1 for Q9: Hyperparameter Tuning (Ridge Alpha)
print("\n" + "="*70)
print("QUESTION 9 STRATEGY 1: Hyperparameter Tuning (Ridge Alpha)")
print("="*70)

# Tune Ridge alpha for difficulty prediction
alphas_q9 = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_alpha_q9 = 1.0
best_r2_q9 = -np.inf

for alpha in alphas_q9:
    ridge_temp_q9 = Ridge(alpha=alpha)
    ridge_temp_q9.fit(X_diff_train_scaled, y_diff_train)
    r2_temp_q9 = r2_score(y_diff_test, ridge_temp_q9.predict(X_diff_test_scaled))
    if r2_temp_q9 > best_r2_q9:
        best_r2_q9 = r2_temp_q9
        best_alpha_q9 = alpha

ridge_q9_tuned = Ridge(alpha=best_alpha_q9)
ridge_q9_tuned.fit(X_diff_train_scaled, y_diff_train)

y_q9_tuned_train_pred = ridge_q9_tuned.predict(X_diff_train_scaled)
y_q9_tuned_test_pred = ridge_q9_tuned.predict(X_diff_test_scaled)

q9_tuned_train_r2 = r2_score(y_diff_train, y_q9_tuned_train_pred)
q9_tuned_test_r2 = r2_score(y_diff_test, y_q9_tuned_test_pred)
q9_tuned_test_rmse = np.sqrt(mean_squared_error(y_diff_test, y_q9_tuned_test_pred))

print(f"\nBest alpha: {best_alpha_q9}")
print(f"Train R²: {q9_tuned_train_r2:.4f}")
print(f"Test R²: {q9_tuned_test_r2:.4f}")
print(f"Test RMSE: {q9_tuned_test_rmse:.4f}")
print(f"\nImprovement over original Q9: {((q9_tuned_test_r2 - diff_test_r2) / diff_test_r2 * 100):.2f}%")

# Strategy 2 for Q9: Tag Interaction Terms
print("\n" + "="*70)
print("QUESTION 9 STRATEGY 2: Tag Interaction Terms")
print("="*70)

# Create interaction terms between top tags for difficulty
X_tags_diff_interact = X_tags_diff.copy()

# Top interactions based on tag importance from Q9
X_tags_diff_interact['ToughGrader_TestHeavy'] = X_tags_diff_interact['ToughGrader_normalized'] * X_tags_diff_interact['TestHeavy_normalized']
X_tags_diff_interact['ToughGrader_LotsToRead'] = X_tags_diff_interact['ToughGrader_normalized'] * X_tags_diff_interact['LotsToRead_normalized']
X_tags_diff_interact['ToughGrader_ClearGrading'] = X_tags_diff_interact['ToughGrader_normalized'] * X_tags_diff_interact['ClearGrading_normalized']
X_tags_diff_interact['TestHeavy_LotsToRead'] = X_tags_diff_interact['TestHeavy_normalized'] * X_tags_diff_interact['LotsToRead_normalized']
X_tags_diff_interact['TestHeavy_LotsOfHomework'] = X_tags_diff_interact['TestHeavy_normalized'] * X_tags_diff_interact['LotsOfHomework_normalized']
X_tags_diff_interact['ClearGrading_Hilarious'] = X_tags_diff_interact['ClearGrading_normalized'] * X_tags_diff_interact['Hilarious_normalized']

# Split
X_diff_int_train, X_diff_int_test, y_diff_int_train, y_diff_int_test = train_test_split(
    X_tags_diff_interact, y_diff, test_size=0.2, random_state=N_NUMBER)

# Standardize
scaler_diff_int = StandardScaler()
X_diff_int_train_scaled = scaler_diff_int.fit_transform(X_diff_int_train)
X_diff_int_test_scaled = scaler_diff_int.transform(X_diff_int_test)

# Ridge with tuned alpha
ridge_diff_int = Ridge(alpha=best_alpha_q9)
ridge_diff_int.fit(X_diff_int_train_scaled, y_diff_int_train)

y_diff_int_train_pred = ridge_diff_int.predict(X_diff_int_train_scaled)
y_diff_int_test_pred = ridge_diff_int.predict(X_diff_int_test_scaled)

q9_int_train_r2 = r2_score(y_diff_int_train, y_diff_int_train_pred)
q9_int_test_r2 = r2_score(y_diff_int_test, y_diff_int_test_pred)
q9_int_test_rmse = np.sqrt(mean_squared_error(y_diff_int_test, y_diff_int_test_pred))

print(f"\nTrain R²: {q9_int_train_r2:.4f}")
print(f"Test R²: {q9_int_test_r2:.4f}")
print(f"Test RMSE: {q9_int_test_rmse:.4f}")
print(f"\nImprovement over tuned Ridge: {((q9_int_test_r2 - q9_tuned_test_r2) / q9_tuned_test_r2 * 100):.2f}%")
print(f"Improvement over original Q9: {((q9_int_test_r2 - diff_test_r2) / diff_test_r2 * 100):.2f}%")

# Strategy 3.5 for Q9: Lasso Regression (L1 Regularization) for Difficulty Prediction
print("\n" + "="*70)
print("QUESTION 9 STRATEGY 3.5: Lasso Regression (L1 Regularization) - Difficulty from Tags")
print("="*70)

# Lasso uses L1 regularization which can perform feature selection by setting coefficients to zero
# This helps identify which tags are most predictive of difficulty
# Lasso is more sensitive to alpha than Ridge, so we use a wider range including smaller values
param_grid_lasso_q9 = {
    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
}

lasso_q9 = Lasso(max_iter=5000)
grid_search_lasso_q9 = GridSearchCV(lasso_q9, param_grid_lasso_q9, cv=5, scoring='r2', n_jobs=-1)
grid_search_lasso_q9.fit(X_diff_train_scaled, y_diff_train)

best_lasso_q9 = grid_search_lasso_q9.best_estimator_
y_lasso_q9_train_pred = best_lasso_q9.predict(X_diff_train_scaled)
y_lasso_q9_test_pred = best_lasso_q9.predict(X_diff_test_scaled)

q9_lasso_train_r2 = r2_score(y_diff_train, y_lasso_q9_train_pred)
q9_lasso_test_r2 = r2_score(y_diff_test, y_lasso_q9_test_pred)
q9_lasso_test_rmse = np.sqrt(mean_squared_error(y_diff_test, y_lasso_q9_test_pred))

print(f"\nBest alpha: {grid_search_lasso_q9.best_params_['alpha']}")
print(f"Train R²: {q9_lasso_train_r2:.4f}")
print(f"Test R²: {q9_lasso_test_r2:.4f}")
print(f"Test RMSE: {q9_lasso_test_rmse:.4f}")
print(f"\nImprovement over original Q9: {((q9_lasso_test_r2 - diff_test_r2) / diff_test_r2 * 100):.2f}%")

# Show feature selection (coefficients that are zero)
coef_df_q9 = pd.DataFrame({
    'Feature': tag_features,
    'Coefficient': best_lasso_q9.coef_
})
coef_df_q9 = coef_df_q9.sort_values('Coefficient', key=abs, ascending=False)
print(f"\nTop 10 Tags by Absolute Coefficient:")
print(coef_df_q9.head(10).to_string(index=False))
zero_features_q9 = coef_df_q9[coef_df_q9['Coefficient'] == 0]
if len(zero_features_q9) > 0:
    print(f"\nTags eliminated by Lasso (coefficient = 0): {len(zero_features_q9)}")
    print(zero_features_q9['Feature'].tolist())

# Strategy 3 for Q9: ElasticNet
print("\n" + "="*70)
print("QUESTION 9 STRATEGY 3: ElasticNet Regression")
print("="*70)

param_grid_q9 = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

elastic_net_q9 = ElasticNet(max_iter=5000)
grid_search_q9 = GridSearchCV(elastic_net_q9, param_grid_q9, cv=5, scoring='r2', n_jobs=-1)
grid_search_q9.fit(X_diff_train_scaled, y_diff_train)

best_en_q9 = grid_search_q9.best_estimator_
y_en_q9_train_pred = best_en_q9.predict(X_diff_train_scaled)
y_en_q9_test_pred = best_en_q9.predict(X_diff_test_scaled)

q9_en_train_r2 = r2_score(y_diff_train, y_en_q9_train_pred)
q9_en_test_r2 = r2_score(y_diff_test, y_en_q9_test_pred)
q9_en_test_rmse = np.sqrt(mean_squared_error(y_diff_test, y_en_q9_test_pred))

print(f"\nBest parameters: {grid_search_q9.best_params_}")
print(f"Train R²: {q9_en_train_r2:.4f}")
print(f"Test R²: {q9_en_test_r2:.4f}")
print(f"Test RMSE: {q9_en_test_rmse:.4f}")
print(f"\nImprovement over original Q9: {((q9_en_test_r2 - diff_test_r2) / diff_test_r2 * 100):.2f}%")

# Strategy 4 for Q9: Random Forest
print("\n" + "="*70)
print("QUESTION 9 STRATEGY 4: Random Forest Regressor")
print("="*70)

rf_q9 = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=N_NUMBER,
    n_jobs=-1
)
rf_q9.fit(X_diff_int_train, y_diff_int_train)

y_rf_q9_train_pred = rf_q9.predict(X_diff_int_train)
y_rf_q9_test_pred = rf_q9.predict(X_diff_int_test)

q9_rf_train_r2 = r2_score(y_diff_int_train, y_rf_q9_train_pred)
q9_rf_test_r2 = r2_score(y_diff_int_test, y_rf_q9_test_pred)
q9_rf_test_rmse = np.sqrt(mean_squared_error(y_diff_int_test, y_rf_q9_test_pred))

print(f"\nTrain R²: {q9_rf_train_r2:.4f}")
print(f"Test R²: {q9_rf_test_r2:.4f}")
print(f"Test RMSE: {q9_rf_test_rmse:.4f}")
print(f"\nImprovement over interaction Ridge: {((q9_rf_test_r2 - q9_int_test_r2) / q9_int_test_r2 * 100):.2f}%")
print(f"Improvement over original Q9: {((q9_rf_test_r2 - diff_test_r2) / diff_test_r2 * 100):.2f}%")

# Feature importance
tag_feature_names_q9 = list(X_tags_diff_interact.columns)
rf_q9_importance = pd.DataFrame({
    'Feature': tag_feature_names_q9,
    'Importance': rf_q9.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(rf_q9_importance.head(10).to_string(index=False))

# Strategy 5 for Q9: Gradient Boosting
print("\n" + "="*70)
print("QUESTION 9 STRATEGY 5: Gradient Boosting Regressor")
print("="*70)

gb_q9 = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=N_NUMBER,
    subsample=0.8
)
gb_q9.fit(X_diff_int_train, y_diff_int_train)

y_gb_q9_train_pred = gb_q9.predict(X_diff_int_train)
y_gb_q9_test_pred = gb_q9.predict(X_diff_int_test)

q9_gb_train_r2 = r2_score(y_diff_int_train, y_gb_q9_train_pred)
q9_gb_test_r2 = r2_score(y_diff_int_test, y_gb_q9_test_pred)
q9_gb_test_rmse = np.sqrt(mean_squared_error(y_diff_int_test, y_gb_q9_test_pred))

print(f"\nTrain R²: {q9_gb_train_r2:.4f}")
print(f"Test R²: {q9_gb_test_r2:.4f}")
print(f"Test RMSE: {q9_gb_test_rmse:.4f}")
print(f"\nImprovement over Random Forest: {((q9_gb_test_r2 - q9_rf_test_r2) / q9_rf_test_r2 * 100):.2f}%")
print(f"Improvement over original Q9: {((q9_gb_test_r2 - diff_test_r2) / diff_test_r2 * 100):.2f}%")

# Feature importance
gb_q9_importance = pd.DataFrame({
    'Feature': tag_feature_names_q9,
    'Importance': gb_q9.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(gb_q9_importance.head(10).to_string(index=False))

# Strategy 6 for Q9: XGBoost
print("\n" + "="*70)
print("QUESTION 9 STRATEGY 6: XGBoost Regressor")
print("="*70)

if XGBOOST_AVAILABLE:
    xgb_q9 = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=N_NUMBER,
        n_jobs=-1,
        verbosity=0
    )
    xgb_q9.fit(X_diff_int_train, y_diff_int_train)

    y_xgb_q9_train_pred = xgb_q9.predict(X_diff_int_train)
    y_xgb_q9_test_pred = xgb_q9.predict(X_diff_int_test)

    q9_xgb_train_r2 = r2_score(y_diff_int_train, y_xgb_q9_train_pred)
    q9_xgb_test_r2 = r2_score(y_diff_int_test, y_xgb_q9_test_pred)
    q9_xgb_test_rmse = np.sqrt(mean_squared_error(y_diff_int_test, y_xgb_q9_test_pred))

    print(f"\nTrain R²: {q9_xgb_train_r2:.4f}")
    print(f"Test R²: {q9_xgb_test_r2:.4f}")
    print(f"Test RMSE: {q9_xgb_test_rmse:.4f}")
    print(f"\nImprovement over Gradient Boosting: {((q9_xgb_test_r2 - q9_gb_test_r2) / q9_gb_test_r2 * 100):.2f}%")
    print(f"Improvement over original Q9: {((q9_xgb_test_r2 - diff_test_r2) / diff_test_r2 * 100):.2f}%")

    # Feature importance
    xgb_q9_importance = pd.DataFrame({
        'Feature': tag_feature_names_q9,
        'Importance': xgb_q9.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 Most Important Features:")
    print(xgb_q9_importance.head(10).to_string(index=False))
else:
    print("\nXGBoost is not installed. Skipping this strategy.")
    print("To install: pip install xgboost")
    q9_xgb_test_r2 = 0
    q9_xgb_test_rmse = 0

# Summary Comparison for Question 9 Strategies
print("\n" + "="*70)
print("SUMMARY: R² Comparison for Question 9 (Difficulty from Tags) Strategies")
print("="*70)

q9_model_list = [
    'Original Q9: Tags Only (Ridge α=1.0)',
    'Strategy 0: OLS (No Regularization)',
    'Strategy 1: Tuned Ridge',
    'Strategy 2: + Tag Interactions',
    'Strategy 3: ElasticNet',
    'Strategy 3.5: Lasso',
    'Strategy 4: Random Forest',
    'Strategy 5: Gradient Boosting',
    'Strategy 6: XGBoost (Default)'
]

q9_r2_list = [
    diff_test_r2,
    q9_ols_test_r2,
    q9_tuned_test_r2,
    q9_int_test_r2,
    q9_en_test_r2,
    q9_lasso_test_r2,
    q9_rf_test_r2,
    q9_gb_test_r2,
    q9_xgb_test_r2 if XGBOOST_AVAILABLE else 0
]

q9_rmse_list = [
    diff_test_rmse,
    q9_ols_test_rmse,
    q9_tuned_test_rmse,
    q9_int_test_rmse,
    q9_en_test_rmse,
    q9_lasso_test_rmse,
    q9_rf_test_rmse,
    q9_gb_test_rmse,
    q9_xgb_test_rmse if XGBOOST_AVAILABLE else 0
]

q9_results_summary = pd.DataFrame({
    'Model': q9_model_list,
    'Test R²': q9_r2_list,
    'Test RMSE': q9_rmse_list
})

q9_results_summary['Improvement %'] = ((q9_results_summary['Test R²'] - diff_test_r2) / diff_test_r2 * 100).round(2)
q9_results_summary = q9_results_summary.sort_values('Test R²', ascending=False)

print("\n" + q9_results_summary.to_string(index=False))

best_q9_model_name = q9_results_summary.iloc[0]['Model']
best_q9_r2_value = q9_results_summary.iloc[0]['Test R²']
print(f"\nBest Model for Q9: {best_q9_model_name}")
print(f"   Best Test R²: {best_q9_r2_value:.4f}")
print(f"   Improvement over baseline: {q9_results_summary.iloc[0]['Improvement %']:.2f}%")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

# R² comparison
colors_q9 = ['#2E86AB' if 'Original' in m else '#F77F00' for m in q9_results_summary['Model']]
bars1 = ax1.barh(range(len(q9_results_summary)), q9_results_summary['Test R²'],
                 color=colors_q9, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax1.set_yticks(range(len(q9_results_summary)))
ax1.set_yticklabels(q9_results_summary['Model'], fontsize=9, fontweight='bold')
ax1.set_xlabel('Test R² Score', fontsize=12, fontweight='bold')
ax1.set_title('R² Comparison for Question 9 Strategies', fontsize=14, fontweight='bold', pad=12)
ax1.grid(True, alpha=0.2, linestyle='--', axis='x')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(q9_results_summary.iterrows()):
    ax1.text(row['Test R²'], i, f"  {row['Test R²']:.4f}",
             va='center', fontsize=9, fontweight='bold')

# RMSE comparison
bars2 = ax2.barh(range(len(q9_results_summary)), q9_results_summary['Test RMSE'],
                 color=colors_q9, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax2.set_yticks(range(len(q9_results_summary)))
ax2.set_yticklabels(q9_results_summary['Model'], fontsize=9, fontweight='bold')
ax2.set_xlabel('Test RMSE', fontsize=12, fontweight='bold')
ax2.set_title('RMSE Comparison for Question 9 Strategies', fontsize=14, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(q9_results_summary.iterrows()):
    ax2.text(row['Test RMSE'], i, f"  {row['Test RMSE']:.4f}",
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('q9_r2_improvement_strategies.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Question 10: Classification model predicting "Pepper" status

"""

# Question 10: Classification model for "Pepper" status
q10_data = df_clean[df_clean['Pepper'].notna()].copy()

# For Question 10, exclude 'Pepper' from features to avoid data leakage
# (Pepper is the target variable we're trying to predict)
q10_numerical_features = [f for f in numerical_features if f != 'Pepper']

# Combine numerical features (without Pepper) and normalized tags
X_class = pd.concat([
    q10_data[q10_numerical_features].fillna(q10_data[q10_numerical_features].median()),
    q10_data[tag_features]
], axis=1)
y_class = q10_data['Pepper'].astype(int)

# Remove NaN
mask_class = ~(X_class.isnull().any(axis=1) | y_class.isnull())
X_class = X_class[mask_class]
y_class = y_class[mask_class]

# Check class imbalance
print("Question 10: Classification Model - Predicting 'Pepper' Status")
print("Class Distribution:")
print(y_class.value_counts())
print(f"\nClass imbalance ratio: {y_class.value_counts()[0] / y_class.value_counts()[1]:.2f}:1")
print(f"Percentage of positive class: {y_class.mean() * 100:.2f}%")

# Split data
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=N_NUMBER, stratify=y_class)

# Standardize
scaler_class = StandardScaler()
X_class_train_scaled = scaler_class.fit_transform(X_class_train)
X_class_test_scaled = scaler_class.transform(X_class_test)

# Use Random Forest to handle class imbalance and non-linearity
# We'll use class_weight='balanced' to address imbalance
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                   random_state=N_NUMBER, max_depth=10)
rf_model.fit(X_class_train_scaled, y_class_train)

# Predictions
y_class_train_pred = rf_model.predict(X_class_train_scaled)
y_class_test_pred = rf_model.predict(X_class_test_scaled)
y_class_test_proba = rf_model.predict_proba(X_class_test_scaled)[:, 1]

# Metrics
train_auc = roc_auc_score(y_class_train, rf_model.predict_proba(X_class_train_scaled)[:, 1])
test_auc = roc_auc_score(y_class_test, y_class_test_proba)

print(f"\nModel Performance:")
print(f"  Training AUC-ROC: {train_auc:.4f}")
print(f"  Test AUC-ROC: {test_auc:.4f}")
print(f"\nClassification Report (Test Set):")
print(classification_report(y_class_test, y_class_test_pred))

# Feature importance
all_features = q10_numerical_features + tag_features
feature_importance_class = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance_class.head(10).to_string(index=False))

# Enhanced visualization with better colors
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17, 13))

# Enhanced ROC Curve
fpr, tpr, thresholds = roc_curve(y_class_test, y_class_test_proba)
ax1.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {test_auc:.4f})',
         color='#2E86AB', zorder=2)
ax1.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
ax1.plot([0, 1], [0, 1], 'r--', linewidth=2.5, label='Random Classifier', zorder=1)
ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax1.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=12)
ax1.legend(fontsize=11, framealpha=0.95, loc='lower right')
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Enhanced Confusion Matrix
cm = confusion_matrix(y_class_test, y_class_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['No Pepper', 'Pepper'], yticklabels=['No Pepper', 'Pepper'],
            cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'},
            linewidths=2, linecolor='black')
ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)

# Enhanced Feature Importance with gradient
top_15_features = feature_importance_class.head(15)
colors_imp = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_15_features)))
bars = ax3.barh(range(len(top_15_features)), top_15_features['Importance'],
                color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
ax3.set_yticks(range(len(top_15_features)))
ax3.set_yticklabels(top_15_features['Feature'], fontsize=9, fontweight='bold')
ax3.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax3.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold', pad=12)
ax3.grid(True, alpha=0.2, linestyle='--', axis='x')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
# Add value labels
for i, (idx, row) in enumerate(top_15_features.iterrows()):
    ax3.text(row['Importance'], i, f"  {row['Importance']:.4f}",
             va='center', fontsize=8, fontweight='bold')

# Enhanced Class distribution
bars = ax4.bar(['No Pepper', 'Pepper'], [y_class.value_counts()[0], y_class.value_counts()[1]],
               color=['#2E86AB', '#F77F00'], alpha=0.8, edgecolor='black', linewidth=2)
ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
ax4.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=12)
ax4.grid(True, alpha=0.2, linestyle='--', axis='y')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, [y_class.value_counts()[0], y_class.value_counts()[1]])):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('q10_classification_pepper.png', dpi=300, bbox_inches='tight')
plt.show()

"""## Extra Credit: Interesting Analysis with Qualitative Data

"""

# Extra Credit: Analysis with qualitative data (Major, University, State)
# Let's explore how ratings vary by state and major

ec_data = df_clean[df_clean['AvgRating'].notna()].copy()
ec_data = ec_data[ec_data['State'].notna() & (ec_data['State'] != '')].copy()
ec_data = ec_data[ec_data['Major'].notna() & (ec_data['Major'] != '')].copy()

# Analysis by State
state_ratings = ec_data.groupby('State').agg({
    'AvgRating': ['mean', 'std', 'count'],
    'AvgDifficulty': 'mean'
}).round(3)
state_ratings.columns = ['Mean_Rating', 'Std_Rating', 'Count', 'Mean_Difficulty']
state_ratings = state_ratings[state_ratings['Count'] >= 100].sort_values('Mean_Rating', ascending=False)

print("Extra Credit: Analysis with Qualitative Data")
print("\nTop 10 States by Average Rating (min 100 professors):")
print(state_ratings.head(10).to_string())

print("\n\nBottom 10 States by Average Rating (min 100 professors):")
print(state_ratings.tail(10).to_string())

# Analysis by Major
major_ratings = ec_data.groupby('Major').agg({
    'AvgRating': ['mean', 'std', 'count'],
    'AvgDifficulty': 'mean'
}).round(3)
major_ratings.columns = ['Mean_Rating', 'Std_Rating', 'Count', 'Mean_Difficulty']
major_ratings = major_ratings[major_ratings['Count'] >= 50].sort_values('Mean_Rating', ascending=False)

print("\n\nTop 10 Majors by Average Rating (min 50 professors):")
print(major_ratings.head(10).to_string())

print("\n\nBottom 10 Majors by Average Rating (min 50 professors):")
print(major_ratings.tail(10).to_string())

# Statistical test: Do ratings differ significantly by state?
# One-way ANOVA for top states
top_states = state_ratings.head(10).index
state_groups = [ec_data[ec_data['State'] == state]['AvgRating'].dropna()
                for state in top_states if len(ec_data[ec_data['State'] == state]) >= 50]

if len(state_groups) > 2:
    f_stat, p_value_state = f_oneway(*state_groups)
    print(f"\n\nOne-way ANOVA for top 10 states:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value_state:.6f}")
    print(f"  Significant at α=0.005? {'Yes' if p_value_state < 0.005 else 'No'}")

# Modern Enhanced Visualization - 3x2 Layout
# Create a 2x2 grid for 4 graphs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# 1. Top 10 States by Rating - Horizontal Bar Chart with gradient
top_10_states = state_ratings.head(10)
colors_states = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_10_states)))
bars1 = ax1.barh(range(len(top_10_states)), top_10_states['Mean_Rating'],
                 color=colors_states, alpha=0.85, edgecolor='white', linewidth=2, height=0.75)
ax1.set_yticks(range(len(top_10_states)))
ax1.set_yticklabels(top_10_states.index, fontsize=11, fontweight='bold')
ax1.set_xlabel('Mean Average Rating', fontsize=13, fontweight='bold')
ax1.set_title('Top 10 States by Average Rating', fontsize=15, fontweight='bold', pad=15)
ax1.set_xlim([3.7, 4.0])
ax1.grid(True, alpha=0.3, linestyle='--', axis='x', color='gray')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#333333')
ax1.spines['bottom'].set_color('#333333')
# Add value labels with count
for i, (idx, row) in enumerate(top_10_states.iterrows()):
    ax1.text(row['Mean_Rating'] + 0.005, i, f"{row['Mean_Rating']:.3f} (n={int(row['Count'])})",
             va='center', fontsize=10, fontweight='bold', color='#2c3e50')

# 2. Top 10 Majors by Rating - Horizontal Bar Chart
top_10_majors = major_ratings.head(10)
colors_majors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_10_majors)))
bars2 = ax2.barh(range(len(top_10_majors)), top_10_majors['Mean_Rating'],
                 color=colors_majors, alpha=0.85, edgecolor='white', linewidth=2, height=0.75)
ax2.set_yticks(range(len(top_10_majors)))
ax2.set_yticklabels(top_10_majors.index, fontsize=10, fontweight='bold')
ax2.set_xlabel('Mean Average Rating', fontsize=13, fontweight='bold')
ax2.set_title('Top 10 Majors by Average Rating', fontsize=15, fontweight='bold', pad=15)
ax2.set_xlim([3.5, 4.3])
ax2.grid(True, alpha=0.3, linestyle='--', axis='x', color='gray')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#333333')
ax2.spines['bottom'].set_color('#333333')
# Add value labels
for i, (idx, row) in enumerate(top_10_majors.iterrows()):
    ax2.text(row['Mean_Rating'] + 0.01, i, f"{row['Mean_Rating']:.3f} (n={int(row['Count'])})",
             va='center', fontsize=9, fontweight='bold', color='#2c3e50')

# 3. State: Rating vs Difficulty - Enhanced Scatter with regression line
scatter1 = ax3.scatter(state_ratings['Mean_Difficulty'], state_ratings['Mean_Rating'],
                      s=state_ratings['Count']/5, alpha=0.65,
                      c=state_ratings['Mean_Rating'], cmap='coolwarm',
                      edgecolors='white', linewidth=1.5, zorder=3)
# Add regression line
z = np.polyfit(state_ratings['Mean_Difficulty'], state_ratings['Mean_Rating'], 1)
p = np.poly1d(z)
ax3.plot(state_ratings['Mean_Difficulty'], p(state_ratings['Mean_Difficulty']),
         "r--", alpha=0.8, linewidth=2.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}', zorder=2)
ax3.set_xlabel('Mean Difficulty', fontsize=13, fontweight='bold')
ax3.set_ylabel('Mean Rating', fontsize=13, fontweight='bold')
ax3.set_title('State: Rating vs Difficulty\n(Bubble size = sample size)',
              fontsize=15, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.25, linestyle='--', color='gray', zorder=1)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(fontsize=10, framealpha=0.9, loc='best')
cbar1 = plt.colorbar(scatter1, ax=ax3, pad=0.02)
cbar1.set_label('Mean Rating', fontsize=11, fontweight='bold')

# 4. Major: Rating vs Difficulty - Enhanced Scatter with regression line
scatter2 = ax4.scatter(major_ratings['Mean_Difficulty'], major_ratings['Mean_Rating'],
                      s=major_ratings['Count']/3, alpha=0.65,
                      c=major_ratings['Mean_Rating'], cmap='plasma',
                      edgecolors='white', linewidth=1.5, zorder=3)
# Add regression line
z2 = np.polyfit(major_ratings['Mean_Difficulty'], major_ratings['Mean_Rating'], 1)
p2 = np.poly1d(z2)
ax4.plot(major_ratings['Mean_Difficulty'], p2(major_ratings['Mean_Difficulty']),
         "r--", alpha=0.8, linewidth=2.5, label=f'Trend: y={z2[0]:.3f}x+{z2[1]:.3f}', zorder=2)
ax4.set_xlabel('Mean Difficulty', fontsize=13, fontweight='bold')
ax4.set_ylabel('Mean Rating', fontsize=13, fontweight='bold')
ax4.set_title('Major: Rating vs Difficulty\n(Bubble size = sample size)',
              fontsize=15, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.25, linestyle='--', color='gray', zorder=1)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.legend(fontsize=10, framealpha=0.9, loc='best')
cbar2 = plt.colorbar(scatter2, ax=ax4, pad=0.02)
cbar2.set_label('Mean Rating', fontsize=11, fontweight='bold')



plt.suptitle('Extra Credit: Comprehensive Analysis of Ratings by State and Major',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('extra_credit_qualitative_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
