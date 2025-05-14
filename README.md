# Comprehensive data analysis
A Python-based command-line tool designed to perform comprehensive statistical analysis on one, two and multiple columns of data from CSV or tab-delimited text files. Built to support both numeric (continuous or discrete) and categorical data, offering a range of descriptive statistics, hypothesis tests, normality assessments.

The script is executed via the command line, taking a file path/file as an argument. It prompts users to select analysis type (single-column, two-column or multiple-column), specify column names, and provide parameters such as custom percentiles or hypothesized values for hypothesis tests. Outputs include a statistical summary file, few visualizations (e.g., scatter plots, box plots), and a detailed log file capturing all terminal interactions.

The script serves multiple purposes:

1- Exploratory Data Analysis (EDA): Provides detailed descriptive statistics (e.g., mean, median, percentiles, null counts) to summarize data distributions.

2- Hypothesis Testing: Conducts statistical tests (e.g., t-test, Wilcoxon test, Chi-Square test) to evaluate hypotheses about means, medians, proportions, or associations.

3- Normality Assessment: Performs tests (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling) to check if numeric data follows a normal distribution.

4- Data Visualization: Generates plots (e.g., box plots, histograms, frequency plots) to visually represent data distributions and relationships.

5- Reproducibility: Saves statistical results, visualizations, and a log file for documentation and future reference.

# Requirements
Python 3+

Required Libraries:
  - `pandas`
  - `matplotlib`
  - `kaleido`
  - `plotly`
  - `statsmodels`

# Script Usage
python <script_file> <input_file>


# Statistical Analysis Script Feature Table

| Statistical Type           | Description                                                                 | Timelines       |
|----------------------------|-----------------------------------------------------------------------------|-----------------|
| Descriptive Statistics     | Null count, Mean, median, mode                                             | 30/4/2025       |
|                            | Minimum, maximum, range                                                    | 30/4/2025       |
|                            | Standard deviation, variance                                               | 30/4/2025       |
|                            | Skewness, kurtosis                                                         | 30/4/2025       |
|                            | Quartiles (Q1, Q3), interquartile range (IQR)                              | 30/4/2025       |
|                            | User-specified percentiles (e.g., 10th, 90th)                              | 30/4/2025       |
| Normality Tests            | Shapiro-Wilk test, Kolmogorov-Smirnov test, Anderson-Darling test          | 30/4/2025       |
| Hypothesis Tests           | One-sample t-test (tests mean against a hypothesized value)                | 30/4/2025       |
|                            | One-sample z-test (requires known population standard deviation)           | 30/4/2025       |
|                            | One-sample sign test (tests median)                                       | 30/4/2025       |
|                            | One-sample Wilcoxon signed-rank test (tests median)                       | 30/4/2025       |
|                            | One-sample z-test for proportions (for discrete data)                     | 07/05/25        |
|                            | Paired t-test (compares means)                                            | 07/05/25        |
|                            | Wilcoxon signed-rank test (compares medians)                              | 07/05/25        |
|                            | Chi-Square goodness-of-fit test (compares observed frequencies to user-specified or equal expected proportions) | 07/05/25        |
|                            | Pearson correlation (linear relationship)                                 | 07/05/25        |
|                            | Spearman correlation (monotonic relationship)                             | 07/05/25        |
|                            | Kendall tau correlation (monotonic relationship)                          | 07/05/25        |
|                            | Chi-Square test of association (tests independence)                       | 07/05/25        |
|                            | Chi-Square test of difference (compares distributions)                    | 07/05/25        |
|                            | Fisher’s Exact test (for 2x2 contingency tables)                          | 07/05/25        |
| Visualizations             | Box plot (shows quartiles, IQR, outliers)                                  | 30/4/2025       |
|                            | Histogram (discrete data distribution)                                     | 30/4/2025       |
|                            | Kernel Density Estimation (KDE) plot (continuous data density)             | 30/4/2025       |
|                            | Frequency bar plot (shows top 5 categories, with an "Other" category for additional categories) | 30/4/2025       |
|                            | Scatter plot (shows relationship between columns)                          | 30/4/2025       |
|                            | Stacked bar plot (shows relationship between columns)                      | 30/4/2025       |


# Timelines
1 column statistical analysis (30/4/2025)

2 column statistical analysis (21/5/2025)

Multiple column statistical analysis (25/6/2025)
