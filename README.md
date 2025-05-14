# Comprehensive data analysis
A Python-based command-line tool designed to perform comprehensive statistical analysis on one, two and multiple columns of data from CSV or tab-delimited text files. Built to supports both numeric (continuous or discrete) and categorical data, offering a range of descriptive statistics, hypothesis tests, normality assessments. It is ideal for researchers, data analysts, and scientists who need to explore datasets, test hypotheses, and generate textual summaries of statistical findings.

The script is executed via the command line, taking a file path as an argument. It interactively prompts users to select analysis type (single-column, two-column or multiple-column), specify columns, and provide parameters such as custom percentiles or hypothesized values for hypothesis tests. Outputs include a statistical summary file, few visualizations (e.g., scatter plots, box plots), and a detailed log file capturing all terminal interactions.

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

# Timelines
# Statistical Analysis Script Feature Table

The following table outlines the statistical types, column type analyses, descriptions, and timelines for the implementation of features in the Statistical Analysis Script.

| Statistical Type           | Column Type Analysis                                                                 | Description                                                                 | Timelines       |
|----------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------|
| **Descriptive Statistics** |                                                                                     |                                                                             |                 |
|                            | Null count, Mean, median, mode                                                     | Calculates the number of missing values, average, central value, and most frequent value. | 30/4/2025       |
|                            | Minimum, maximum, range                                                            | Identifies the smallest value, largest value, and difference between them.   | 30/4/2025       |
|                            | Standard deviation, variance                                                       | Measures data dispersion and squared deviation from the mean.                | 30/4/2025       |
|                            | Skewness, kurtosis                                                                 | Assesses asymmetry and tailedness of the data distribution.                 | 30/4/2025       |
|                            | Quartiles (Q1, Q3), interquartile range (IQR)                                      | Computes 25th and 75th percentiles and the range between them.              | 30/4/2025       |
|                            | User-specified percentiles (e.g., 10th, 90th)                                      | Calculates custom percentiles as specified by the user.                     | 30/4/2025       |
| **Normality Tests**        |                                                                                     |                                                                             |                 |
|                            | Shapiro-Wilk test, Kolmogorov-Smirnov test, Anderson-Darling test                  | Tests whether data follows a normal distribution using different statistical methods. | 30/4/2025       |
| **Hypothesis Tests**       | 1 column statistical analysis (30/4/2025)<br>2 column statistical analysis (21/5/2025)<br>Multiple column statistical analysis (25/6/2025) |                                                                             |                 |
|                            | One-sample t-test (tests mean against a hypothesized value)                        | Tests if the sample mean differs significantly from a hypothesized mean.    | 30/4/2025       |
|                            | One-sample z-test (requires known population standard deviation)                    | Tests the sample mean with known population standard deviation.             | 30/4/2025       |
|                            | One-sample sign test (tests median)                                                | Non-parametric test for the median against a hypothesized value.            | 30/4/2025       |
|                            | One-sample Wilcoxon signed-rank test (tests median)                                | Non-parametric test for the median using signed ranks.                     | 30/4/2025       |
|                            | One-sample z-test for proportions (for discrete data)                              | Tests if the sample proportion differs from a hypothesized proportion.      | 07/05/2025      |
|                            | Paired t-test (compares means)                                                    | Tests if the means of two related samples differ significantly.             | 07/05/2025      |
|                            | Wilcoxon signed-rank test (compares medians)                                       | Non-parametric test comparing medians of two related samples.              | 07/05/2025      |
|                            | Chi-Square goodness-of-fit test (compares observed frequencies to user-specified or equal expected proportions) | Tests if observed categorical frequencies match expected proportions.       | 07/05/2025      |
|                            | Pearson correlation (linear relationship)                                          | Measures the strength and direction of the linear relationship between two variables. | 07/05/2025      |
|                            | Spearman correlation (monotonic relationship)                                      | Measures the strength of a monotonic relationship between two variables.    | 07/05/2025      |
|                            | Kendall tau correlation (monotonic relationship)                                   | Measures the ordinal association between two variables.                     | 07/05/2025      |
|                            | Chi-Square test of association (tests independence)                                | Tests if two categorical variables are independent.                        | 07/05/2025      |
|                            | Chi-Square test of difference (compares distributions)                             | Tests if the distributions of two categorical variables differ.             | 07/05/2025      |
|                            | Fisher’s Exact test (for 2x2 contingency tables)                                   | Tests association in 2x2 contingency tables for small sample sizes.         | 07/05/2025      |
| **Visualizations**         |                                                                                     |                                                                             |                 |
|                            | Box plot (shows quartiles, IQR, outliers)                                          | Visualizes data distribution, including quartiles, IQR, and outliers.       | 30/4/2025       |
|                            | Histogram (discrete data distribution)                                             | Displays the frequency distribution of discrete data.                       | 30/4/2025       |
|                            | Kernel Density Estimation (KDE) plot (continuous data density)                     | Estimates and visualizes the probability density of continuous data.        | 30/4/2025       |
|                            | Frequency bar plot (shows top 5 categories, with an "Other" category for additional categories) | Visualizes the frequency of categorical data, limited to top 5 categories.  | 30/4/2025       |
|                            | Scatter plot (shows relationship between columns)                                  | Displays the relationship between two numeric variables.                    | 30/4/2025       |
|                            | Stacked bar plot (shows relationship between columns)                              | Visualizes the relationship between two categorical variables.              | 30/4/2025       |

## Notes
- **Timelines**: Indicate the planned or completed implementation dates for each feature.
- **Column Type Analysis**: Specifies the scope of analysis (e.g., single-column, two-column, or multiple-column statistical analysis) and associated timelines where applicable.
- **Kendall tau correlation**: Included in the table but not yet implemented in the provided script, scheduled for 07/05/2025.
- **Multiple column statistical analysis**: Planned for 25/06/2025, indicating future expansion beyond the current one- and two-column capabilities.

This table serves as a roadmap for the development and implementation of statistical features in the Statistical Analysis Script, ensuring comprehensive coverage of descriptive, inferential, and visual analysis methods.


