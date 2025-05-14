# Comprehensive data analysis
A Python-based command-line tool designed to perform comprehensive statistical analysis on one, two and multiple columns of data from CSV or tab-delimited text files. Built to supports both numeric (continuous or discrete) and categorical data, offering a range of descriptive statistics, hypothesis tests, normality assessments. It is ideal for researchers, data analysts, and scientists who need to explore datasets, test hypotheses, and generate textual summaries of statistical findings.

The script is executed via the command line, taking a file path as an argument. It interactively prompts users to select analysis type (single-column, two-column or multiple-column), specify columns, and provide parameters such as custom percentiles or hypothesized values for hypothesis tests. Outputs include a statistical summary file, few visualizations (e.g., scatter plots, box plots), and a detailed log file capturing all terminal interactions.

The script serves multiple purposes:

Exploratory Data Analysis (EDA): Provides detailed descriptive statistics (e.g., mean, median, percentiles, null counts) to summarize data distributions.
Hypothesis Testing: Conducts statistical tests (e.g., t-test, Wilcoxon test, Chi-Square test) to evaluate hypotheses about means, medians, proportions, or associations.
Normality Assessment: Performs tests (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling) to check if numeric data follows a normal distribution.
Data Visualization: Generates plots (e.g., box plots, histograms, frequency plots) to visually represent data distributions and relationships.
Reproducibility: Saves statistical results, visualizations, and a log file for documentation and future reference.

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
| No of columns      | Completion dates |
|--------------------|------------------|
| 1 column           | Done             |
| 2 columns          | 14/5/2025        |
| Multiple columns   | 04/06/25         |


