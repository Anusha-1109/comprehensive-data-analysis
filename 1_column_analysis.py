#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import os
import numpy as np
from scipy import stats
import plotly.express as px
from statsmodels.stats.weightstats import zconfint
from datetime import datetime
import shutil

def get_custom_percentiles():
    """Get custom percentile values from the terminal."""
    percentiles = []
    print("Enter custom percentiles (e.g., 10, 90, 95, 99) one at a time. Type 'done' to finish.")
    while True:
        percentile = input("Enter a percentile (0-100) or 'done': ")
        if percentile.lower() == 'done':
            break
        try:
            percentile = float(percentile)
            if 0 <= percentile <= 100:
                percentiles.append(percentile)
            else:
                print("Error: Percentile must be between 0 and 100. Try again.")
        except ValueError:
            print("Error: Please enter a valid number or 'done'. Try again.")
    return percentiles if percentiles else [10, 90, 95, 99]  # Default if no valid input

def analyze_log2_data(file_path, output_dir=None):
    """
    Analyze data from a user-provided file and save plots and summaries in a timestamped directory.
    For numeric data: KDE, histogram, boxplot, and trend analysis. For non-numeric: interactive frequency bar plot using Plotly (HTML) and static bar plot using Matplotlib with stats.
    Adds descriptive statistics including percentiles, IQR, outliers (z-score), normality test, confidence intervals, and trend analysis (if ordered).
    Saves a summary in a text file with detailed insights and a stats summary.
    
    Parameters:
    file_path (str): Path to the input file containing data
    output_dir (str): Base directory to save outputs (default: current directory)
    
    Returns:
    None: Saves plots as PNG or HTML/PNG files and summaries in the timestamped directory
    """
    try:
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = os.getcwd()
        output_dir = os.path.join(output_dir, f"analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get custom percentiles from user
        custom_percentiles = get_custom_percentiles()

        # Read the file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")

        # Print the dataframe for debugging
        print("Analyzing the data")
        print("Sample data:", df.head())

        # Get the column name (assuming single column or first column with data)
        if len(df.columns) == 0:
            raise ValueError("No columns found in the input file.")
        column_name = df.columns[0]  # Use the first column name

        # Clean the data based on type
        if pd.to_numeric(df[column_name], errors='coerce').notnull().all():
            # Numeric data: no string operations, just handle NaN
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        else:
            # Non-numeric data: strip whitespace and handle empty strings
            df[column_name] = df[column_name].str.strip().replace('', pd.NA)

        # List to store plot summary
        plot_summary = []
        stats_summary = []

        # Calculate statistics and count null entries
        data = df[column_name]
        null_entries = data.isna().sum()
        data = data.dropna()
        count = len(data)
        if pd.to_numeric(data, errors='coerce').notnull().all():
            numeric_data = pd.to_numeric(data)
            mean = numeric_data.mean()
            median = numeric_data.median()
            mode = numeric_data.mode().tolist()
            min_val = numeric_data.min()
            max_val = numeric_data.max()
            range_val = max_val - min_val
            std_dev = np.std(numeric_data)
            variance = np.var(numeric_data)
            skewness = stats.skew(numeric_data)
            kurtosis = stats.kurtosis(numeric_data, fisher=True)  # Excess kurtosis
            std_error = std_dev / np.sqrt(count) if count > 0 else 0
            # Calculate fixed percentiles
            q1 = np.percentile(numeric_data, 25)
            q3 = np.percentile(numeric_data, 75)
            iqr = q3 - q1
            # Calculate custom percentiles
            custom_percentile_values = {f"P{p:.0f}": np.percentile(numeric_data, p) for p in custom_percentiles}
            # Determine skewness direction
            skewness_desc = "Approximately symmetric"
            if skewness > 0.5:
                skewness_desc = "Positively skewed (right-tailed)"
            elif skewness < -0.5:
                skewness_desc = "Negatively skewed (left-tailed)"

            # Outlier detection using z-scores (> 3 standard deviations)
            z_scores = np.abs((numeric_data - mean) / std_dev)
            outliers = numeric_data[z_scores > 3]
            outlier_count = len(outliers)
            outlier_values = outliers.tolist() if outlier_count > 0 else ["None"]

            # Normality test (Kolmogorov-Smirnov)
            ks_statistic, ks_p_value = stats.kstest(numeric_data, 'norm', args=(mean, std_dev))
            is_normal = ks_p_value > 0.05  # Typical significance level

            # 95% Confidence Interval for the mean
            ci_lower, ci_upper = zconfint(numeric_data, alpha=0.05)

            # Trend analysis (moving average and slope for ordered data)
            if len(numeric_data) > 1 and pd.api.types.is_numeric_dtype(df.index):
                window_size = min(5, len(numeric_data))  # Use 5 or data length if smaller
                moving_avg = numeric_data.rolling(window=window_size, min_periods=1).mean()
                trend_slope, _, _, _, _ = stats.linregress(range(len(numeric_data)), numeric_data)
            else:
                moving_avg = pd.Series([np.nan])
                trend_slope = np.nan

            # Stats summary for numeric data
            stats_summary.append(f"Analysis of Numeric Data: {column_name}")
            stats_summary.append(f"- Total entries: {count}, Null entries: {null_entries}")
            stats_summary.append(f"- Mean: {mean:.2f}, Median: {median:.2f}, Mode: {', '.join(map(str, mode))}")
            stats_summary.append(f"- Range: {range_val:.2f} (Min: {min_val:.2f}, Max: {max_val:.2f})")
            stats_summary.append(f"- Standard Deviation: {std_dev:.2f}, Variance: {variance:.2f}")
            stats_summary.append(f"- Skewness: {skewness:.2f} ({skewness_desc})")
            stats_summary.append(f"- Kurtosis: {kurtosis:.2f}")
            stats_summary.append(f"- Percentiles: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
            for p, val in custom_percentile_values.items():
                stats_summary.append(f"  - {p}: {val:.2f}")
            stats_summary.append(f"- Outliers (Z>3): {outlier_count} ({', '.join(map(str, outlier_values))})")
            stats_summary.append(f"- Normality (KS test): p={ks_p_value:.4f} ({'Normal' if is_normal else 'Not Normal'})")
            stats_summary.append(f"- 95% CI for Mean: [{ci_lower:.2f}, {ci_upper:.2f}]")
            if not np.isnan(trend_slope):
                stats_summary.append(f"- Trend Slope: {trend_slope:.4f}")

        else:
            mean = "N/A (Non-numeric)"
            median = "N/A (Non-numeric)"
            mode = data.mode().tolist()
            min_val = "N/A (Non-numeric)"
            max_val = "N/A (Non-numeric)"
            range_val = "N/A (Non-numeric)"
            std_dev = "N/A (Non-numeric)"
            variance = "N/A (Non-numeric)"
            skewness = "N/A (Non-numeric)"
            kurtosis = "N/A (Non-numeric)"
            std_error = "N/A (Non-numeric)"
            q1 = "N/A (Non-numeric)"
            q3 = "N/A (Non-numeric)"
            iqr = "N/A (Non-numeric)"
            custom_percentile_values = {f"P{p}": "N/A (Non-numeric)" for p in custom_percentiles}
            skewness_desc = "N/A (Non-numeric)"
            outlier_count = "N/A (Non-numeric)"
            outlier_values = ["N/A (Non-numeric)"]
            ks_statistic = "N/A (Non-numeric)"
            ks_p_value = "N/A (Non-numeric)"
            is_normal = "N/A (Non-numeric)"
            ci_lower = "N/A (Non-numeric)"
            ci_upper = "N/A (Non-numeric)"
            moving_avg = pd.Series(["N/A (Non-numeric)"])
            trend_slope = "N/A (Non-numeric)"

            # Stats summary for non-numeric data
            value_counts = data.value_counts()
            total_unique = len(value_counts)
            total_entries = len(data)
            stats_summary.append(f"Analysis of Non-Numeric Data: {column_name}")
            stats_summary.append(f"- Total entries: {count}, Null entries: {null_entries}")
            stats_summary.append(f"- Unique values: {total_unique}")
            stats_summary.append(f"- Mode: {', '.join(map(str, mode))}")
            stats_summary.append(f"- Top 5 most frequent values:")
            for idx, (val, freq) in enumerate(value_counts.head(5).items()):
                stats_summary.append(f"  - {val}: {freq} occurrences")

        # Check if the data is numeric
        is_numeric = pd.to_numeric(df[column_name], errors='coerce').notnull().all()

        if is_numeric:
            # Remove duplicates for numeric data
            df = df.drop_duplicates()

            # Create and save KDE plot
            plt.figure(figsize=(10, 6), dpi=100)
            df[column_name].plot.kde()
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title(f'{column_name} Data', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            # Add descriptive statistics to right side
            stats_text = f"Descriptive Statistics\nMean: {mean:.2f}\nStandard Error: {std_error:.2f}\nStandard Deviation: {std_dev:.2f}\nVariance: {variance:.2f}\nSum: {numeric_data.sum():.2f}\nCount: {count}\nMinimum: {min_val:.2f}\nMaximum: {max_val:.2f}\nRange: {range_val:.2f}\nKurtosis: {kurtosis:.2f}\nSkewness: {skewness:.2f} ({skewness_desc})\nQ1 (25th): {q1:.2f}\nQ3 (75th): {q3:.2f}\nIQR: {iqr:.2f}\nOutliers (Z>3): {outlier_count} ({', '.join(map(str, outlier_values))})\nNormality (KS test): p={ks_p_value:.4f} ({'Normal' if is_normal else 'Not Normal'})\n95% CI for Mean: [{ci_lower:.2f}, {ci_upper:.2f}]"
            for p, val in custom_percentile_values.items():
                stats_text += f"\n{p}: {val:.2f}" if isinstance(val, (int, float)) else f"\n{p}: {val}"
            plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            kde_path = os.path.join(output_dir, 'kde.png')
            plt.savefig(kde_path, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            plot_summary.append(f"**Plot Type: KDE**\nFile: {kde_path}\nCount: {count}\nNull Entries: {null_entries}\nMean: {mean:.2f}\nMedian: {median:.2f}\nMode: {mode}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nRange: {range_val:.2f}\nStandard Deviation: {std_dev:.2f}\nVariance: {variance:.2f}\nSkewness: {skewness:.2f} ({skewness_desc})\nKurtosis: {kurtosis:.2f}\nQ1 (25th): {q1:.2f}\nQ3 (75th): {q3:.2f}\nIQR: {iqr:.2f}\nOutliers (Z>3): {outlier_count} ({', '.join(map(str, outlier_values))})\nNormality (KS test): p={ks_p_value:.4f} ({'Normal' if is_normal else 'Not Normal'})\n95% CI for Mean: [{ci_lower:.2f}, {ci_upper:.2f}]")
            for p, val in custom_percentile_values.items():
                plot_summary[-1] += f"\n{p}: {val:.2f}" if isinstance(val, (int, float)) else f"\n{p}: {val}"
            plot_summary[-1] += "\nDescription: Estimates the probability density function of the data.\n\n"

            # Create and save histogram
            plt.figure(figsize=(10, 6), dpi=100)
            bin_count = max(10, int(1 + 3.322 * np.log(len(df))))
            plt.hist(df[column_name], bins=bin_count, color='lightgreen', edgecolor='black')
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'{column_name} Data', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            # Add descriptive statistics to right side
            stats_text = f"Descriptive Statistics\nMean: {mean:.2f}\nStandard Error: {std_error:.2f}\nStandard Deviation: {std_dev:.2f}\nVariance: {variance:.2f}\nSum: {numeric_data.sum():.2f}\nCount: {count}\nMinimum: {min_val:.2f}\nMaximum: {max_val:.2f}\nRange: {range_val:.2f}\nKurtosis: {kurtosis:.2f}\nSkewness: {skewness:.2f} ({skewness_desc})\nQ1 (25th): {q1:.2f}\nQ3 (75th): {q3:.2f}\nIQR: {iqr:.2f}\nOutliers (Z>3): {outlier_count} ({', '.join(map(str, outlier_values))})\nNormality (KS test): p={ks_p_value:.4f} ({'Normal' if is_normal else 'Not Normal'})\n95% CI for Mean: [{ci_lower:.2f}, {ci_upper:.2f}]"
            for p, val in custom_percentile_values.items():
                stats_text += f"\n{p}: {val:.2f}" if isinstance(val, (int, float)) else f"\n{p}: {val}"
            plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            histogram_path = os.path.join(output_dir, 'histogram.png')
            plt.savefig(histogram_path, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            # Prepare bin frequency details for summary
            n, bins, patches = plt.hist(df[column_name], bins=bin_count)
            bin_edges = bins[:-1]  # Exclude the last edge (included in next bin)
            bin_freqs = n
            bin_details = "\nBin Details:\n"
            for edge, freq in zip(bin_edges, bin_freqs):
                bin_details += f"Bin Edge: {edge:.2f}, Frequency: {freq}\n"
            plot_summary.append(f"**Plot Type: Histogram**\nFile: {histogram_path}\nCount: {count}\nNull Entries: {null_entries}\nMean: {mean:.2f}\nMedian: {median:.2f}\nMode: {mode}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nRange: {range_val:.2f}\nStandard Deviation: {std_dev:.2f}\nVariance: {variance:.2f}\nSkewness: {skewness:.2f} ({skewness_desc})\nKurtosis: {kurtosis:.2f}\nQ1 (25th): {q1:.2f}\nQ3 (75th): {q3:.2f}\nIQR: {iqr:.2f}\nOutliers (Z>3): {outlier_count} ({', '.join(map(str, outlier_values))})\nNormality (KS test): p={ks_p_value:.4f} ({'Normal' if is_normal else 'Not Normal'})\n95% CI for Mean: [{ci_lower:.2f}, {ci_upper:.2f}]")
            for p, val in custom_percentile_values.items():
                plot_summary[-1] += f"\n{p}: {val:.2f}" if isinstance(val, (int, float)) else f"\n{p}: {val}"
            plot_summary[-1] += f"\nDescription: Frequency distribution of values across bins.{bin_details}\n\n"

            # Create and save box plot with outliers
            plt.figure(figsize=(10, 6), dpi=100)
            plt.boxplot(df[column_name], vert=True, patch_artist=True, showfliers=True)
            plt.title(f'{column_name} Data', fontsize=14)
            plt.ylabel(f'{column_name} Values', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            # Add descriptive statistics to right side
            stats_text = f"Descriptive Statistics\nMean: {mean:.2f}\nStandard Error: {std_error:.2f}\nStandard Deviation: {std_dev:.2f}\nVariance: {variance:.2f}\nSum: {numeric_data.sum():.2f}\nCount: {count}\nMinimum: {min_val:.2f}\nMaximum: {max_val:.2f}\nRange: {range_val:.2f}\nKurtosis: {kurtosis:.2f}\nSkewness: {skewness:.2f} ({skewness_desc})\nQ1 (25th): {q1:.2f}\nQ3 (75th): {q3:.2f}\nIQR: {iqr:.2f}\nOutliers (Z>3): {outlier_count} ({', '.join(map(str, outlier_values))})\nNormality (KS test): p={ks_p_value:.4f} ({'Normal' if is_normal else 'Not Normal'})\n95% CI for Mean: [{ci_lower:.2f}, {ci_upper:.2f}]"
            for p, val in custom_percentile_values.items():
                stats_text += f"\n{p}: {val:.2f}" if isinstance(val, (int, float)) else f"\n{p}: {val}"
            plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            boxplot_path = os.path.join(output_dir, 'boxplot.png')
            plt.savefig(boxplot_path, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            plot_summary.append(f"**Plot Type: Boxplot**\nFile: {boxplot_path}\nCount: {count}\nNull Entries: {null_entries}\nMean: {mean:.2f}\nMedian: {median:.2f}\nMode: {mode}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nRange: {range_val:.2f}\nStandard Deviation: {std_dev:.2f}\nVariance: {variance:.2f}\nSkewness: {skewness:.2f} ({skewness_desc})\nKurtosis: {kurtosis:.2f}\nQ1 (25th): {q1:.2f}\nQ3 (75th): {q3:.2f}\nIQR: {iqr:.2f}\nOutliers (Z>3): {outlier_count} ({', '.join(map(str, outlier_values))})\nNormality (KS test): p={ks_p_value:.4f} ({'Normal' if is_normal else 'Not Normal'})\n95% CI for Mean: [{ci_lower:.2f}, {ci_upper:.2f}]")
            if len(moving_avg) > 1 and not np.all(np.isnan(moving_avg)):
                plot_summary[-1] += f"\nTrend Slope: {trend_slope:.4f}\nMoving Average (window={window_size}): {', '.join(map(lambda x: f'{x:.2f}', moving_avg.dropna()))}"
            for p, val in custom_percentile_values.items():
                plot_summary[-1] += f"\n{p}: {val:.2f}" if isinstance(val, (int, float)) else f"\n{p}: {val}"
            plot_summary[-1] += "\nDescription: Displays the distribution of data with quartiles, whiskers, and outliers.\n\n"

            # Create and save trend analysis plot
            if len(numeric_data) > 1 and pd.api.types.is_numeric_dtype(df.index):
                plt.figure(figsize=(12, 6), dpi=100)
                plt.plot(numeric_data.index, numeric_data, label='Original Data', marker='o')
                plt.plot(numeric_data.index, moving_avg, label=f'Moving Average (window={window_size})', color='red')
                plt.title(f'{column_name} Trend Analysis', fontsize=14)
                plt.xlabel('Index', fontsize=12)
                plt.ylabel(column_name, fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                # Annotate slope
                plt.text(0.02, 0.98, f'Slope: {trend_slope:.4f}', transform=plt.gca().transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                trend_path = os.path.join(output_dir, 'trend.png')
                plt.savefig(trend_path, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                plot_summary.append(f"**Plot Type: Trend Analysis**\nFile: {trend_path}\nCount: {count}\nTrend Slope: {trend_slope:.4f}\nMoving Average (window={window_size}): {', '.join(map(lambda x: f'{x:.2f}', moving_avg.dropna()))}\nDescription: Shows the original data and moving average to highlight trends.\n\n")

        else:
            # For non-numeric data, use Plotly for HTML and Matplotlib bar plot with stats
            value_counts = df[column_name].value_counts()
            total_unique = len(value_counts)
            total_entries = len(df[column_name])

            # Create interactive bar plot with Plotly for HTML
            fig = px.bar(x=value_counts.index, y=value_counts.values, title=f'Frequency of {column_name}',
                         labels={'x': column_name, 'y': 'Frequency'},
                         hover_data=[value_counts.index, value_counts.values])
            fig.update_traces(marker_color='skyblue')
            fig.update_layout(
                xaxis={'tickangle': 45},
                yaxis_title='Frequency',
                xaxis_title=column_name,
                height=600 + min(total_unique, 100) * 5,  # Dynamic height based on unique entries, capped at 1000
                width=1200
            )
            html_path = os.path.join(output_dir, 'frequency_bar.html')
            fig.write_html(html_path)

            # Create static bar plot with Matplotlib for PNG
            plt.figure(figsize=(max(10, total_unique / 10), 6), dpi=100)  # Dynamic width based on unique entries
            value_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'{column_name} Data', fontsize=14)
            plt.xticks(rotation=90, ha='right', fontsize=8)  # Rotate x-axis labels for readability
            plt.grid(True, linestyle='--', alpha=0.7)
            # Add descriptive statistics to right side
            stats_text = f"Descriptive Statistics\nMean: {mean}\nStandard Error: {std_error}\nStandard Deviation: {std_dev}\nVariance: {variance}\nSum: {total_entries}\nCount: {count}\nMinimum: {min_val}\nMaximum: {max_val}\nRange: {range_val}\nKurtosis: {kurtosis}\nSkewness: {skewness} ({skewness_desc})\nQ1 (25th): {q1}\nQ3 (75th): {q3}\nIQR: {iqr}"
            for p, val in custom_percentile_values.items():
                stats_text += f"\n{p}: {val}"
            plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            png_path = os.path.join(output_dir, 'frequency_bar.png')
            plt.tight_layout()
            plt.savefig(png_path, format='png', bbox_inches='tight', dpi=100)
            plt.close()

            # Add summary
            plot_summary.append(f"**Plot Type: Frequency Bar**\nFiles: {html_path}, {png_path}\nCount: {count}\nNull Entries: {null_entries}\nTotal Unique {column_name}: {total_unique}\nTotal Entries: {total_entries}\nDescription: Bar plot showing frequency of all {total_unique} {column_name} entries. Open {html_path} in a browser for exploration; {png_path} for static view.\n\n")

        # Save plot summary to text file
        summary_path = os.path.join(output_dir, 'plot_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Plot Summary\n")
            f.write("============\n\n")
            f.writelines(plot_summary)

        # Save stats summary to text file
        stats_summary_path = os.path.join(output_dir, 'stats_summary.txt')
        with open(stats_summary_path, 'w') as f:
            f.write("Statistical Analysis Summary\n")
            f.write("===========================\n\n")
            f.write('\n'.join(stats_summary))
            f.write("\n\nGenerated Plots:\n")
            if is_numeric:
                f.write("- KDE plot: kde.png\n")
                f.write("- Histogram: histogram.png\n")
                f.write("- Box plot: boxplot.png\n")
                if len(numeric_data) > 1 and pd.api.types.is_numeric_dtype(df.index):
                    f.write("- Trend plot: trend.png\n")
            else:
                f.write("- Frequency Bar: frequency_bar.html, frequency_bar.png\n")

        # Print stats summary
        print("\nStatistical Analysis Summary")
        print("===========================\n")
        print('\n'.join(stats_summary))
        print("\nGenerated Outputs:")
        print(f"- All outputs saved in: {output_dir}")
        if is_numeric:
            print("- KDE plot: kde.png")
            print("- Histogram: histogram.png")
            print("- Box plot: boxplot.png")
            if len(numeric_data) > 1 and pd.api.types.is_numeric_dtype(df.index):
                print("- Trend plot: trend.png")
        else:
            print("- Frequency Bar: frequency_bar.html, frequency_bar.png")
        print(f"- Detailed summary: plot_summary.txt")
        print(f"- Statistical summary: stats_summary.txt")
        print("\nAnalysis complete.")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Column not found in the input file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze data and generate visualization plots from a CSV or text file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV or text file containing data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (default: current directory)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Analyze the data and save results
    analyze_log2_data(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()
