#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend to avoid X11 issues
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

def get_plot_types(is_numeric):
    """Get user-selected plot types to be applied to both columns."""
    plot_types = []
    available_plots = ['kde', 'histogram', 'boxplot', 'trend'] if is_numeric else ['frequency_bar', 'stacked_bar']
    example_plots = 'kde, histogram' if is_numeric else 'frequency_bar, stacked_bar'
    print(f"\nSelect plot types to apply to both columns. Available: {', '.join(available_plots)}")
    print(f"Enter plot types one at a time (e.g., {example_plots}). Type 'done' to finish.")
    while True:
        plot_type = input("Enter a plot type or 'done': ").lower()
        if plot_type == 'done':
            break
        if plot_type in available_plots:
            plot_types.append(plot_type)
        else:
            print(f"Error: Invalid plot type. Choose from {', '.join(available_plots)}.")
    return plot_types if plot_types else available_plots  # Default to all available if none selected

def get_analysis_methods(is_numeric1, is_numeric2):
    """Get user-selected analysis methods based on column types."""
    if is_numeric1 and is_numeric2:
        available_methods = ['pearson', 'spearman', 'kendall', 'all', 'linear', 'polynomial', 'paired_t']
    elif not is_numeric1 and not is_numeric2:
        available_methods = ['chi2']
    else:  # One numeric, one categorical
        available_methods = ['annova']
    print(f"\nSelect analysis method(s). Available: {', '.join(available_methods)}")
    while True:
        method = input(f"Enter analysis method ({', '.join(available_methods)}): ").lower()
        if method in available_methods:
            return method
        print(f"Error: Invalid method. Choose from {', '.join(available_methods)}.")

def analyze_two_columns(file_path, output_dir=None, detect_outliers=False, skip_plotly_html=False):
    """
    Analyze data from two columns in a user-provided file and save plots and summaries
    in a timestamped directory. For numeric data: generates user-selected combined KDE, histogram, boxplot, trend analysis
    (showing both columns in a single plot), and correlation/regression analysis (Pearson's, Spearman's, Kendall's Tau,
    linear regression, polynomial regression, and/or paired t-test). For non-numeric: generates frequency bar plot,
    stacked bar plot, and chi-squared test of independence. For one numeric and one categorical: performs ANNOVA test.
    Includes descriptive statistics, percentiles, IQR, outliers (z-score if detect_outliers is True),
    normality test, confidence intervals, and trend analysis (if ordered). Saves summaries in text files and plots as
    PNG or HTML/PNG files. Prompts user for plot types (applied to both columns), analysis method, and custom percentiles.
    
    Parameters:
    file_path (str): Path to the input file containing two columns of data
    output_dir (str): Base directory to save outputs (default: current directory)
    detect_outliers (bool): Whether to perform and visualize outlier detection using z-score (default: False)
    skip_plotly_html (bool): Skip generating Plotly HTML plots to save time (default: False)
    
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
        print(f"Created output directory: {output_dir}")
        
        # Get custom percentiles from user
        custom_percentiles = get_custom_percentiles()
        print(f"Selected percentiles: {custom_percentiles}")

        # Read the file based on extension
        print("Reading input file...")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")

        # Validate that the file has exactly two columns
        if len(df.columns) != 2:
            raise ValueError("Input file must contain exactly two columns.")

        # Get column names
        col1_name, col2_name = df.columns
        print(f"Analyzing columns: {col1_name}, {col2_name}")
        print("Sample data:", df.head())

        # Clean data for both columns
        print("Cleaning data...")
        for col in df.columns:
            if pd.to_numeric(df[col], errors='coerce').notnull().all():
                # Numeric data: handle NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # Non-numeric data: strip whitespace and handle empty strings
                df[col] = df[col].astype(str).str.strip().replace('', pd.NA)

        # Determine if data is numeric for each column
        is_col1_numeric = pd.to_numeric(df[col1_name], errors='coerce').notnull().all()
        is_col2_numeric = pd.to_numeric(df[col2_name], errors='coerce').notnull().all()
        is_numeric = is_col1_numeric and is_col2_numeric
        print(f"Column {col1_name} is {'numeric' if is_col1_numeric else 'non-numeric'}")
        print(f"Column {col2_name} is {'numeric' if is_col2_numeric else 'non-numeric'}")

        # Get number of null entries for each column
        null_entries_col1 = df[col1_name].isna().sum()
        null_entries_col2 = df[col2_name].isna().sum()
        print(f"Number of null entries - {col1_name}: {null_entries_col1}, {col2_name}: {null_entries_col2}")

        # Get user-selected plot types (once for both columns)
        selected_plots = get_plot_types(is_numeric)
        print(f"Selected plot types: {selected_plots}")

        # Get user-selected analysis method based on column types
        analysis_method = get_analysis_methods(is_col1_numeric, is_col2_numeric)
        print(f"Selected analysis method: {analysis_method}")

        # Initialize summaries
        plot_summary = []
        stats_summary = []

        # Synchronize data cleaning
        print("Synchronizing data cleaning...")
        if is_numeric:
            # Drop rows where either column is NaN and remove duplicates
            df_clean = df[[col1_name, col2_name]].dropna().drop_duplicates()
        else:
            # For non-numeric or mixed, keep as is and handle NA
            df_clean = df[[col1_name, col2_name]].dropna().drop_duplicates()
        print(f"Cleaned data size: {len(df_clean)} rows")

        # Analyze each column for statistics
        print("Computing statistics for each column...")
        stats_dict = {}
        for col_name in df.columns:
            is_numeric_col = pd.to_numeric(df[col_name], errors='coerce').notnull().all()
            data = df_clean[col_name]
            null_entries = data.isna().sum()
            data = data.dropna()
            count = len(data)

            # Calculate statistics
            if is_numeric_col:
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
                kurtosis = stats.kurtosis(numeric_data, fisher=True)
                std_error = std_dev / np.sqrt(count) if count > 0 else 0
                q1 = np.percentile(numeric_data, 25)
                q3 = np.percentile(numeric_data, 75)
                iqr = q3 - q1
                custom_percentile_values = {f"P{p:.0f}": np.percentile(numeric_data, p) for p in custom_percentiles}
                skewness_desc = "Approximately symmetric" if -0.5 <= skewness <= 0.5 else \
                                "Positively skewed (right-tailed)" if skewness > 0.5 else \
                                "Negatively skewed (left-tailed)"
                z_scores = np.abs((numeric_data - mean) / std_dev)
                outliers = numeric_data[z_scores > 3] if detect_outliers else pd.Series([])
                outlier_count = len(outliers)
                outlier_values = outliers.tolist() if outlier_count > 0 else ["None"]
                ks_statistic, ks_p_value = stats.kstest(numeric_data, 'norm', args=(mean, std_dev))
                is_normal = ks_p_value > 0.05
                ci_lower, ci_upper = zconfint(numeric_data, alpha=0.05)
                
                # Trend analysis
                moving_avg = pd.Series([np.nan])
                trend_slope = np.nan
                if len(numeric_data) > 1:
                    window_size = min(5, len(numeric_data))
                    moving_avg = numeric_data.rolling(window=window_size, min_periods=1).mean()
                    trend_slope, _, _, _, _ = stats.linregress(range(len(numeric_data)), numeric_data)

                # Store stats for combined plots
                stats_dict[col_name] = {
                    'mean': mean, 'std_error': std_error, 'std_dev': std_dev, 'variance': variance,
                    'count': count, 'min_val': min_val, 'max_val': max_val, 'skewness': skewness,
                    'kurtosis': kurtosis, 'q1': q1, 'q3': q3, 'iqr': iqr, 'outlier_count': outlier_count,
                    'ks_p_value': ks_p_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
                    'custom_percentile_values': custom_percentile_values, 'null_entries': null_entries,
                    'trend_slope': trend_slope, 'moving_avg': moving_avg, 'numeric_data': numeric_data,
                    'outliers': outliers
                }

                # Stats summary for numeric data
                stats_summary.append(f"Analysis of Numeric Data: {col_name}")
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
                mean = median = min_val = max_val = range_val = std_dev = variance = skewness = kurtosis = std_error = \
                q1 = q3 = iqr = "N/A (Non-numeric)"
                custom_percentile_values = {f"P{p}": "N/A (Non-numeric)" for p in custom_percentiles}
                skewness_desc = outlier_count = ks_statistic = ks_p_value = is_normal = ci_lower = ci_upper = \
                moving_avg = trend_slope = "N/A (Non-numeric)"
                outlier_values = ["N/A (Non-numeric)"]
                mode = data.mode().tolist()
                
                # Stats summary for non-numeric data
                value_counts = data.value_counts()
                total_unique = len(value_counts)
                stats_summary.append(f"Analysis of Non-Numeric Data: {col_name}")
                stats_summary.append(f"- Total entries: {count}, Null entries: {null_entries}")
                stats_summary.append(f"- Unique values: {total_unique}")
                stats_summary.append(f"- Mode: {', '.join(map(str, mode))}")
                stats_summary.append(f"- Top 5 most frequent values:")
                for val, freq in value_counts.head(5).items():
                    stats_summary.append(f"  - {val}: {freq} occurrences")
            print(f"Finished statistics for {col_name}")

        # Generate combined descriptive plots for numeric data
        if is_numeric and selected_plots:
            print("Generating numeric data plots...")
            if 'kde' in selected_plots:
                print("Generating KDE plot...")
                plt.figure(figsize=(10, 6), dpi=100)
                df_clean[col1_name].plot.kde(label=col1_name, color='blue')
                df_clean[col2_name].plot.kde(label=col2_name, color='orange')
                plt.xlabel('Value', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.title('Combined KDE Plot', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = (f"{col1_name}:\nMean: {stats_dict[col1_name]['mean']:.2f}\nStd Dev: {stats_dict[col1_name]['std_dev']:.2f}\nCount: {stats_dict[col1_name]['count']}\n"
                              f"{col2_name}:\nMean: {stats_dict[col2_name]['mean']:.2f}\nStd Dev: {stats_dict[col2_name]['std_dev']:.2f}\nCount: {stats_dict[col2_name]['count']}")
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                kde_path = os.path.join(output_dir, 'combined_kde.png')
                try:
                    plt.savefig(kde_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Combined KDE**\nFile: {kde_path}\nCount ({col1_name}): {stats_dict[col1_name]['count']}\nCount ({col2_name}): {stats_dict[col2_name]['count']}\n"
                                        f"Description: Combined KDE plot showing probability density for {col1_name} and {col2_name}.\n")
                    print("KDE plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save KDE plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Combined KDE**\nFile: Failed to save\nDescription: Could not generate due to resource constraints.\n")

            if 'histogram' in selected_plots:
                print("Generating histogram plot...")
                plt.figure(figsize=(10, 6), dpi=100)
                bin_count = max(10, int(1 + 3.322 * np.log(len(df_clean))))
                plt.hist(df_clean[col1_name], bins=bin_count, color='blue', alpha=0.5, label=col1_name, edgecolor='black')
                plt.hist(df_clean[col2_name], bins=bin_count, color='orange', alpha=0.5, label=col2_name, edgecolor='black')
                plt.xlabel('Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title('Combined Histogram', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = (f"{col1_name}:\nMean: {stats_dict[col1_name]['mean']:.2f}\nStd Dev: {stats_dict[col1_name]['std_dev']:.2f}\nCount: {stats_dict[col1_name]['count']}\n"
                              f"{col2_name}:\nMean: {stats_dict[col2_name]['mean']:.2f}\nStd Dev: {stats_dict[col2_name]['std_dev']:.2f}\nCount: {stats_dict[col2_name]['count']}")
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                histogram_path = os.path.join(output_dir, 'combined_histogram.png')
                try:
                    plt.savefig(histogram_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    n1, bins1, _ = plt.hist(df_clean[col1_name], bins=bin_count)
                    n2, bins2, _ = plt.hist(df_clean[col2_name], bins=bin_count)
                    bin_details = (f"\nBin Details ({col1_name}):\n" + "\n".join(f"Bin Edge: {edge:.2f}, Freq: {freq}" for edge, freq in zip(bins1[:-1], n1)) +
                                   f"\nBin Details ({col2_name}):\n" + "\n".join(f"Bin Edge: {edge:.2f}, Freq: {freq}" for edge, freq in zip(bins2[:-1], n2)))
                    plot_summary.append(f"**Plot Type: Combined Histogram**\nFile: {histogram_path}\nCount ({col1_name}): {stats_dict[col1_name]['count']}\nCount ({col2_name}): {stats_dict[col2_name]['count']}\n"
                                        f"Description: Combined histogram showing frequency distributions for {col1_name} and {col2_name}.{bin_details}\n")
                    print("Histogram plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save histogram plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Combined Histogram**\nFile: Failed to save\nDescription: Could not generate due to resource constraints.\n")

            if 'boxplot' in selected_plots:
                print("Generating boxplot...")
                plt.figure(figsize=(10, 6), dpi=100)
                plt.boxplot([df_clean[col1_name].dropna(), df_clean[col2_name].dropna()], vert=True, patch_artist=True, labels=[col1_name, col2_name])
                plt.title('Combined Boxplot', fontsize=14)
                plt.ylabel('Values', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                stats_text = (f"{col1_name}:\nMean: {stats_dict[col1_name]['mean']:.2f}\nQ1: {stats_dict[col1_name]['q1']:.2f}\nQ3: {stats_dict[col1_name]['q3']:.2f}\nOutliers: {stats_dict[col1_name]['outlier_count']}\n"
                              f"{col2_name}:\nMean: {stats_dict[col2_name]['mean']:.2f}\nQ1: {stats_dict[col2_name]['q1']:.2f}\nQ3: {stats_dict[col2_name]['q3']:.2f}\nOutliers: {stats_dict[col2_name]['outlier_count']}")
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                boxplot_path = os.path.join(output_dir, 'combined_boxplot.png')
                try:
                    plt.savefig(boxplot_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Combined Boxplot**\nFile: {boxplot_path}\nCount ({col1_name}): {stats_dict[col1_name]['count']}\nCount ({col2_name}): {stats_dict[col2_name]['count']}\n"
                                        f"Description: Combined boxplot showing quartiles, whiskers, and outliers for {col1_name} and {col2_name}.\n")
                    print("Boxplot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save boxplot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Combined Boxplot**\nFile: Failed to save\nDescription: Could not generate due to resource constraints.\n")

            if 'trend' in selected_plots and len(df_clean) > 1 and pd.api.types.is_numeric_dtype(df_clean.index):
                print("Generating trend plot...")
                plt.figure(figsize=(12, 6), dpi=100)
                plt.plot(df_clean.index, df_clean[col1_name], label=f'{col1_name} Data', marker='o', color='blue')
                plt.plot(df_clean.index, stats_dict[col1_name]['moving_avg'], label=f'{col1_name} Moving Avg', color='blue', linestyle='--')
                plt.plot(df_clean.index, df_clean[col2_name], label=f'{col2_name} Data', marker='s', color='orange')
                plt.plot(df_clean.index, stats_dict[col2_name]['moving_avg'], label=f'{col2_name} Moving Avg', color='orange', linestyle='--')
                plt.title('Combined Trend Analysis', fontsize=14)
                plt.xlabel('Index', fontsize=12)
                plt.ylabel('Values', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = (f"{col1_name} Slope: {stats_dict[col1_name]['trend_slope']:.4f}\n"
                              f"{col2_name} Slope: {stats_dict[col2_name]['trend_slope']:.4f}\n")
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                trend_path = os.path.join(output_dir, 'combined_trend.png')
                try:
                    plt.savefig(trend_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Combined Trend**\nFile: {trend_path}\nCount ({col1_name}): {stats_dict[col1_name]['count']}\nCount ({col2_name}): {stats_dict[col2_name]['count']}\n"
                                        f"Slope ({col1_name}): {stats_dict[col1_name]['trend_slope']:.4f}\nSlope ({col2_name}): {stats_dict[col2_name]['trend_slope']:.4f}\n"
                                        f"Description: Combined trend plot showing data and moving averages for {col1_name} and {col2_name}.\n")
                    print("Trend plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save trend plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Combined Trend**\nFile: Failed to save\nDescription: Could not generate due to resource constraints.\n")

            # Generate outlier detection plot if enabled
            if detect_outliers:
                print("Generating outlier detection plot...")
                plt.figure(figsize=(12, 6), dpi=100)
                plt.scatter(range(len(df_clean[col1_name])), df_clean[col1_name], c='blue', label=f'{col1_name} Data', alpha=0.5)
                plt.scatter(range(len(df_clean[col2_name])), df_clean[col2_name], c='orange', label=f'{col2_name} Data', alpha=0.5)
                outliers_col1 = stats_dict[col1_name]['outliers']
                outliers_col2 = stats_dict[col2_name]['outliers']
                if not outliers_col1.empty:
                    outlier_indices_col1 = [i for i, val in enumerate(df_clean[col1_name]) if val in outliers_col1]
                    plt.scatter(outlier_indices_col1, outliers_col1, c='red', label=f'{col1_name} Outliers', marker='x')
                if not outliers_col2.empty:
                    outlier_indices_col2 = [i for i, val in enumerate(df_clean[col2_name]) if val in outliers_col2]
                    plt.scatter(outlier_indices_col2, outliers_col2, c='red', label=f'{col2_name} Outliers', marker='x')
                plt.title('Data with Outliers (Z > 3)', fontsize=14)
                plt.xlabel('Index', fontsize=12)
                plt.ylabel('Values', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = (f"{col1_name} Outliers: {stats_dict[col1_name]['outlier_count']}\n"
                              f"{col2_name} Outliers: {stats_dict[col2_name]['outlier_count']}")
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                outlier_path = os.path.join(output_dir, 'outlier_detection.png')
                try:
                    plt.savefig(outlier_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Outlier Detection**\nFile: {outlier_path}\nCount ({col1_name}): {stats_dict[col1_name]['count']}\nCount ({col2_name}): {stats_dict[col2_name]['count']}\n"
                                        f"Outliers ({col1_name}): {stats_dict[col1_name]['outlier_count']}\nOutliers ({col2_name}): {stats_dict[col2_name]['outlier_count']}\n"
                                        f"Description: Scatter plot highlighting outliers (Z > 3) for {col1_name} and {col2_name}.\n")
                    print("Outlier detection plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save outlier detection plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Outlier Detection**\nFile: Failed to save\nDescription: Could not generate due to resource constraints.\n")

        # Generate plots for non-numeric data
        if not is_numeric:
            print("Generating non-numeric data plots...")
            # Combined frequency bar plots for both columns
            if 'frequency_bar' in selected_plots:
                print("Generating combined frequency bar plot...")
                data_col1 = df_clean[col1_name].dropna()
                data_col2 = df_clean[col2_name].dropna()
                value_counts_col1 = data_col1.value_counts()
                value_counts_col2 = data_col2.value_counts()
                # Filter categories with frequency > 2
                value_counts_col1 = value_counts_col1[value_counts_col1 > 2]
                value_counts_col2 = value_counts_col2[value_counts_col2 > 2]
                # Get common categories for consistent plotting
                categories = value_counts_col1.index.union(value_counts_col2.index)
                value_counts_col1 = value_counts_col1.reindex(categories, fill_value=0)
                value_counts_col2 = value_counts_col2.reindex(categories, fill_value=0)
                total_unique_col1 = len(value_counts_col1)
                total_unique_col2 = len(value_counts_col2)
                plot_note = " (Genes with frequency > 2 only)"

                if not skip_plotly_html:
                    print("Generating Plotly HTML plot...")
                    # Create a DataFrame for Plotly
                    df_plotly = pd.DataFrame({
                        'Categories': categories,
                        col1_name: value_counts_col1,
                        col2_name: value_counts_col2
                    }).melt(id_vars='Categories', var_name='Column', value_name='Frequency')
                    fig = px.bar(df_plotly, x='Categories', y='Frequency', color='Column', barmode='group',
                                 title=f'Frequency of {col1_name} and {col2_name}{plot_note}',
                                 labels={'Categories': 'Genes', 'Frequency': 'Frequency'},
                                 color_discrete_sequence=['blue', 'orange'])
                    fig.update_layout(xaxis={'tickangle': 45}, height=600, width=1200, font_size=12)
                    html_path = os.path.join(output_dir, 'combined_frequency_bar.html')
                    try:
                        fig.write_html(html_path)
                        print("Plotly HTML plot saved")
                    except Exception as e:
                        print(f"Warning: Failed to save frequency bar HTML: {str(e)}")
                        plot_summary.append(f"**Plot Type: Combined Frequency Bar**\nFiles: Failed to save HTML\nCount ({col1_name}): {len(data_col1)}\nCount ({col2_name}): {len(data_col2)}\nUnique ({col1_name}): {total_unique_col1}\nUnique ({col2_name}): {total_unique_col2}\nDescription: Could not generate HTML plot due to resource constraints{plot_note}.\n")

                print("Generating Matplotlib PNG plot...")
                plt.figure(figsize=(14, 8), dpi=100)  # Increased size for clarity
                x = np.arange(len(categories))
                width = 0.35  # Adjusted bar width
                plt.bar(x - width/2, value_counts_col1, width, label=col1_name, color='blue', edgecolor='black')
                plt.bar(x + width/2, value_counts_col2, width, label=col2_name, color='orange', edgecolor='black')
                plt.xlabel('Genes', fontsize=14)
                plt.ylabel('Frequency', fontsize=14)
                plt.title(f'Combined Frequency Bar Plot of {col1_name} and {col2_name}{plot_note}', fontsize=16)
                plt.xticks(x, categories, rotation=45, ha='right', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                stats_text = f"Count ({col1_name}): {len(data_col1)}\nUnique ({col1_name}): {total_unique_col1}\nCount ({col2_name}): {len(data_col2)}\nUnique ({col2_name}): {total_unique_col2}\nNull Entries ({col1_name}): {null_entries_col1}\nNull Entries ({col2_name}): {null_entries_col2}"
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
                plt.tight_layout(pad=2.0)  # Add padding for better layout
                png_path = os.path.join(output_dir, 'combined_frequency_bar.png')
                try:
                    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Combined Frequency Bar**\nFiles: {html_path if not skip_plotly_html and os.path.exists(html_path) else 'No HTML'}, {png_path}\nCount ({col1_name}): {len(data_col1)}\nCount ({col2_name}): {len(data_col2)}\nUnique ({col1_name}): {total_unique_col1}\nUnique ({col2_name}): {total_unique_col2}\nDescription: Combined frequency bar plot showing distributions for {col1_name} and {col2_name}{plot_note}.\n")
                    print("Matplotlib PNG plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save frequency bar PNG: {str(e)}")
                    plot_summary.append(f"**Plot Type: Combined Frequency Bar**\nFiles: {html_path if not skip_plotly_html and os.path.exists(html_path) else 'No HTML'}, Failed to save PNG\nCount ({col1_name}): {len(data_col1)}\nCount ({col2_name}): {len(data_col2)}\nUnique ({col1_name}): {total_unique_col1}\nUnique ({col2_name}): {total_unique_col2}\nDescription: Could not generate PNG plot due to resource constraints{plot_note}.\n")

            # Stacked bar plot for non-numeric data
            if 'stacked_bar' in selected_plots:
                print("Generating stacked bar plot...")
                # Filter categories with frequency > 2
                value_counts_col1 = df_clean[col1_name].value_counts()
                value_counts_col2 = df_clean[col2_name].value_counts()
                valid_categories_col1 = value_counts_col1[value_counts_col1 > 2].index
                valid_categories_col2 = value_counts_col2[value_counts_col2 > 2].index
                # Use intersection of valid categories
                df_limited = df_clean[df_clean[col1_name].isin(valid_categories_col1) & df_clean[col2_name].isin(valid_categories_col2)]
                contingency_table = pd.crosstab(df_limited[col1_name], df_limited[col2_name])
                plot_note = " (Genes with frequency > 2 only)"
                print(f"Contingency table size: {contingency_table.shape}")
                
                plt.figure(figsize=(14, 8), dpi=100)  # Increased size for clarity
                contingency_table.plot(kind='bar', stacked=True, colormap='viridis', edgecolor='black')
                plt.xlabel(col1_name, fontsize=14)
                plt.ylabel('Count', fontsize=14)
                plt.title(f'Stacked Bar Plot of {col1_name} vs {col2_name}{plot_note}', fontsize=16)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(title=col2_name, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
                stats_text = f"Total Count: {len(df_limited)}\nUnique ({col1_name}): {len(contingency_table.index)}\nUnique ({col2_name}): {len(contingency_table.columns)}\nNull Entries ({col1_name}): {null_entries_col1}\nNull Entries ({col2_name}): {null_entries_col2}"
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
                plt.tight_layout(pad=2.0)
                stacked_bar_path = os.path.join(output_dir, 'stacked_bar.png')
                try:
                    plt.savefig(stacked_bar_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Stacked Bar**\nFile: {stacked_bar_path}\nCount: {len(df_limited)}\nUnique ({col1_name}): {len(contingency_table.index)}\nUnique ({col2_name}): {len(contingency_table.columns)}\n"
                                        f"Description: Stacked bar plot showing the distribution of {col2_name} within each category of {col1_name}{plot_note}.\n")
                    print("Stacked bar plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save stacked bar plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Stacked Bar**\nFile: Failed to save\nCount: {len(df_limited)}\nUnique ({col1_name}): {len(contingency_table.index)}\nUnique ({col2_name}): {len(contingency_table.columns)}\n"
                                        f"Description: Could not generate stacked bar plot due to resource constraints{plot_note}.\n")

        # Analysis for numeric data
        if is_col1_numeric and is_col2_numeric:
            print("Performing numeric data analysis...")
            # Clean data for analysis
            df_clean_corr = df[[col1_name, col2_name]].dropna()
            x = pd.to_numeric(df_clean_corr[col1_name])
            y = pd.to_numeric(df_clean_corr[col2_name])
            count = len(df_clean_corr)

            # Perform correlation, regression, and paired t-test analysis
            correlation_results = []
            if analysis_method in ['pearson', 'all']:
                print("Computing Pearson's correlation...")
                pearson_corr, pearson_p = stats.pearsonr(x, y)
                correlation_results.append(f"Pearson's Correlation: r={pearson_corr:.4f}, p-value={pearson_p:.4f}")
                stats_summary.append(f"Correlation Analysis (Pearson's): r={pearson_corr:.4f}, p-value={pearson_p:.4f}")
                
                plt.figure(figsize=(10, 6), dpi=100)
                plt.scatter(x, y, alpha=0.5, label='Data Points')
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), color='red', label=f'Regression Line (slope={z[0]:.2f})')
                plt.xlabel(col1_name, fontsize=12)
                plt.ylabel(col2_name, fontsize=12)
                plt.title(f"Scatter Plot with Pearson's Correlation (r={pearson_corr:.4f})", fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = f"Count: {count}\nPearson's r: {pearson_corr:.4f}\np-value: {pearson_p:.4f}"
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                pearson_plot_path = os.path.join(output_dir, 'pearson_correlation.png')
                try:
                    plt.savefig(pearson_plot_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Pearson's Correlation**\nFile: {pearson_plot_path}\nCount: {count}\nPearson's r: {pearson_corr:.4f}\np-value: {pearson_p:.4f}\nDescription: Scatter plot with regression line showing Pearson's correlation between {col1_name} and {col2_name}.\n")
                    print("Pearson's correlation plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save Pearson's correlation plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Pearson's Correlation**\nFile: Failed to save\nCount: {count}\nDescription: Could not generate plot due to resource constraints.\n")

            if analysis_method in ['spearman', 'all']:
                print("Computing Spearman's correlation...")
                spearman_corr, spearman_p = stats.spearmanr(x, y)
                correlation_results.append(f"Spearman's Correlation: rho={spearman_corr:.4f}, p-value={spearman_p:.4f}")
                stats_summary.append(f"Correlation Analysis (Spearman's): rho={spearman_corr:.4f}, p-value={spearman_p:.4f}")
                
                plt.figure(figsize=(10, 6), dpi=100)
                plt.scatter(x, y, alpha=0.5, label='Data Points')
                plt.xlabel(col1_name, fontsize=12)
                plt.ylabel(col2_name, fontsize=12)
                plt.title(f"Scatter Plot with Spearman's Correlation (rho={spearman_corr:.4f})", fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = f"Count: {count}\nSpearman's rho: {spearman_corr:.4f}\np-value: {spearman_p:.4f}"
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                spearman_plot_path = os.path.join(output_dir, 'spearman_correlation.png')
                try:
                    plt.savefig(spearman_plot_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Spearman's Correlation**\nFile: {spearman_plot_path}\nCount: {count}\nSpearman's rho: {spearman_corr:.4f}\np-value: {spearman_p:.4f}\nDescription: Scatter plot showing Spearman's correlation between {col1_name} and {col2_name}.\n")
                    print("Spearman's correlation plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save Spearman's correlation plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Spearman's Correlation**\nFile: Failed to save\nCount: {count}\nDescription: Could not generate plot due to resource constraints.\n")

            if analysis_method in ['kendall', 'all']:
                print("Computing Kendall's Tau correlation...")
                kendall_corr, kendall_p = stats.kendalltau(x, y)
                correlation_results.append(f"Kendall's Tau Correlation: tau={kendall_corr:.4f}, p-value={kendall_p:.4f}")
                stats_summary.append(f"Correlation Analysis (Kendall's Tau): tau={kendall_corr:.4f}, p-value={kendall_p:.4f}")
                
                plt.figure(figsize=(10, 6), dpi=100)
                plt.scatter(x, y, alpha=0.5, label='Data Points')
                plt.xlabel(col1_name, fontsize=12)
                plt.ylabel(col2_name, fontsize=12)
                plt.title(f"Scatter Plot with Kendall's Tau Correlation (tau={kendall_corr:.4f})", fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = f"Count: {count}\nKendall's tau: {kendall_corr:.4f}\np-value: {kendall_p:.4f}"
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                kendall_plot_path = os.path.join(output_dir, 'kendall_correlation.png')
                try:
                    plt.savefig(kendall_plot_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Kendall's Tau Correlation**\nFile: {kendall_plot_path}\nCount: {count}\nKendall's tau: {kendall_corr:.4f}\np-value: {kendall_p:.4f}\nDescription: Scatter plot showing Kendall's Tau correlation between {col1_name} and {col2_name}.\n")
                    print("Kendall's correlation plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save Kendall's correlation plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Kendall's Tau Correlation**\nFile: Failed to save\nCount: {count}\nDescription: Could not generate plot due to resource constraints.\n")

            if analysis_method in ['linear', 'all']:
                print("Computing linear regression...")
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                regression_line = slope * x + intercept
                correlation_results.append(f"Linear Regression: slope={slope:.4f}, intercept={intercept:.4f}, r={r_value:.4f}, p={p_value:.4f}, std_err={std_err:.4f}")
                stats_summary.append(f"Regression Analysis (Linear): slope={slope:.4f}, intercept={intercept:.4f}, r={r_value:.4f}, p={p_value:.4f}, std_err={std_err:.4f}")
                
                plt.figure(figsize=(10, 6), dpi=100)
                plt.scatter(x, y, alpha=0.5, label='Data Points')
                plt.plot(x, regression_line, color='green', label=f'Linear Fit (slope={slope:.2f})')
                plt.xlabel(col1_name, fontsize=12)
                plt.ylabel(col2_name, fontsize=12)
                plt.title(f"Scatter Plot with Linear Regression (r={r_value:.4f})", fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = f"Count: {count}\nSlope: {slope:.4f}\nIntercept: {intercept:.4f}\nr: {r_value:.4f}\np: {p_value:.4f}\nStd Err: {std_err:.4f}"
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                linear_plot_path = os.path.join(output_dir, 'linear_regression.png')
                try:
                    plt.savefig(linear_plot_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Linear Regression**\nFile: {linear_plot_path}\nCount: {count}\nSlope: {slope:.4f}\nIntercept: {intercept:.4f}\nr: {r_value:.4f}\np: {p_value:.4f}\nStd Err: {std_err:.4f}\nDescription: Scatter plot with linear regression line between {col1_name} and {col2_name}.\n")
                    print("Linear regression plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save linear regression plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Linear Regression**\nFile: Failed to save\nCount: {count}\nDescription: Could not generate plot due to resource constraints.\n")

            if analysis_method in ['polynomial', 'all']:
                print("Computing polynomial regression...")
                poly_degree = 2
                coeffs = np.polyfit(x, y, poly_degree)
                poly = np.poly1d(coeffs)
                correlation_results.append(f"Polynomial Regression (degree={poly_degree}): coeffs={coeffs}")
                stats_summary.append(f"Regression Analysis (Polynomial, degree={poly_degree}): coeffs={coeffs}")
                
                plt.figure(figsize=(10, 6), dpi=100)
                x_smooth = np.linspace(min(x), max(x), 100)
                plt.scatter(x, y, alpha=0.5, label='Data Points')
                plt.plot(x_smooth, poly(x_smooth), color='purple', label=f'Polynomial Fit (deg={poly_degree})')
                plt.xlabel(col1_name, fontsize=12)
                plt.ylabel(col2_name, fontsize=12)
                plt.title(f"Scatter Plot with Polynomial Regression (degree={poly_degree})", fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                stats_text = f"Count: {count}\nCoefficients: {coeffs}\nDegree: {poly_degree}"
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                poly_plot_path = os.path.join(output_dir, 'polynomial_regression.png')
                try:
                    plt.savefig(poly_plot_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Polynomial Regression**\nFile: {poly_plot_path}\nCount: {count}\nCoefficients: {coeffs}\nDegree: {poly_degree}\nDescription: Scatter plot with polynomial regression curve (degree={poly_degree}) between {col1_name} and {col2_name}.\n")
                    print("Polynomial regression plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save polynomial regression plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Polynomial Regression**\nFile: Failed to save\nCount: {count}\nDescription: Could not generate plot due to resource constraints.\n")

            if analysis_method in ['paired_t', 'all']:
                print("Computing paired t-test...")
                t_stat, p_value = stats.ttest_rel(x, y)
                correlation_results.append(f"Paired T-Test: t={t_stat:.4f}, p-value={p_value:.4f}")
                stats_summary.append(f"T-Test Analysis (Paired): t={t_stat:.4f}, p-value={p_value:.4f}")
                
                means = [x.mean(), y.mean()]
                std_errors = [stats_dict[col1_name]['std_error'], stats_dict[col2_name]['std_error']]
                plt.figure(figsize=(8, 6), dpi=100)
                plt.bar([col1_name, col2_name], means, yerr=std_errors, capsize=5, color=['blue', 'orange'], alpha=0.7)
                plt.xlabel('Columns', fontsize=12)
                plt.ylabel('Mean Value', fontsize=12)
                plt.title(f'Paired T-Test: Mean Comparison (t={t_stat:.4f}, p={p_value:.4f})', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                stats_text = f"Count: {count}\nt-statistic: {t_stat:.4f}\np-value: {p_value:.4f}"
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ttest_plot_path = os.path.join(output_dir, 'paired_t_test.png')
                try:
                    plt.savefig(ttest_plot_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Paired T-Test**\nFile: {ttest_plot_path}\nCount: {count}\nt-statistic: {t_stat:.4f}\np-value: {p_value:.4f}\nDescription: Bar plot comparing means of {col1_name} and {col2_name} with paired t-test results.\n")
                    print("Paired t-test plot saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save paired t-test plot: {str(e)}")
                    plot_summary.append(f"**Plot Type: Paired T-Test**\nFile: Failed to save\nCount: {count}\nDescription: Could not generate plot due to resource constraints.\n")

            # Print analysis results
            print("\nAnalysis Results:")
            for result in correlation_results:
                print(result)
        elif not is_col1_numeric and not is_col2_numeric:
            # Chi-squared test for non-numeric data
            if analysis_method == 'chi2':
                print("Performing chi-squared test...")
                # Filter categories with frequency > 2 for chi-squared test
                value_counts_col1 = df_clean[col1_name].value_counts()
                value_counts_col2 = df_clean[col2_name].value_counts()
                valid_categories_col1 = value_counts_col1[value_counts_col1 > 2].index
                valid_categories_col2 = value_counts_col2[value_counts_col2 > 2].index
                df_chi2 = df_clean[df_clean[col1_name].isin(valid_categories_col1) & df_clean[col2_name].isin(valid_categories_col2)]
                contingency_table = pd.crosstab(df_chi2[col1_name], df_chi2[col2_name])
                plot_note = " (Genes with frequency > 2 only)"
                print(f"Contingency table size for chi-squared: {contingency_table.shape}")
                
                # Perform chi-squared test
                try:
                    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    stats_summary.append(f"Chi-Squared Test: chi2={chi2_stat:.4f}, p-value={p_value:.4f}, dof={dof}{plot_note}")
                    print(f"Chi-squared test completed: chi2={chi2_stat:.4f}, p-value={p_value:.4f}, dof={dof}")
                except Exception as e:
                    chi2_stat, p_value, dof = np.nan, np.nan, np.nan
                    stats_summary.append(f"Chi-Squared Test: Failed to compute due to {str(e)}{plot_note}")
                    print(f"Warning: Chi-squared test failed: {str(e)}")
                
                # Generate heatmap for chi-squared test
                print("Generating chi-squared heatmap...")
                plt.figure(figsize=(14, 8), dpi=100)  # Increased size for clarity
                plt.imshow(contingency_table, interpolation='nearest', cmap='Blues')
                plt.colorbar(label='Count')
                plt.xticks(np.arange(len(contingency_table.columns)), contingency_table.columns, rotation=45, ha='right', fontsize=12)
                plt.yticks(np.arange(len(contingency_table.index)), contingency_table.index, fontsize=12)
                plt.xlabel(col2_name, fontsize=14)
                plt.ylabel(col1_name, fontsize=14)
                plt.title(f'Chi-Squared Test Heatmap (chi2={chi2_stat:.4f}, p={p_value:.4f}){plot_note}', fontsize=16)
                # Limit annotations for large tables
                if contingency_table.shape[0] * contingency_table.shape[1] <= 50:  # Reduced threshold
                    for i in range(len(contingency_table.index)):
                        for j in range(len(contingency_table.columns)):
                            plt.text(j, i, contingency_table.iloc[i, j], ha='center', va='center', color='black', fontsize=10)
                stats_text = f"Count: {len(df_chi2)}\nChi2: {chi2_stat:.4f}\np-value: {p_value:.4f}\nDOF: {dof}\nNull Entries ({col1_name}): {null_entries_col1}\nNull Entries ({col2_name}): {null_entries_col2}"
                plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
                plt.tight_layout(pad=2.0)
                chi2_plot_path = os.path.join(output_dir, 'chi2_heatmap.png')
                try:
                    plt.savefig(chi2_plot_path, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                    plot_summary.append(f"**Plot Type: Chi-Squared Test Heatmap**\nFile: {chi2_plot_path}\nCount: {len(df_chi2)}\nChi2: {chi2_stat:.4f}\np-value: {p_value:.4f}\nDOF: {dof}\nDescription: Heatmap of contingency table with chi-squared test results for {col1_name} and {col2_name}{plot_note}.\n")
                    print("Chi-squared heatmap saved")
                except Exception as e:
                    plt.close()
                    print(f"Warning: Failed to save chi-squared heatmap: {str(e)}")
                    plot_summary.append(f"**Plot Type: Chi-Squared Test Heatmap**\nFile: Failed to save\nCount: {len(df_chi2)}\nDescription: Could not generate plot due to resource constraints{plot_note}.\n")
                print(f"\nChi-Squared Test: chi2={chi2_stat:.4f}, p-value={p_value:.4f}, dof={dof}")
        elif (is_col1_numeric and not is_col2_numeric) or (not is_col1_numeric and is_col2_numeric):
            # Determine which column is numeric and which is categorical
            numeric_col = col1_name if is_col1_numeric else col2_name
            categorical_col = col2_name if is_col1_numeric else col1_name
            numeric_data = pd.to_numeric(df_clean[numeric_col])
            categorical_data = df_clean[categorical_col]

            if analysis_method == 'annova':
                print("Performing ANNOVA...")
                # Perform one-way ANNOVA
                try:
                    f_stat, p_value = stats.f_oneway(*[numeric_data[categorical_data == cat] for cat in categorical_data.unique()])
                    stats_summary.append(f"ANNOVA: F={f_stat:.4f}, p-value={p_value:.4f}")
                    print(f"ANNOVA completed: F={f_stat:.4f}, p-value={p_value:.4f}")

                    # Generate boxplot for ANNOVA visualization
                    plt.figure(figsize=(12, 8), dpi=100)  # Increased size for more categories
                    df_clean.boxplot(column=numeric_col, by=categorical_col, grid=True, patch_artist=True)
                    plt.title(f'Boxplot of {numeric_col} by {categorical_col} (ANNOVA: F={f_stat:.4f}, p={p_value:.4f})', fontsize=14)
                    plt.suptitle('')  # Remove default title
                    plt.xlabel(categorical_col, fontsize=12)
                    plt.ylabel(numeric_col, fontsize=12)
                    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
                    stats_text = f"Count: {len(numeric_data)}\nF-statistic: {f_stat:.4f}\np-value: {p_value:.4f}"
                    plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, verticalalignment='center',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    annova_plot_path = os.path.join(output_dir, 'annova_boxplot.png')
                    try:
                        plt.savefig(annova_plot_path, format='png', bbox_inches='tight', dpi=100)
                        plt.close()
                        plot_summary.append(f"**Plot Type: ANNOVA Boxplot**\nFile: {annova_plot_path}\nCount: {len(numeric_data)}\nF-statistic: {f_stat:.4f}\np-value: {p_value:.4f}\nDescription: Boxplot showing {numeric_col} distribution across {categorical_col} categories with ANNOVA results.\n")
                        print("ANNOVA boxplot saved")
                    except Exception as e:
                        plt.close()
                        print(f"Warning: Failed to save ANNOVA boxplot: {str(e)}")
                        plot_summary.append(f"**Plot Type: ANNOVA Boxplot**\nFile: Failed to save\nCount: {len(numeric_data)}\nDescription: Could not generate plot due to resource constraints.\n")
                except Exception as e:
                    stats_summary.append(f"ANNOVA: Failed to compute due to {str(e)}")
                    print(f"Warning: ANNOVA failed: {str(e)}")
        else:
            stats_summary.append("Analysis: Not performed (incompatible column types for selected method)")
            print("\nAnalysis: Skipped (incompatible column types for selected method)")

        # Save summaries
        print("Saving summaries...")
        summary_path = os.path.join(output_dir, 'plot_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Plot Summary\n============\n\n")
            f.writelines(s + '\n' for s in plot_summary)
        print("Plot summary saved")

        stats_summary_path = os.path.join(output_dir, 'stats_summary.txt')
        with open(stats_summary_path, 'w') as f:
            f.write("Statistical Analysis Summary\n===========================\n\n")
            f.write('\n'.join(stats_summary))
        print("Stats summary saved")

        # Print results
        print("\nStatistical Analysis Summary")
        print("===========================\n")
        print('\n'.join(stats_summary))
        print("\nGenerated Outputs:")
        print(f"- All outputs saved in: {output_dir}")
        print(f"- Detailed summary: plot_summary.txt")
        print(f"- Statistical summary: stats_summary.txt")
        print("\nAnalysis complete.")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        sys.exit(1)
    except ValueError as ve:
        print(f"Error: {str(ve)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze two columns of data and generate user-selected combined visualization plots (applied to both columns) and user-selected correlation/regression/t-test/chi-squared/ANNOVA analysis."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV or text file with two columns"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (default: current directory)"
    )
    parser.add_argument(
        "--detect-outliers",
        action='store_true',
        default=False,
        help="Enable outlier detection using z-score (default: False)"
    )
    parser.add_argument(
        "--skip-plotly-html",
        action='store_true',
        default=False,
        help="Skip generating Plotly HTML plots to save time (default: False)"
    )
    args = parser.parse_args()
    analyze_two_columns(args.input_file, args.output_dir, args.detect_outliers, args.skip_plotly_html)

if __name__ == "__main__":
    main()
