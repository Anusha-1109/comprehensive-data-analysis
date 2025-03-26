#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import os

def analyze_log2_data(file_path, summary_file="plot_summary.txt"):
    """
    Analyze data from a user-provided file and save plots as PNG files in the current directory.
    For numeric data: KDE, histogram, boxplot. For non-numeric: only frequency bar plot.
    Saves a summary in a text file, without displaying plots. X-axis uses the column name from the file.
    
    Parameters:
    file_path (str): Path to the input file containing data
    summary_file (str): Name of the text file to save plot summary (default: "plot_summary.txt")
    
    Returns:
    None: Saves plots as PNG files in the current directory and writes summary to a text file
    """
    try:
        # Read the file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")

        # Print the dataframe
        print("Analysing the data")
        #print(df)

        # Get the column name (assuming single column or first column with data)
        if len(df.columns) == 0:
            raise ValueError("No columns found in the input file.")
        column_name = df.columns[0]  # Use the first column name

        # Get current working directory
        current_dir = os.getcwd()

        # List to store plot summary
        plot_summary = []

        # Check if the data is numeric
        is_numeric = pd.to_numeric(df[column_name], errors='coerce').notnull().all()

        if is_numeric:
            # Remove duplicates for numeric data
            df = df.drop_duplicates()

            # Create and save KDE plot
            plt.figure(figsize=(10, 6), dpi=100)
            df[column_name].plot.kde(bw_method=0.5)
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title(f'KDE Plot of {column_name} Values', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            kde_path = os.path.join(current_dir, 'kde.png')
            plt.savefig(kde_path, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            plot_summary.append(f"Plot Type: KDE\nFile: {kde_path}\n")

            # Create and save histogram
            plt.figure(figsize=(10, 6), dpi=100)
            df[column_name].hist(bins=20, edgecolor='black')
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Histogram of {column_name} Values', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            histogram_path = os.path.join(current_dir, 'histogram.png')
            plt.savefig(histogram_path, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            plot_summary.append(f"Plot Type: Histogram\nFile: {histogram_path}\n")

            # Create and save box plot
            plt.figure(figsize=(10, 6), dpi=100)
            plt.boxplot(df[column_name], vert=True, patch_artist=True)
            plt.title(f'Box Plot of {column_name} Values', fontsize=14)
            plt.ylabel(f'{column_name} Values', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            boxplot_path = os.path.join(current_dir, 'boxplot.png')
            plt.savefig(boxplot_path, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            plot_summary.append(f"Plot Type: Boxplot\nFile: {boxplot_path}\n")

        else:
            # For non-numeric data (e.g., gene names or chromosomes), don't remove duplicates
            value_counts = df[column_name].value_counts()  # Count occurrences of each unique value
            total_unique = len(value_counts)
            total_entries = len(df[column_name])

            # Create and save frequency bar plot
            plt.figure(figsize=(12, 6), dpi=100)
            value_counts.plot(kind='bar')  # Plot all values (no top 20 limit)
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Frequency of {column_name}', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            bar_path = os.path.join(current_dir, 'frequency_bar.png')
            plt.savefig(bar_path, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            plot_summary.append(f"Plot Type: Frequency Bar\nFile: {bar_path}\nTotal Unique {column_name}: {total_unique}\nTotal Entries: {total_entries}\n")

        # Save plot summary to text file
        summary_path = os.path.join(current_dir, summary_file)
        with open(summary_path, 'w') as f:
            f.write("Plot Summary\n")
            f.write("============\n\n")
            f.writelines(plot_summary)

        print(f"Analysis complete.")
        print(f"Plots saved in current folder: {current_dir}")
        if is_numeric:
            print(f"- KDE plot: kde.png")
            print(f"- Histogram: histogram.png")
            print(f"- Box plot: boxplot.png")
        else:
            print(f"- Frequency Bar: frequency_bar.png")
        print(f"Plot summary saved to: {summary_path}")

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

    # Parse arguments
    args = parser.parse_args()

    # Analyze the data and save results
    analyze_log2_data(args.input_file)

if __name__ == "__main__":
    main()
