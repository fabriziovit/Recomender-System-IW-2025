import pandas as pd


def filter_unavailable_abstracts(output_file):
    # Read CSV into a DataFrame
    df = pd.read_csv("Datasets_dbpedia/movies_with_abstracts_complete.csv", dtype=str, quotechar='"', on_bad_lines="skip")

    # Filter rows where dbpedia_abstract is "Abstract not available"
    filtered_df = df[(df["dbpedia_abstract"] == "Abstract not available") | (df["dbpedia_abstract"].isnull())]

    # Save result to a new CSV
    filtered_df.to_csv(output_file, index=False)

    print(f"File saved: {output_file}")


# Example usage
output_csv = "output.csv"
filter_unavailable_abstracts(output_csv)
