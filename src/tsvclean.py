import pandas as pd

# Read the TSV file
df = pd.read_csv("../testfiles/CV000/CV000.tsv", sep="\t")

# Select the second and third columns
selected_columns = df.iloc[:, [1, 2]]

# Write the selected columns to a new TSV file
selected_columns.to_csv("cleanCV000.tsv", sep="\t", index=False)
