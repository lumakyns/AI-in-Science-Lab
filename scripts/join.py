import pandas as pd

# ====== CONFIG ======
csv_path_1 = "/home/kyns/Desktop/code/research/AI-In-Science-Lab/cnn_redundancy/results/luca_fn/redundancy_speed_mean.csv"
csv_path_2 = "/home/kyns/Desktop/code/research/AI-In-Science-Lab/cnn_redundancy/results/max_fn/redundancy_speed_mean.csv"

join_columns = ["kernel_radius", "b", "c", "h", "w"]
output_path = "joined.csv"
# ====================

def main():
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    join_columns = ["kernel_radius", "b", "c", "h", "w"]

    df1 = df1[join_columns + ["luca_fn_sec", "luca_fn_value"]]
    df2 = df2[join_columns + ["max_fn_sec", "max_fn_value"]]

    merged = df1.merge(df2, on=join_columns, how="inner")

    merged.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()