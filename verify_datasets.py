"""Verify DI values for all three sample datasets."""
import pandas as pd

def check_di(csv_path, pos_label):
    df = pd.read_csv(csv_path)
    print(f"\nFile: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Decisions: {df['Decision'].unique()}")
    male_rate = (df[df['Gender'] == 'Male']['Decision'].str.lower() == pos_label.lower()).mean()
    fem_rate  = (df[df['Gender'] == 'Female']['Decision'].str.lower() == pos_label.lower()).mean()
    print(f"Male selection rate:   {male_rate:.4f} ({male_rate*100:.1f}%)")
    print(f"Female selection rate: {fem_rate:.4f}  ({fem_rate*100:.1f}%)")
    if max(male_rate, fem_rate) > 0:
        di = min(male_rate, fem_rate) / max(male_rate, fem_rate)
        print(f"DI = {di:.4f}")
        if di >= 0.80:
            print("-> Expected: Low Risk  | DI >= 0.80 FAIR")
        elif di >= 0.50:
            print("-> Expected: Medium Risk | 0.50 <= DI < 0.80 MODERATE BIAS")
        else:
            print("-> Expected: High Risk | DI < 0.50 SEVERE BIAS")
        return di
    else:
        print("DI = undefined (max rate is 0)")
        return None

check_di('datasets/balanced_dataset.csv', 'Approved')
check_di('datasets/moderate_bias_dataset.csv', 'Selected')
check_di('datasets/extreme_bias_dataset.csv', 'Accepted')
