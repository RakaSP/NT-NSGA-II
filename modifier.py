import pandas as pd

# Read your CSV file
df = pd.read_csv('Problems/GIB_raw.csv')

# Add id column starting from 0
df['id'] = range(len(df))

# Add demand column - set to 0 for Basecamp, you'll need to specify for others
df['demand'] = 0

# Set demand for Basecamp specifically to 0 (or NaN if you prefer)
# For other locations, you'll need to add their demand values
# For example:
# df.loc[df['ponpes'] == 'Pondok Pesantren Al-Anwar Bolon', 'demand'] = your_value

# Save back to CSV
df.to_csv('Problems/GIB.csv', index=False)