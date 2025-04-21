import pandas as pd
import numpy as np
from datetime import timedelta

# Assuming df_eda is the dataframe from the previous step
df_fe = df_eda.copy()

# --- Create the Target Variable ---
# Define the trial end date (15 days after initial access)
df_fe['trial_end_date'] = df_fe['initial_access'] + timedelta(days=15)

# Define the end of the 7-day conversion window (22 days after initial access)
df_fe['conversion_window_end_date'] = df_fe['initial_access'] + timedelta(days=22)

# Create the target variable 'converted_within_7_days'
# Condition 1: User subscribed (subscription_status == 1)
# Condition 2: Subscription date is not null
# Condition 3: Subscription date is AFTER the trial ends (>= day 16)
# Condition 4: Subscription date is WITHIN the 7-day window (<= day 22)
df_fe['converted_within_7_days'] = np.where(
    (df_fe['subscription_status'] == 1) &
    (df_fe['subscription_date'].notna()) &
    (df_fe['subscription_date'] > df_fe['trial_end_date']) &
    (df_fe['subscription_date'] <= df_fe['conversion_window_end_date']),
    1,  # Converted within the window
    0   # Did not convert within the window (or never subscribed)
)

print("\n--- Target Variable Creation ---")
print("Value Counts for 'converted_within_7_days':")
print(df_fe['converted_within_7_days'].value_counts(normalize=True).to_markdown(numalign="left", stralign="left"))

# --- Time-Based Features ---
df_fe['initial_access_dayofweek'] = df_fe['initial_access'].dt.dayofweek
df_fe['initial_access_month'] = df_fe['initial_access'].dt.month
df_fe['initial_access_dayofyear'] = df_fe['initial_access'].dt.dayofyear # Example

# --- Engagement Metrics (Using Existing/Derived Features) ---
# Count number of screens viewed (simple approach)
df_fe['num_screens_viewed'] = df_fe['viewed_screens'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

# Average session duration (if multiple sessions were aggregated, otherwise just use session_duration)
# Assuming 'session_duration' is per user for the relevant period
df_fe['avg_session_duration'] = df_fe['session_duration'] # Renaming for clarity, assuming it's representative

# --- Behavioral Patterns / Interaction Terms ---
# Example: Interaction between age and screen count
df_fe['age_x_screen_count'] = df_fe['user_age'] * df_fe['screen_count']
# Example: Ratio of premium features used to total screens viewed (avoid division by zero)
df_fe['premium_use_ratio'] = (df_fe['premium_features_used'] / df_fe['screen_count'].replace(0, 1)).fillna(0)


# --- Feature Engineering Complete ---
print("\n--- Feature Engineering Summary ---")
print("New features created:")
print("- converted_within_7_days (Target Variable)")
print("- trial_end_date")
print("- conversion_window_end_date")
print("- initial_access_dayofweek")
print("- initial_access_month")
print("- initial_access_dayofyear")
print("- num_screens_viewed")
print("- avg_session_duration")
print("- age_x_screen_count")
print("- premium_use_ratio")

print("\nDataFrame shape after Feature Engineering:", df_fe.shape)
print(df_fe[['initial_access', 'subscription_date', 'trial_end_date', 'conversion_window_end_date', 'subscription_status', 'converted_within_7_days']].head().to_markdown(index=False))

# Keep the dataframe for the next steps
df_processed = df_fe.copy()
```
> * **Code Description:** This script calculates the trial end date and the conversion window end date based on `initial_access`. It then creates the binary target variable `converted_within_7_days` according to the specified logic (subscribed AND subscription date within days 16-22). It also extracts time-based features from `initial_access`, calculates the number of screens viewed from the `viewed_screens` string, and creates example interaction terms.
> * **Next Steps:** The next stage involves preparing the data for modeling: handling missing values, encoding categorical features, scaling numerical features, and finally, selecting the most relevant featur