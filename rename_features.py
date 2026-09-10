import os
import pandas as pd

datasets = ['creditcard.csv', 'balanced_creditcard.csv']

# Meaningful names for V1 to V28
feature_mapping = {
    'V1': 'location_city',
    'V2': 'location_country',
    'V3': 'device_type',
    'V4': 'device_os',
    'V5': 'browser_type',
    'V6': 'ip_address_type',
    'V7': 'merchant_category',
    'V8': 'merchant_id',
    'V9': 'transaction_channel',
    'V10': 'card_present_status',
    'V11': 'pin_entered_status',
    'V12': 'contactless_used',
    'V13': 'distance_from_home',
    'V14': 'distance_from_last_tx',
    'V15': 'time_since_last_tx',
    'V16': 'account_age_days',
    'V17': 'card_age_days',
    'V18': 'daily_tx_count',
    'V19': 'daily_tx_amount',
    'V20': 'failed_tx_count_24h',
    'V21': 'is_vpn_used',
    'V22': 'is_proxy_used',
    'V23': 'billing_shipping_match',
    'V24': 'user_risk_score',
    'V25': 'email_domain_risk',
    'V26': 'phone_risk_score',
    'V27': 'address_risk_score',
    'V28': 'date'
}

for filename in datasets:
    if os.path.exists(filename):
        print(f"Processing {filename}...")
        
        # Read the file
        df = pd.read_csv(filename)
        
        # Rename the columns
        df.rename(columns=feature_mapping, inplace=True)
        
        # Save back to CSV
        df.to_csv(filename, index=False)
        print(f"Successfully updated headers in {filename}")
    else:
        print(f"File {filename} not found.")

print("All tasks completed.")
