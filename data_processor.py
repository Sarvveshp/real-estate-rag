import pandas as pd
import re
from typing import List, Dict, Tuple
import numpy as np

def normalize_price(price_str: str) -> float:
    """Convert price strings to numeric values, handling various formats:
    - Indian-style commas (e.g., '92,50,000')
    - Crore notation (e.g., '1.2 Cr')
    - Lakh notation (e.g., '80L')
    - Simple numeric values
    """
    # Remove currency symbol and whitespace
    price_str = price_str.replace('₹', '').strip()
    
    # Handle Indian-style comma separators
    if ',' in price_str:
        # Remove commas and convert to float
        try:
            return float(price_str.replace(',', ''))
        except ValueError:
            pass
    
    # Handle Crore notation
    if 'Cr' in price_str:
        return float(price_str.replace('Cr', '').strip()) * 10000000
    
    # Handle Lakh notation
    if 'L' in price_str:
        return float(price_str.replace('L', '').strip()) * 100000
    
    # Handle simple numeric values
    try:
        return float(price_str)
    except ValueError:
        raise ValueError(f"Could not parse price: {price_str}")

def parse_amenities(amenities_str: str) -> List[str]:
    """Convert comma-separated amenities string to list."""
    return [amen.strip() for amen in amenities_str.split(',') if amen.strip()]

def parse_nearby(nearby_str: str) -> List[str]:
    """Convert comma-separated nearby locations string to list."""
    return [loc.strip() for loc in nearby_str.split(',') if loc.strip()]

class PropertyProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        
    def create_property_metadata(self, row: pd.Series) -> Dict:
        """Create metadata dictionary for FAISS storage."""
        return {
            'property_id': row['Property ID'],
            'location': row['Location'],
            'bhk': row['BHK'],
            'price': row['Start Price'],
            'furnishing': row['Furnishing'],
            'amenities': row['Amenities'],
            'nearby': row['Nearby']
        }
    
    def preprocess_csv(self) -> pd.DataFrame:
        """Load and preprocess the properties CSV."""
        df = pd.read_csv(self.csv_path)
        
        # Normalize prices
        df['Start Price'] = df['Start Price'].apply(normalize_price)
        
        # Convert amenities and nearby to lists
        df['Amenities'] = df['Amenities'].apply(parse_amenities)
        df['Nearby'] = df['Nearby'].apply(parse_nearby)
        
        # Filter out sold properties
        df = df[df['Status'] != 'Sold']
        
        # Convert BHK to numeric
        df['BHK'] = df['BHK'].str.extract('(\d+)').astype(float)
        
        return df

def create_property_metadata(row: pd.Series) -> Dict:
    """Create metadata dictionary for FAISS storage."""
    return {
        'property_id': row['Property ID'],
        'location': row['Location'],
        'bhk': row['BHK'],
        'price': row['Start Price'],
        'furnishing': row['Furnishing'],
        'amenities': row['Amenities'],
        'nearby': row['Nearby']
    }
