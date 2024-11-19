import json
import re
from datetime import datetime

class LogParser:
    def __init__(self, file_path):
        """Initialize with the path to the JSON file and load data."""
        self.file_path = file_path
        self.data = self.load_data_from_file()
        self.processed_data = self.process_data()

    def load_data_from_file(self):
        """Load JSON data from the file."""
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        return data

    def camel_case_to_words(self, text):
        """Convert camelCase text to spaced words."""
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    def format_date(self, date_str):
        """Convert ISO format date string to a readable date."""
        try:
            if '.' in date_str:  # Check if milliseconds are present
                date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
            else:
                date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            return date_obj.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Date format not recognized: {date_str}")
        return date_str

    def process_data(self):
        """Process JSON data for readability and return a list of formatted strings."""
        output = []
        
        def format_value(key, value):
            """Helper function to format individual key-value pairs."""
            key_readable = self.camel_case_to_words(key)
            
            if isinstance(value, bool):
                return f"{key_readable}: {'Yes' if value else 'No'}"
            elif isinstance(value, (int, float)):
                return f"{key_readable}: {value}"
            elif isinstance(value, str):
                if "date" in key.lower():
                    return f"{key_readable}: {self.format_date(value)}"
                else:
                    return f"{key_readable}: {value}"
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    nested_output = []
                    for item in value:
                        nested_output.append(", ".join(format_value(k, v) for k, v in item.items()))
                    return f"{key_readable}: [{'; '.join(nested_output)}]"
                return f"{key_readable}: []"
            elif isinstance(value, dict):
                nested_output = []
                for sub_key, sub_value in value.items():
                    nested_output.append(f"{self.camel_case_to_words(sub_key)}: {sub_value}")
                return f"{key_readable}: [{'; '.join(nested_output)}]"
            else:
                return f"{key_readable}: {value}"

        # Process each item in _source
        for key, value in self.data["_source"].items():
            output.append(format_value(key, value))
        
        return output

    def get_all_data(self):
        """Return all processed data in a structured list."""
        return self.processed_data


