import re
import json
from utils_no_quant import LLMHandler


class BondDataExtractor:
    def __init__(self):
        self.llm_handler = LLMHandler()

    def extract_bond_data(self, message: str):
        """
        Extract bond-related information from the given message.

        Args:
            message (str): Input message containing bond details.

        Returns:
            list: A list of dictionaries with bond details.
        """
        # Define regex patterns for extracting data
        isin_pattern = r"\b[A-Z]{2}[A-Z0-9]{9}\d\b"
        security_pattern = r"(\d{1,2}\.\d{1,4}%.*?\d{2,4})"
        issuer_pattern = r"\bby\s([A-Za-z\s&]+)"  # Removed look-behind
        coupon_pattern = r"(\d{1,2}\.\d{1,4})(?=%)"
        maturity_pattern = r"(\d{1,2}[ -]?[A-Za-z]{3,9}[ -]?\d{2,4})"
        quantam_pattern = r"(\d+(?:\.\d+)?)(?=\s?[Ll]acs|\s?[Mm]ultiples?)"
        offer_pattern = r"Bidding at (\d{1,2}\.\d{1,4})%"

        bonds = []

        # Find all potential bond segments in the message
        lines = message.split("\n")
        for line in lines:
            # Extract individual fields using regex
            isin = re.search(isin_pattern, line)
            security = re.search(security_pattern, line)
            issuer = re.search(issuer_pattern, line)
            coupon = re.search(coupon_pattern, line)
            maturity = re.search(maturity_pattern, line)
            quantam = re.search(quantam_pattern, line)
            offer_ytm = re.search(offer_pattern, line)

            # Parse and clean data
            bond_data = {
                "isinNo": isin.group(0) if isin else None,
                "security": security.group(0) if security else None,
                "issuer": issuer.group(1).strip() if issuer else None,
                "coupon": float(coupon.group(0)) if coupon else None,
                "maturityDate": self.parse_date(maturity.group(0)) if maturity else None,
                "offerYtm": float(offer_ytm.group(1)) if offer_ytm else None,
                "quantam": int(float(quantam.group(1))) if quantam else None,
            }

            # Add bond data if any field is non-null
            if any(value is not None for value in bond_data.values()):
                bonds.append(bond_data)

        return bonds

    @staticmethod
    def parse_date(date_str: str):
        """
        Parse various date formats into standard yyyy-mm-dd format.

        Args:
            date_str (str): Input date string.

        Returns:
            str: Formatted date string.
        """
        from datetime import datetime

        # Handle potential date formats
        date_formats = ["%d-%b-%Y", "%d %b %Y", "%b %d %Y", "%d-%m-%Y", "%Y-%m-%d"]
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.replace("-", " "), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue

        # If unable to parse, return the original string for manual correction
        return None


def main():
    extractor = BondDataExtractor()

    while True:
        # Allow input data to be provided from a file or terminal
        print("\nEnter the input method:")
        print("1: Input from a file (e.g., input.txt)")
        print("2: Provide input manually")
        print("3: Exit the program")
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == "3":
            print("Exiting the program.")
            break

        if choice == "1":
            # Read input from a file
            file_name = input("Enter the file name (e.g., input.txt): ").strip()
            try:
                with open(file_name, "r") as f:
                    input_message = f.read()
            except FileNotFoundError:
                print(f"Error: File '{file_name}' not found.")
                continue
        elif choice == "2":
            # Accept manual input
            print("Enter the bond details (end input with an empty line):")
            input_lines = []
            while True:
                line = input()
                if not line.strip():  # Stop on empty input
                    break
                input_lines.append(line)
            input_message = "\n".join(input_lines)
        else:
            print("Invalid choice. Please try again.")
            continue

        # Process the input
        extracted_data = extractor.extract_bond_data(input_message)

        # Output the result as JSON
        print("\nExtracted Bond Data:")
        print(json.dumps(extracted_data, indent=4))


if __name__ == "__main__":
    main()
