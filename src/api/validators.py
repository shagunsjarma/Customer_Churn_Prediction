import re
from typing import Dict, Any, List, Optional

class DataValidator:
    """
    Data validation utilities for API requests
    """
    
    @staticmethod
    def validate_age(age: Any) -> tuple[bool, str]:
        """
        Validate age field
        """
        try:
            age_float = float(age)
            if not (18 <= age_float <= 100):
                return False, "Age must be between 18 and 100"
            return True, ""
        except (ValueError, TypeError):
            return False, "Age must be a valid number"
    
    @staticmethod
    def validate_gender(gender: Any) -> tuple[bool, str]:
        """
        Validate gender field
        """
        if not isinstance(gender, str):
            return False, "Gender must be a string"
        
        if gender not in ['Male', 'Female']:
            return False, "Gender must be either 'Male' or 'Female'"
        
        return True, ""
    
    @staticmethod
    def validate_tenure(tenure: Any) -> tuple[bool, str]:
        """
        Validate tenure field
        """
        try:
            tenure_float = float(tenure)
            if not (0 <= tenure_float <= 120):
                return False, "Tenure must be between 0 and 120 months"
            return True, ""
        except (ValueError, TypeError):
            return False, "Tenure must be a valid number"
    
    @staticmethod
    def validate_usage_frequency(usage: Any) -> tuple[bool, str]:
        """
        Validate usage frequency field
        """
        try:
            usage_float = float(usage)
            if not (0 <= usage_float <= 50):
                return False, "Usage Frequency must be between 0 and 50"
            return True, ""
        except (ValueError, TypeError):
            return False, "Usage Frequency must be a valid number"
    
    @staticmethod
    def validate_support_calls(calls: Any) -> tuple[bool, str]:
        """
        Validate support calls field
        """
        try:
            calls_float = float(calls)
            if not (0 <= calls_float <= 50):
                return False, "Support Calls must be between 0 and 50"
            return True, ""
        except (ValueError, TypeError):
            return False, "Support Calls must be a valid number"
    
    @staticmethod
    def validate_payment_delay(delay: Any) -> tuple[bool, str]:
        """
        Validate payment delay field
        """
        try:
            delay_float = float(delay)
            if not (0 <= delay_float <= 90):
                return False, "Payment Delay must be between 0 and 90 days"
            return True, ""
        except (ValueError, TypeError):
            return False, "Payment Delay must be a valid number"
    
    @staticmethod
    def validate_subscription_type(sub_type: Any) -> tuple[bool, str]:
        """
        Validate subscription type field
        """
        if not isinstance(sub_type, str):
            return False, "Subscription Type must be a string"
        
        if sub_type not in ['Basic', 'Standard', 'Premium']:
            return False, "Subscription Type must be 'Basic', 'Standard', or 'Premium'"
        
        return True, ""
    
    @staticmethod
    def validate_contract_length(length: Any) -> tuple[bool, str]:
        """
        Validate contract length field
        """
        if not isinstance(length, str):
            return False, "Contract Length must be a string"
        
        if length not in ['Monthly', 'Quarterly', 'Annual']:
            return False, "Contract Length must be 'Monthly', 'Quarterly', or 'Annual'"
        
        return True, ""
    
    @staticmethod
    def validate_total_spend(spend: Any) -> tuple[bool, str]:
        """
        Validate total spend field
        """
        try:
            spend_float = float(spend)
            if not (0 <= spend_float <= 10000):
                return False, "Total Spend must be between 0 and 10000"
            return True, ""
        except (ValueError, TypeError):
            return False, "Total Spend must be a valid number"
    
    @staticmethod
    def validate_last_interaction(interaction: Any) -> tuple[bool, str]:
        """
        Validate last interaction field
        """
        try:
            interaction_float = float(interaction)
            if not (0 <= interaction_float <= 365):
                return False, "Last Interaction must be between 0 and 365 days"
            return True, ""
        except (ValueError, TypeError):
            return False, "Last Interaction must be a valid number"
    
    @classmethod
    def validate_customer_data(cls, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate complete customer data
        Returns (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = [
            'Age', 'Gender', 'Tenure', 'Usage_Frequency', 'Support_Calls',
            'Payment_Delay', 'Subscription_Type', 'Contract_Length', 
            'Total_Spend', 'Last_Interaction'
        ]
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
            return False, errors
        
        # Validate each field
        validators = {
            'Age': cls.validate_age,
            'Gender': cls.validate_gender,
            'Tenure': cls.validate_tenure,
            'Usage_Frequency': cls.validate_usage_frequency,
            'Support_Calls': cls.validate_support_calls,
            'Payment_Delay': cls.validate_payment_delay,
            'Subscription_Type': cls.validate_subscription_type,
            'Contract_Length': cls.validate_contract_length,
            'Total_Spend': cls.validate_total_spend,
            'Last_Interaction': cls.validate_last_interaction
        }
        
        for field, validator in validators.items():
            is_valid, error_msg = validator(data[field])
            if not is_valid:
                errors.append(f"{field}: {error_msg}")
        
        return len(errors) == 0, errors

class BatchValidator:
    """
    Validator for batch prediction requests
    """
    
    @staticmethod
    def validate_batch_request(data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate batch prediction request
        """
        errors = []
        
        if 'customers' not in data:
            errors.append("Missing 'customers' field in request")
            return False, errors
        
        customers = data['customers']
        
        if not isinstance(customers, list):
            errors.append("'customers' must be a list")
            return False, errors
        
        if len(customers) == 0:
            errors.append("'customers' list cannot be empty")
            return False, errors
        
        if len(customers) > 100:
            errors.append("Maximum 100 customers allowed per batch request")
            return False, errors
        
        # Validate each customer
        for i, customer in enumerate(customers):
            if not isinstance(customer, dict):
                errors.append(f"Customer at index {i} must be a dictionary")
                continue
            
            is_valid, customer_errors = DataValidator.validate_customer_data(customer)
            if not is_valid:
                for error in customer_errors:
                    errors.append(f"Customer {i}: {error}")
        
        return len(errors) == 0, errors
