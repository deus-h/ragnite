"""
Date Filter Builder

This module provides the DateFilterBuilder class for constructing filters
based on date and time fields for vector database queries.
"""

import logging
import copy
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

from .base_filter_builder import BaseFilterBuilder

# Configure logging
logger = logging.getLogger(__name__)

class DateFilterBuilder(BaseFilterBuilder):
    """
    Filter builder for constructing filters based on date and time fields.
    
    This builder allows for creating filters that match specific dates, date ranges,
    relative dates, and more complex date-based conditions. It supports
    various vector database formats and provides a unified interface.
    """
    
    def __init__(self):
        """Initialize the date filter builder."""
        super().__init__()
        self._conditions = []
    
    def equals(self, field: str, date_value: Union[str, datetime.datetime, datetime.date]):
        """
        Add an equals condition to the filter.
        
        Args:
            field: Date field name
            date_value: Date to match
            
        Returns:
            self: For method chaining
        """
        date_value = self._ensure_date(date_value)
        self._conditions.append({
            "type": "equals", 
            "field": field, 
            "value": date_value
        })
        return self
    
    def not_equals(self, field: str, date_value: Union[str, datetime.datetime, datetime.date]):
        """
        Add a not equals condition to the filter.
        
        Args:
            field: Date field name
            date_value: Date to not match
            
        Returns:
            self: For method chaining
        """
        date_value = self._ensure_date(date_value)
        self._conditions.append({
            "type": "not_equals", 
            "field": field, 
            "value": date_value
        })
        return self
    
    def before(self, field: str, date_value: Union[str, datetime.datetime, datetime.date], inclusive: bool = False):
        """
        Add a before condition to the filter.
        
        Args:
            field: Date field name
            date_value: Date to compare against
            inclusive: Whether to include the given date in the comparison
            
        Returns:
            self: For method chaining
        """
        date_value = self._ensure_date(date_value)
        self._conditions.append({
            "type": "lt" if not inclusive else "lte", 
            "field": field, 
            "value": date_value
        })
        return self
    
    def after(self, field: str, date_value: Union[str, datetime.datetime, datetime.date], inclusive: bool = False):
        """
        Add an after condition to the filter.
        
        Args:
            field: Date field name
            date_value: Date to compare against
            inclusive: Whether to include the given date in the comparison
            
        Returns:
            self: For method chaining
        """
        date_value = self._ensure_date(date_value)
        self._conditions.append({
            "type": "gt" if not inclusive else "gte", 
            "field": field, 
            "value": date_value
        })
        return self
    
    def between(self, field: str, 
                start_date: Union[str, datetime.datetime, datetime.date], 
                end_date: Union[str, datetime.datetime, datetime.date], 
                inclusive: bool = True):
        """
        Add a between condition to the filter.
        
        Args:
            field: Date field name
            start_date: Start date
            end_date: End date
            inclusive: Whether the range is inclusive or exclusive
            
        Returns:
            self: For method chaining
        """
        start_date = self._ensure_date(start_date)
        end_date = self._ensure_date(end_date)
        
        self._conditions.append({
            "type": "between",
            "field": field,
            "start_date": start_date,
            "end_date": end_date,
            "inclusive": inclusive
        })
        return self
    
    def on_date(self, field: str, date_value: Union[str, datetime.datetime, datetime.date]):
        """
        Add a condition to match a specific date (ignoring time components).
        
        Args:
            field: Date field name
            date_value: Date to match
            
        Returns:
            self: For method chaining
        """
        date_value = self._ensure_date(date_value)
        # If it's a datetime, set time to midnight
        if isinstance(date_value, datetime.datetime):
            start_of_day = date_value.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + datetime.timedelta(days=1) - datetime.timedelta(microseconds=1)
        else:  # It's a date
            start_of_day = datetime.datetime.combine(date_value, datetime.time.min)
            end_of_day = datetime.datetime.combine(date_value, datetime.time.max)
            
        self._conditions.append({
            "type": "between",
            "field": field,
            "start_date": start_of_day,
            "end_date": end_of_day,
            "inclusive": True
        })
        return self
    
    def in_date_list(self, field: str, date_values: List[Union[str, datetime.datetime, datetime.date]]):
        """
        Add an in list condition to the filter.
        
        Args:
            field: Date field name
            date_values: List of dates to match against
            
        Returns:
            self: For method chaining
        """
        date_values = [self._ensure_date(date) for date in date_values]
        self._conditions.append({
            "type": "in", 
            "field": field, 
            "values": date_values
        })
        return self
    
    def not_in_date_list(self, field: str, date_values: List[Union[str, datetime.datetime, datetime.date]]):
        """
        Add a not in list condition to the filter.
        
        Args:
            field: Date field name
            date_values: List of dates to not match against
            
        Returns:
            self: For method chaining
        """
        date_values = [self._ensure_date(date) for date in date_values]
        self._conditions.append({
            "type": "not_in", 
            "field": field, 
            "values": date_values
        })
        return self
    
    def in_past(self, field: str, amount: int, unit: str, inclusive: bool = True):
        """
        Add a condition for dates in the past.
        
        Args:
            field: Date field name
            amount: Number of time units
            unit: Time unit ("days", "weeks", "months", "years")
            inclusive: Whether to include the current moment
            
        Returns:
            self: For method chaining
        """
        now = datetime.datetime.now()
        past_date = self._get_relative_date(now, -amount, unit)
        
        if inclusive:
            self._conditions.append({
                "type": "gte",
                "field": field,
                "value": past_date
            })
        else:
            self._conditions.append({
                "type": "gt",
                "field": field,
                "value": past_date
            })
        
        return self
    
    def in_future(self, field: str, amount: int, unit: str, inclusive: bool = True):
        """
        Add a condition for dates in the future.
        
        Args:
            field: Date field name
            amount: Number of time units
            unit: Time unit ("days", "weeks", "months", "years")
            inclusive: Whether to include the current moment
            
        Returns:
            self: For method chaining
        """
        now = datetime.datetime.now()
        future_date = self._get_relative_date(now, amount, unit)
        
        if inclusive:
            self._conditions.append({
                "type": "lte",
                "field": field,
                "value": future_date
            })
        else:
            self._conditions.append({
                "type": "lt",
                "field": field,
                "value": future_date
            })
        
        return self
    
    def in_last(self, field: str, amount: int, unit: str):
        """
        Add a condition for dates in the last N time units.
        Alias for in_past with inclusive=True.
        
        Args:
            field: Date field name
            amount: Number of time units
            unit: Time unit ("days", "weeks", "months", "years")
            
        Returns:
            self: For method chaining
        """
        return self.in_past(field, amount, unit, inclusive=True)
    
    def in_next(self, field: str, amount: int, unit: str):
        """
        Add a condition for dates in the next N time units.
        Alias for in_future with inclusive=True.
        
        Args:
            field: Date field name
            amount: Number of time units
            unit: Time unit ("days", "weeks", "months", "years")
            
        Returns:
            self: For method chaining
        """
        return self.in_future(field, amount, unit, inclusive=True)
    
    def today(self, field: str):
        """
        Add a condition for dates that match today.
        
        Args:
            field: Date field name
            
        Returns:
            self: For method chaining
        """
        today = datetime.date.today()
        return self.on_date(field, today)
    
    def yesterday(self, field: str):
        """
        Add a condition for dates that match yesterday.
        
        Args:
            field: Date field name
            
        Returns:
            self: For method chaining
        """
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        return self.on_date(field, yesterday)
    
    def tomorrow(self, field: str):
        """
        Add a condition for dates that match tomorrow.
        
        Args:
            field: Date field name
            
        Returns:
            self: For method chaining
        """
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        return self.on_date(field, tomorrow)
    
    def this_week(self, field: str):
        """
        Add a condition for dates in the current week.
        
        Args:
            field: Date field name
            
        Returns:
            self: For method chaining
        """
        today = datetime.date.today()
        start_of_week = today - datetime.timedelta(days=today.weekday())
        end_of_week = start_of_week + datetime.timedelta(days=6)
        
        return self.between(field, start_of_week, end_of_week)
    
    def this_month(self, field: str):
        """
        Add a condition for dates in the current month.
        
        Args:
            field: Date field name
            
        Returns:
            self: For method chaining
        """
        today = datetime.date.today()
        start_of_month = today.replace(day=1)
        
        # Calculate end of month
        if today.month == 12:
            end_of_month = today.replace(year=today.year+1, month=1, day=1) - datetime.timedelta(days=1)
        else:
            end_of_month = today.replace(month=today.month+1, day=1) - datetime.timedelta(days=1)
        
        return self.between(field, start_of_month, end_of_month)
    
    def this_year(self, field: str):
        """
        Add a condition for dates in the current year.
        
        Args:
            field: Date field name
            
        Returns:
            self: For method chaining
        """
        today = datetime.date.today()
        start_of_year = today.replace(month=1, day=1)
        end_of_year = today.replace(month=12, day=31)
        
        return self.between(field, start_of_year, end_of_year)
    
    def exists(self, field: str):
        """
        Add an exists condition to the filter.
        
        Args:
            field: Date field name
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "exists", "field": field})
        return self
    
    def not_exists(self, field: str):
        """
        Add a not exists condition to the filter.
        
        Args:
            field: Date field name
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "not_exists", "field": field})
        return self
    
    def and_operator(self, *filter_builders):
        """
        Combine multiple filter builders with AND logic.
        
        Args:
            *filter_builders: Filter builders to combine
            
        Returns:
            self: For method chaining
        """
        combined_conditions = []
        for builder in filter_builders:
            if isinstance(builder, DateFilterBuilder):
                combined_conditions.extend(builder._conditions)
        
        if combined_conditions:
            self._conditions.append({
                "type": "and",
                "conditions": combined_conditions
            })
        
        return self
    
    def or_operator(self, *filter_builders):
        """
        Combine multiple filter builders with OR logic.
        
        Args:
            *filter_builders: Filter builders to combine
            
        Returns:
            self: For method chaining
        """
        combined_conditions = []
        for builder in filter_builders:
            if isinstance(builder, DateFilterBuilder):
                combined_conditions.extend(builder._conditions)
        
        if combined_conditions:
            self._conditions.append({
                "type": "or",
                "conditions": combined_conditions
            })
        
        return self
    
    def not_operator(self, filter_builder):
        """
        Negate a filter builder.
        
        Args:
            filter_builder: Filter builder to negate
            
        Returns:
            self: For method chaining
        """
        if isinstance(filter_builder, DateFilterBuilder) and filter_builder._conditions:
            self._conditions.append({
                "type": "not",
                "condition": filter_builder._conditions
            })
        
        return self
    
    def reset(self):
        """Reset the filter to empty state."""
        self._conditions = []
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build and return the filter in the specified format.
        
        Returns:
            Dict[str, Any]: Filter in the specified format
        """
        if not self._conditions:
            return {}
        
        # First build a generic filter structure
        generic_filter = {
            "type": "date_filter",
            "conditions": copy.deepcopy(self._conditions)
        }
        
        # Convert dates to strings in ISO format for JSON serialization
        generic_filter = self._format_date_values(generic_filter)
        
        # Format the filter for the target database
        return self._format_for_target(generic_filter)
    
    def _format_date_values(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format date values to strings.
        
        Args:
            filter_dict: Filter dictionary
            
        Returns:
            Dict[str, Any]: Filter with formatted date values
        """
        if "conditions" in filter_dict:
            for i, condition in enumerate(filter_dict["conditions"]):
                filter_dict["conditions"][i] = self._format_date_values(condition)
        
        if "condition" in filter_dict:
            filter_dict["condition"] = self._format_date_values(filter_dict["condition"])
        
        if "value" in filter_dict and isinstance(filter_dict["value"], (datetime.datetime, datetime.date)):
            filter_dict["value"] = self._format_date(filter_dict["value"])
        
        if "values" in filter_dict:
            filter_dict["values"] = [
                self._format_date(value) if isinstance(value, (datetime.datetime, datetime.date)) else value
                for value in filter_dict["values"]
            ]
        
        if "start_date" in filter_dict and isinstance(filter_dict["start_date"], (datetime.datetime, datetime.date)):
            filter_dict["start_date"] = self._format_date(filter_dict["start_date"])
        
        if "end_date" in filter_dict and isinstance(filter_dict["end_date"], (datetime.datetime, datetime.date)):
            filter_dict["end_date"] = self._format_date(filter_dict["end_date"])
        
        return filter_dict
    
    def _ensure_date(self, date_value: Union[str, datetime.datetime, datetime.date]) -> Union[datetime.datetime, datetime.date]:
        """
        Ensure a value is a date or datetime object.
        
        Args:
            date_value: Date value to convert
            
        Returns:
            Union[datetime.datetime, datetime.date]: Date or datetime object
        """
        if isinstance(date_value, (datetime.datetime, datetime.date)):
            return date_value
        
        try:
            return parse_date(date_value)
        except Exception as e:
            logger.error(f"Could not parse date '{date_value}': {e}")
            raise ValueError(f"Invalid date format: {date_value}")
    
    def _format_date(self, date_value: Union[datetime.datetime, datetime.date]) -> str:
        """
        Format a date or datetime object as an ISO string.
        
        Args:
            date_value: Date value to format
            
        Returns:
            str: Formatted date string
        """
        if isinstance(date_value, datetime.datetime):
            return date_value.isoformat()
        else:  # date
            return date_value.isoformat()
    
    def _get_relative_date(self, base_date: datetime.datetime, amount: int, unit: str) -> datetime.datetime:
        """
        Get a date relative to the base date.
        
        Args:
            base_date: Base date
            amount: Number of time units (positive for future, negative for past)
            unit: Time unit ("days", "weeks", "months", "years")
            
        Returns:
            datetime.datetime: Relative date
        """
        unit = unit.lower().strip()
        
        if unit == "days" or unit == "day":
            return base_date + datetime.timedelta(days=amount)
        elif unit == "weeks" or unit == "week":
            return base_date + datetime.timedelta(weeks=amount)
        elif unit == "months" or unit == "month":
            return base_date + relativedelta(months=amount)
        elif unit == "years" or unit == "year":
            return base_date + relativedelta(years=amount)
        else:
            raise ValueError(f"Unsupported time unit: {unit}. Use 'days', 'weeks', 'months', or 'years'.")
    
    def _format_for_target(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a generic filter for the specified target database.
        
        Args:
            generic_filter: Generic filter representation
            
        Returns:
            Dict[str, Any]: Filter formatted for the target database
        """
        if self._target_format == "generic":
            return generic_filter
        
        # Format for specific target databases
        if self._target_format == "chroma":
            return self._format_for_chroma(generic_filter)
        elif self._target_format == "qdrant":
            return self._format_for_qdrant(generic_filter)
        elif self._target_format == "pinecone":
            return self._format_for_pinecone(generic_filter)
        elif self._target_format == "weaviate":
            return self._format_for_weaviate(generic_filter)
        elif self._target_format == "milvus":
            return self._format_for_milvus(generic_filter)
        elif self._target_format == "pgvector":
            return self._format_for_pgvector(generic_filter)
        else:
            logger.warning(f"No specific formatting implemented for {self._target_format}. Using generic format.")
            return generic_filter
    
    def _format_for_chroma(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Chroma DB."""
        # Implementation will depend on Chroma's specific filter syntax
        return generic_filter
    
    def _format_for_qdrant(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Qdrant."""
        # Implementation will depend on Qdrant's specific filter syntax
        return generic_filter
    
    def _format_for_pinecone(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Pinecone."""
        # Implementation will depend on Pinecone's specific filter syntax
        return generic_filter
    
    def _format_for_weaviate(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Weaviate."""
        # Implementation will depend on Weaviate's specific filter syntax
        return generic_filter
    
    def _format_for_milvus(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Milvus."""
        # Implementation will depend on Milvus's specific filter syntax
        return generic_filter
    
    def _format_for_pgvector(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for pgvector."""
        # Implementation will depend on pgvector's specific filter syntax
        return generic_filter 