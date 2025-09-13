import re
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union
from pathlib import Path

# Handle imports for both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class CriteriaEvaluator:
    def __init__(self):
        self.date_operators = {
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b
        }
        
        self.numeric_operators = {
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b
        }
    
    def evaluate(self, note_metadata: Dict, criteria: List[Dict], logical_operator: str = "AND") -> bool:
        """Evaluate if note metadata matches the given criteria"""
        if not criteria:
            return True
        
        results = []
        for criterion in criteria:
            result = self._evaluate_single_criterion(note_metadata, criterion)
            results.append(result)
        
        if logical_operator.upper() == "OR":
            return any(results)
        else:  # AND
            return all(results)
    
    def _evaluate_single_criterion(self, note_metadata: Dict, criterion: Dict) -> bool:
        """Evaluate a single criterion"""
        criterion_type = criterion.get('type')
        operator = criterion.get('operator', '==')
        value = criterion.get('value')
        
        if criterion_type == 'tag':
            return self._evaluate_tag_criterion(note_metadata, operator, value)
        elif criterion_type == 'date':
            return self._evaluate_date_criterion(note_metadata, criterion)
        elif criterion_type == 'keyword':
            return self._evaluate_keyword_criterion(note_metadata, criterion)
        elif criterion_type == 'content_length':
            return self._evaluate_content_length_criterion(note_metadata, operator, value)
        elif criterion_type == 'file_type':
            return self._evaluate_file_type_criterion(note_metadata, operator, value)
        elif criterion_type == 'path':
            return self._evaluate_path_criterion(note_metadata, operator, value)
        
        return False
    
    def _evaluate_tag_criterion(self, note_metadata: Dict, operator: str, value: Union[str, List[str]]) -> bool:
        """Evaluate tag-based criteria"""
        note_tags = note_metadata.get('tags', [])
        
        if isinstance(value, str):
            value = [value]
        
        if operator == 'contains_any':
            return any(tag in note_tags for tag in value)
        elif operator == 'contains_all':
            return all(tag in note_tags for tag in value)
        elif operator == 'equals':
            return set(note_tags) == set(value)
        elif operator == 'not_contains':
            return not any(tag in note_tags for tag in value)
        
        return False
    
    def _parse_date_value(self, value: str) -> datetime:
        """Parse date value, supporting relative dates like 'now-7d'"""
        if value.startswith('now'):
            now = datetime.now()
            if value == 'now':
                return now
            
            # Parse relative date like 'now-7d', 'now+1w'
            match = re.match(r'now([+-])(\d+)([dwmyh])', value)
            if match:
                sign, amount, unit = match.groups()
                amount = int(amount)
                if sign == '-':
                    amount = -amount
                
                if unit == 'd':
                    return now + timedelta(days=amount)
                elif unit == 'w':
                    return now + timedelta(weeks=amount)
                elif unit == 'h':
                    return now + timedelta(hours=amount)
                elif unit == 'm':
                    return now + timedelta(days=amount * 30)  # Approximate month
                elif unit == 'y':
                    return now + timedelta(days=amount * 365)  # Approximate year
        
        # Try to parse as ISO date
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            # Fallback: try common date formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d']:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            # If all fails, return a very old date so comparison fails gracefully
            return datetime(1900, 1, 1)

    def _evaluate_date_criterion(self, note_metadata: Dict, criterion: Dict) -> bool:
        """Evaluate date-based criteria"""
        field = criterion.get('field', 'modified_date')
        operator = criterion.get('operator', '>=' )
        value = criterion.get('value')
        
        note_date_str = note_metadata.get(field)
        if not note_date_str:
            return False
        
        try:
            # Handle different date formats from file system
            if isinstance(note_date_str, str):
                # Try parsing ISO format first
                try:
                    note_date = datetime.fromisoformat(note_date_str.replace('Z', '+00:00'))
                except ValueError:
                    # Fallback to timestamp if it's a number string
                    try:
                        note_date = datetime.fromtimestamp(float(note_date_str))
                    except (ValueError, TypeError):
                        return False
            else:
                return False
                
            target_date = self._parse_date_value(value)
            
            if operator in self.date_operators:
                return self.date_operators[operator](note_date, target_date)
        except (ValueError, TypeError) as e:
            print(f"Date evaluation error: {e}, note_date: {note_date_str}, target: {value}")
            return False
        
        return False
    
    def _evaluate_keyword_criterion(self, note_metadata: Dict, criterion: Dict) -> bool:
        """Evaluate keyword-based criteria"""
        operator = criterion.get('operator', 'contains_any')
        value = criterion.get('value', [])
        case_sensitive = criterion.get('case_sensitive', False)
        
        if isinstance(value, str):
            value = [value]
        
        # Get content to search
        content = note_metadata.get('content', '')
        title = note_metadata.get('title', '')
        search_text = f"{title} {content}"
        
        if not case_sensitive:
            search_text = search_text.lower()
            value = [v.lower() for v in value]
        
        if operator == 'contains_any':
            return any(keyword in search_text for keyword in value)
        elif operator == 'contains_all':
            return all(keyword in search_text for keyword in value)
        elif operator == 'not_contains':
            return not any(keyword in search_text for keyword in value)
        elif operator == 'regex':
            pattern = value[0] if value else ''
            flags = 0 if case_sensitive else re.IGNORECASE
            return bool(re.search(pattern, search_text, flags))
        
        return False
    
    def _evaluate_content_length_criterion(self, note_metadata: Dict, operator: str, value: int) -> bool:
        """Evaluate content length criteria"""
        content = note_metadata.get('content', '')
        word_count = len(content.split())
        
        if operator in self.numeric_operators:
            return self.numeric_operators[operator](word_count, value)
        
        return False
    
    def _evaluate_file_type_criterion(self, note_metadata: Dict, operator: str, value: Union[str, List[str]]) -> bool:
        """Evaluate file type criteria"""
        file_path = note_metadata.get('path', '')
        file_extension = Path(file_path).suffix.lower()
        
        if isinstance(value, str):
            value = [value]
        
        value = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in value]
        
        if operator == 'equals':
            return file_extension in value
        elif operator == 'not_equals':
            return file_extension not in value
        
        return False
    
    def _evaluate_path_criterion(self, note_metadata: Dict, operator: str, value: str) -> bool:
        """Evaluate file path criteria"""
        file_path = note_metadata.get('path', '')
        
        if operator == 'contains':
            return value in file_path
        elif operator == 'starts_with':
            return file_path.startswith(value)
        elif operator == 'ends_with':
            return file_path.endswith(value)
        elif operator == 'regex':
            return bool(re.search(value, file_path))
        
        return False
