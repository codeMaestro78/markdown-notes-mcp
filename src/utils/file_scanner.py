import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Handle frontmatter import
try:
    import frontmatter
except ImportError:
    print("Warning: python-frontmatter not installed. Install with: pip install python-frontmatter")
    frontmatter = None


class FileScanner:
    def __init__(self):
        self.supported_extensions = {'.md', '.markdown', '.txt'}
    
    def scan_directory(self, directory: str) -> List[str]:
        """Scan directory for supported file types"""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if Path(filename).suffix.lower() in self.supported_extensions:
                    files.append(os.path.join(root, filename))
        return files
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a file"""
        try:
            stat = os.stat(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse frontmatter if exists and available
            frontmatter_data = {}
            content_without_frontmatter = content
            
            if frontmatter:
                try:
                    post = frontmatter.loads(content)
                    frontmatter_data = post.metadata
                    content_without_frontmatter = post.content
                except:
                    pass
            
            # Extract title
            title = self._extract_title(content_without_frontmatter, file_path, frontmatter_data)
            
            # Extract tags
            tags = self._extract_tags(content_without_frontmatter, frontmatter_data)
            
            # Convert timestamps to proper ISO format
            created_date = datetime.fromtimestamp(stat.st_ctime).isoformat()
            modified_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            return {
                'path': file_path,
                'title': title,
                'content': content_without_frontmatter,
                'tags': tags,
                'size': stat.st_size,
                'created_date': created_date,
                'modified_date': modified_date,
                'frontmatter': frontmatter_data,
                'word_count': len(content_without_frontmatter.split()),
                'line_count': len(content_without_frontmatter.splitlines())
            }
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {e}")
            return {
                'path': file_path,
                'title': Path(file_path).stem,
                'content': '',
                'tags': [],
                'size': 0,
                'created_date': datetime.now().isoformat(),
                'modified_date': datetime.now().isoformat(),
                'frontmatter': {},
                'word_count': 0,
                'line_count': 0,
                'error': str(e)
            }
    
    def _extract_title(self, content: str, file_path: str, frontmatter_data: Dict) -> str:
        """Extract title from content or frontmatter"""
        # Check frontmatter first
        if 'title' in frontmatter_data:
            return frontmatter_data['title']
        
        # Look for first H1 header
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fallback to filename
        return Path(file_path).stem
    
    def _extract_tags(self, content: str, frontmatter_data: Dict) -> List[str]:
        """Extract tags from content and frontmatter"""
        tags = set()
        
        # From frontmatter
        if 'tags' in frontmatter_data:
            fm_tags = frontmatter_data['tags']
            if isinstance(fm_tags, list):
                tags.update(fm_tags)
            elif isinstance(fm_tags, str):
                tags.update([tag.strip() for tag in fm_tags.split(',')])
        
        # From content (hashtags)
        hashtag_pattern = r'#(\w+)'
        hashtags = re.findall(hashtag_pattern, content)
        tags.update(hashtags)
        
        # From content (tag: format)
        tag_pattern = r'(?:^|\s)(?:tags?|Tags?|TAGS?):\s*([^\n]+)'
        tag_matches = re.findall(tag_pattern, content, re.MULTILINE)
        for match in tag_matches:
            tag_list = [tag.strip() for tag in match.split(',')]
            tags.update(tag_list)
        
        return list(tags)
