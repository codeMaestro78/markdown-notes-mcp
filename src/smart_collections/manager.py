import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
import time

# Handle imports for both direct execution and module import
try:
    from smart_collections.criteria_evaluator import CriteriaEvaluator
    from utils.file_scanner import FileScanner
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from smart_collections.criteria_evaluator import CriteriaEvaluator
    from utils.file_scanner import FileScanner


class SmartCollectionsManager:
    def __init__(self, config_path: str, notes_directory: str):
        self.config_path = config_path
        self.notes_directory = notes_directory
        self.collections: Dict[str, Dict] = {}
        self.criteria_evaluator = CriteriaEvaluator()
        self.file_scanner = FileScanner()
        self._cache = {}
        self._lock = threading.RLock()
        self._load_config()
        self._auto_update_thread = None
        self._stop_auto_update = False

    def _load_config(self):
        """Load smart collections configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.settings = self.config.get('smart_collections', {}).get('settings', {})
            self._load_template_collections()
        except FileNotFoundError:
            self.config = {"smart_collections": {"templates": {}, "settings": {}}}
            self.settings = {}

    def _load_template_collections(self):
        """Load predefined template collections"""
        templates = self.config.get('smart_collections', {}).get('templates', {})
        for template_id, template_config in templates.items():
            self.collections[template_id] = {
                **template_config,
                'id': template_id,
                'type': 'template',
                'created_at': datetime.now().isoformat(),
                'last_updated': None,
                'note_count': 0,
                'notes': []
            }

    def create_collection(self, name: str, description: str, criteria: List[Dict],
                         logical_operator: str = "AND", auto_update: bool = True,
                         icon: str = "📁") -> str:
        """Create a new smart collection"""
        collection_id = f"custom_{len([c for c in self.collections.values() if c.get('type') == 'custom'])}"

        collection = {
            'id': collection_id,
            'name': name,
            'description': description,
            'criteria': criteria,
            'logical_operator': logical_operator,
            'auto_update': auto_update,
            'icon': icon,
            'type': 'custom',
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'note_count': 0,
            'notes': []
        }

        self.collections[collection_id] = collection
        self.update_collection(collection_id)
        return collection_id

    def update_collection(self, collection_id: str) -> bool:
        """Update a specific collection by re-evaluating its criteria"""
        if collection_id not in self.collections:
            return False

        collection = self.collections[collection_id]

        # Check cache first
        cache_key = f"{collection_id}_{hash(str(collection['criteria']))}"
        cache_ttl = self.settings.get('cache_ttl', 600)

        if (self.settings.get('cache_results', True) and
            cache_key in self._cache and
            time.time() - self._cache[cache_key]['timestamp'] < cache_ttl):
            collection['notes'] = self._cache[cache_key]['notes']
            collection['note_count'] = len(collection['notes'])
            return True

        # Scan for notes
        all_notes = self.file_scanner.scan_directory(self.notes_directory)
        matching_notes = []

        for note_path in all_notes:
            note_metadata = self.file_scanner.extract_metadata(note_path)
            if self.criteria_evaluator.evaluate(note_metadata, collection['criteria'],
                                               collection.get('logical_operator', 'AND')):
                matching_notes.append({
                    'path': note_path,
                    'title': note_metadata.get('title', Path(note_path).stem),
                    'modified_date': note_metadata.get('modified_date'),
                    'tags': note_metadata.get('tags', []),
                    'size': note_metadata.get('size', 0)
                })

        # Sort by modification date (newest first)
        matching_notes.sort(key=lambda x: x.get('modified_date', ''), reverse=True)

        collection['notes'] = matching_notes
        collection['note_count'] = len(matching_notes)
        collection['last_updated'] = datetime.now().isoformat()

        # Cache results
        if self.settings.get('cache_results', True):
            self._cache[cache_key] = {
                'notes': matching_notes,
                'timestamp': time.time()
            }

        return True

    def update_all_collections(self):
        """Update all collections that have auto_update enabled"""
        # Create a snapshot of collections to avoid dictionary changed size during iteration
        collections_snapshot = dict(self.collections)
        for collection_id, collection in collections_snapshot.items():
            if collection.get('auto_update', True):
                self.update_collection(collection_id)

    def get_collection(self, collection_id: str) -> Optional[Dict]:
        """Get a specific collection"""
        return self.collections.get(collection_id)

    def list_collections(self) -> List[Dict]:
        """List all collections with summary info"""
        # Create a snapshot to avoid thread safety issues
        collections_snapshot = dict(self.collections)
        return [
            {
                'id': coll['id'],
                'name': coll['name'],
                'description': coll['description'],
                'icon': coll.get('icon', '📁'),
                'note_count': coll.get('note_count', 0),
                'last_updated': coll.get('last_updated'),
                'auto_update': coll.get('auto_update', True),
                'type': coll.get('type', 'custom')
            }
            for coll in collections_snapshot.values()
        ]

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a custom collection (template collections cannot be deleted)"""
        if (collection_id in self.collections and
            self.collections[collection_id].get('type') == 'custom'):
            del self.collections[collection_id]
            return True
        return False

    def start_auto_update(self):
        """Start automatic collection updates"""
        if self._auto_update_thread is None or not self._auto_update_thread.is_alive():
            self._stop_auto_update = False
            self._auto_update_thread = threading.Thread(target=self._auto_update_worker)
            self._auto_update_thread.daemon = True
            self._auto_update_thread.start()

    def stop_auto_update(self):
        """Stop automatic collection updates"""
        self._stop_auto_update = True
        if self._auto_update_thread:
            self._auto_update_thread.join(timeout=5)

    def _auto_update_worker(self):
        """Background worker for automatic updates"""
        interval = self.settings.get('auto_update_interval', 300)  # 5 minutes default

        while not self._stop_auto_update:
            try:
                self.update_all_collections()
                time.sleep(interval)
            except Exception as e:
                print(f"Error in auto-update worker: {e}")
                time.sleep(interval)


def main():
    """Main function for testing the manager directly"""
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "smart_collections.json"
    notes_directory = project_root / "notes"

    print(f"Config path: {config_path}")
    print(f"Notes directory: {notes_directory}")

    # Create manager
    manager = SmartCollectionsManager(str(config_path), str(notes_directory))

    # Test functionality
    print(f"Loaded {len(manager.collections)} collections")
    for collection_id, collection in manager.collections.items():
        print(f"  - {collection['name']}: {collection.get('note_count', 0)} notes")

    # Update all collections
    print("\nUpdating collections...")
    manager.update_all_collections()

    print("\nAfter update:")
    for collection_id, collection in manager.collections.items():
        print(f"  - {collection['name']}: {collection.get('note_count', 0)} notes")


if __name__ == "__main__":
    main()
