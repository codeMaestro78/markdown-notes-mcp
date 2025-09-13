import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from smart_collections.manager import SmartCollectionsManager
from mcp.collections_handler import CollectionsHandler


def main():
    # Setup paths
    config_path = project_root / "config" / "smart_collections.json"
    notes_directory = project_root / "notes"  # Change this to your notes directory
    
    # Create notes directory if it doesn't exist
    notes_directory.mkdir(exist_ok=True)
    
    # Initialize Smart Collections Manager
    print("Initializing Smart Collections Manager...")
    manager = SmartCollectionsManager(str(config_path), str(notes_directory))
    
    # Initialize MCP Handler
    handler = CollectionsHandler(manager)
    
    # Start auto-update
    manager.start_auto_update()
    print("Smart Collections system started with auto-update enabled")
    
    # Demo the functionality
    demo_smart_collections(handler, manager)
    
    try:
        print("\nPress Ctrl+C to stop...")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Smart Collections system...")
        manager.stop_auto_update()
        print("System stopped.")


def demo_smart_collections(handler, manager):
    """Demo the Smart Collections functionality"""
    print("\n=== Smart Collections Demo ===")
    
    # Debug: Check what notes are found
    print(f"\nScanning notes in: {manager.notes_directory}")
    all_notes = manager.file_scanner.scan_directory(manager.notes_directory)
    print(f"Found {len(all_notes)} notes total:")
    for note in all_notes[:5]:  # Show first 5
        metadata = manager.file_scanner.extract_metadata(note)
        print(f"  - {metadata['title']} (modified: {metadata['modified_date']}, tags: {metadata['tags']})")
    if len(all_notes) > 5:
        print(f"  ... and {len(all_notes) - 5} more")
    
    # List existing collections
    result = handler.handle_list_collections()
    print(f"\nExisting collections: {len(result['collections'])}")
    for collection in result['collections']:
        print(f"  - {collection['icon']} {collection['name']}: {collection['note_count']} notes")
    
    # Debug: Check why template collections have 0 notes
    print("\n=== Debugging Template Collections ===")
    for collection_id in ['recent_notes', 'tagged_important', 'meeting_notes']:
        collection = manager.get_collection(collection_id)
        if collection:
            print(f"\nChecking '{collection['name']}' criteria:")
            print(f"  Criteria: {collection['criteria']}")
            
            # Test criteria against first note
            if all_notes:
                test_note = all_notes[0]
                metadata = manager.file_scanner.extract_metadata(test_note)
                result = manager.criteria_evaluator.evaluate(
                    metadata, collection['criteria'], collection.get('logical_operator', 'AND')
                )
                print(f"  Test note '{metadata['title']}' matches: {result}")
                print(f"  Note modified_date: {metadata['modified_date']}")
                print(f"  Note tags: {metadata['tags']}")
    
    # Create a custom collection
    print("\nCreating a custom collection for Python-related notes...")
    criteria = [
        {
            "type": "keyword",
            "operator": "contains_any",
            "value": ["python", "programming", "code"],
            "case_sensitive": False
        }
    ]
    
    create_result = handler.handle_create_collection(
        name="Python Notes",
        description="Notes related to Python programming",
        criteria=criteria,
        icon="🐍"
    )
    
    if create_result['success']:
        print(f"✅ {create_result['message']}")
        
        # Get the created collection
        collection_id = create_result['collection_id']
        collection_result = handler.handle_get_collection(collection_id)
        if collection_result['success']:
            collection = collection_result['collection']
            print(f"   Found {collection['note_count']} Python-related notes")
    else:
        print(f"❌ Error: {create_result['error']}")
    
    # Update all collections
    print("\nUpdating all collections...")
    update_result = handler.handle_update_all_collections()
    if update_result['success']:
        print(f"✅ {update_result['message']}")
        
        # Show updated counts
        result = handler.handle_list_collections()
        print(f"\nUpdated collections:")
        for collection in result['collections']:
            print(f"  - {collection['icon']} {collection['name']}: {collection['note_count']} notes")
    else:
        print(f"❌ Error: {update_result['error']}")


if __name__ == "__main__":
    main()
