from typing import List, Dict, Any, Optional
import json
from smart_collections.manager import SmartCollectionsManager


class CollectionsHandler:
    def __init__(self, smart_collections_manager: SmartCollectionsManager):
        self.manager = smart_collections_manager
    
    def handle_list_collections(self) -> Dict[str, Any]:
        """Handle request to list all smart collections"""
        try:
            collections = self.manager.list_collections()
            return {
                "success": True,
                "collections": collections,
                "total_count": len(collections)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_get_collection(self, collection_id: str) -> Dict[str, Any]:
        """Handle request to get a specific collection with its notes"""
        try:
            collection = self.manager.get_collection(collection_id)
            if not collection:
                return {
                    "success": False,
                    "error": f"Collection '{collection_id}' not found"
                }
            
            return {
                "success": True,
                "collection": collection
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_create_collection(self, name: str, description: str, criteria: List[Dict], 
                               logical_operator: str = "AND", auto_update: bool = True, 
                               icon: str = "📁") -> Dict[str, Any]:
        """Handle request to create a new smart collection"""
        try:
            collection_id = self.manager.create_collection(
                name, description, criteria, logical_operator, auto_update, icon
            )
            return {
                "success": True,
                "collection_id": collection_id,
                "message": f"Collection '{name}' created successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_update_collection(self, collection_id: str) -> Dict[str, Any]:
        """Handle request to update a specific collection"""
        try:
            success = self.manager.update_collection(collection_id)
            if success:
                collection = self.manager.get_collection(collection_id)
                return {
                    "success": True,
                    "message": f"Collection updated successfully",
                    "note_count": collection.get('note_count', 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"Collection '{collection_id}' not found"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_delete_collection(self, collection_id: str) -> Dict[str, Any]:
        """Handle request to delete a collection"""
        try:
            success = self.manager.delete_collection(collection_id)
            if success:
                return {
                    "success": True,
                    "message": f"Collection '{collection_id}' deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "error": f"Collection '{collection_id}' not found or cannot be deleted"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_update_all_collections(self) -> Dict[str, Any]:
        """Handle request to update all collections"""
        try:
            self.manager.update_all_collections()
            collections = self.manager.list_collections()
            return {
                "success": True,
                "message": "All collections updated successfully",
                "updated_count": len([c for c in collections if c.get('auto_update', True)])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
