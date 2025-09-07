---
title: "Advanced Configuration - Practical Examples & Real-World Scenarios"
tags: [examples, practical, real-world, scenarios, implementation, case-studies]
created: 2025-09-07
updated: 2025-09-07
model_config: "high_quality"
chunking_strategy: "hybrid"
search_priority: "technical"
---

# Advanced Configuration - Practical Examples & Real-World Scenarios

This document provides **hands-on examples** and **real-world scenarios** demonstrating how to leverage the advanced configuration system for various use cases and deployment scenarios.

## 🎯 **Quick Start Examples**

### **Example 1: Basic Setup with Environment Detection**

```python
#!/usr/bin/env python3
"""
Basic setup example demonstrating automatic environment detection
and configuration loading.
"""

from config import AdvancedConfig
import os

def main():
    # Initialize configuration (auto-detects environment)
    config = AdvancedConfig()

    print("🚀 Advanced Configuration Demo")
    print(f"Environment: {config.environment.value}")
    print(f"Model: {config.model_name}")
    print(f"Chunk Size: {config.chunk_size}")
    print(f"GPU Enabled: {config.performance.enable_gpu}")
    print(f"Batch Size: {config.batch_size}")

    # Validate configuration
    is_valid, errors = config.validate()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return

    print("✅ Configuration is valid!")

    # Save current configuration
    config.save_config("config/current_demo.json")
    print("💾 Configuration saved to config/current_demo.json")

if __name__ == "__main__":
    main()
```

### **Example 2: Dynamic Model Switching**

```python
#!/usr/bin/env python3
"""
Dynamic model switching example showing how to change models
at runtime for different use cases.
"""

from config import AdvancedConfig
from notes_mcp_server import NotesMCPServer
import time

def benchmark_model(config, model_name, test_queries):
    """Benchmark a specific model with test queries."""
    print(f"\n🔄 Switching to model: {model_name}")

    # Switch model
    config.set_model(model_name)

    # Initialize server with new configuration
    server = NotesMCPServer(config)

    # Benchmark queries
    total_time = 0
    for query in test_queries:
        start_time = time.time()
        results = server.search_notes(query, top_k=3)
        end_time = time.time()

        query_time = end_time - start_time
        total_time += query_time
        print(".2f")

    avg_time = total_time / len(test_queries)
    print(".2f"
    return avg_time

def main():
    config = AdvancedConfig()

    # Test queries for different scenarios
    test_queries = [
        "machine learning algorithms",
        "neural network optimization",
        "data preprocessing techniques",
        "model evaluation metrics",
        "deep learning architectures"
    ]

    models_to_test = [
        "all-MiniLM-L6-v2",      # Fast model
        "all-mpnet-base-v2"      # High-quality model
    ]

    results = {}

    print("🧪 Model Performance Benchmark")
    print("=" * 50)

    for model in models_to_test:
        try:
            avg_time = benchmark_model(config, model, test_queries)
            results[model] = avg_time
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")

    # Print summary
    print("\n📊 Benchmark Results Summary")
    print("=" * 50)
    for model, avg_time in results.items():
        print(".2f")

    # Recommend best model
    if results:
        best_model = min(results, key=results.get)
        print(f"\n🎯 Recommended model: {best_model} (fastest)")

if __name__ == "__main__":
    main()
```

## 🏢 **Real-World Scenarios**

### **Scenario 1: Enterprise Knowledge Base**

#### **Company Setup**
```python
# config/enterprise.json
{
  "model_name": "all-mpnet-base-v2",
  "chunk_size": 200,
  "batch_size": 128,
  "max_file_size": 104857600,  // 100MB
  "request_timeout": 60,
  "host": "0.0.0.0",
  "port": 8181,
  "admin_token": "enterprise_token_secure_123",
  "performance": {
    "enable_gpu": true,
    "max_workers": 16,
    "memory_limit_mb": 16384,
    "gpu_memory_fraction": 0.9
  },
  "security": {
    "enable_auth": true,
    "ssl_enabled": true,
    "rate_limit": "5000/hour",
    "allowed_ips": ["10.0.0.0/8", "172.16.0.0/12"]
  },
  "monitoring": {
    "enable_metrics": true,
    "log_level": "INFO",
    "metrics_port": 9090
  }
}
```

#### **Usage Example**
```python
#!/usr/bin/env python3
"""
Enterprise knowledge base implementation with advanced features.
"""

from config import AdvancedConfig
from notes_mcp_server import NotesMCPServer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnterpriseKnowledgeBase:
    def __init__(self, config_path="config/enterprise.json"):
        # Load enterprise configuration
        self.config = AdvancedConfig()
        self.config.load_config(config_path)

        # Initialize server
        self.server = NotesMCPServer(self.config)

        # Enterprise-specific settings
        self.departments = {
            "engineering": "tech_docs/",
            "sales": "sales_docs/",
            "hr": "hr_docs/",
            "finance": "finance_docs/"
        }

        logger.info("🏢 Enterprise Knowledge Base initialized")

    def search_by_department(self, query, department, top_k=10):
        """Search within specific department."""
        if department not in self.departments:
            raise ValueError(f"Unknown department: {department}")

        # Filter search to department
        dept_path = self.departments[department]
        results = self.server.search_notes(
            query,
            top_k=top_k,
            filter_path=dept_path
        )

        logger.info(f"Department search: {department} - {query}")
        return results

    def cross_department_search(self, query, departments=None, top_k=5):
        """Search across multiple departments."""
        if departments is None:
            departments = list(self.departments.keys())

        all_results = []
        for dept in departments:
            try:
                results = self.search_by_department(query, dept, top_k=top_k)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching {dept}: {e}")

        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:top_k]

    def get_department_stats(self):
        """Get statistics for each department."""
        stats = {}
        for dept, path in self.departments.items():
            try:
                # Get document count and other metrics
                doc_count = self.server.get_document_count(path)
                stats[dept] = {
                    "document_count": doc_count,
                    "path": path,
                    "last_updated": self.server.get_last_updated(path)
                }
            except Exception as e:
                logger.error(f"Error getting stats for {dept}: {e}")
                stats[dept] = {"error": str(e)}

        return stats

def main():
    # Initialize enterprise knowledge base
    kb = EnterpriseKnowledgeBase()

    # Example searches
    print("🏢 Enterprise Knowledge Base Demo")
    print("=" * 50)

    # Department-specific search
    print("\n🔍 Engineering Search:")
    eng_results = kb.search_by_department(
        "machine learning deployment", "engineering", top_k=3
    )
    for result in eng_results:
        print(f"  - {result.get('title', 'Unknown')}: {result.get('score', 0):.3f}")

    # Cross-department search
    print("\n🔍 Cross-Department Search:")
    cross_results = kb.cross_department_search(
        "project management", ["engineering", "sales"], top_k=3
    )
    for result in cross_results:
        print(f"  - {result.get('title', 'Unknown')}: {result.get('score', 0):.3f}")

    # Department statistics
    print("\n📊 Department Statistics:")
    stats = kb.get_department_stats()
    for dept, info in stats.items():
        if "error" not in info:
            print(f"  {dept.title()}: {info['document_count']} documents")
        else:
            print(f"  {dept.title()}: Error - {info['error']}")

if __name__ == "__main__":
    main()
```

### **Scenario 2: Research Paper Analysis System**

#### **Research Configuration**
```python
# config/research.json
{
  "model_name": "all-mpnet-base-v2",
  "chunking_strategy": "semantic",
  "chunk_size": 300,
  "overlap": 50,
  "batch_size": 64,
  "preprocessing": {
    "remove_headers": false,
    "normalize_whitespace": true,
    "remove_code_blocks": false,
    "extract_links": true,
    "custom_filters": [
      {
        "pattern": "\\[\\d+\\]",
        "replacement": "",
        "description": "Remove citation numbers"
      },
      {
        "pattern": "Fig\\. \\d+",
        "replacement": "",
        "description": "Remove figure references"
      }
    ]
  },
  "performance": {
    "enable_gpu": true,
    "max_workers": 8,
    "memory_limit_mb": 8192
  }
}
```

#### **Research Analysis Implementation**
```python
#!/usr/bin/env python3
"""
Research paper analysis system with advanced semantic search
and citation analysis capabilities.
"""

from config import AdvancedConfig
from notes_mcp_server import NotesMCPServer
from collections import defaultdict
import re
import json

class ResearchAnalysisSystem:
    def __init__(self, config_path="config/research.json"):
        self.config = AdvancedConfig()
        self.config.load_config(config_path)
        self.server = NotesMCPServer(self.config)

        # Research-specific patterns
        self.citation_pattern = re.compile(r'\[(\d+)\]')
        self.doi_pattern = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
        self.author_pattern = re.compile(r'([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+|[A-Z][a-z]+, [A-Z]\.)')

    def analyze_paper(self, paper_content):
        """Analyze a research paper for key components."""
        analysis = {
            "citations": self.extract_citations(paper_content),
            "doi": self.extract_doi(paper_content),
            "authors": self.extract_authors(paper_content),
            "key_concepts": self.extract_key_concepts(paper_content),
            "methodology": self.identify_methodology(paper_content),
            "findings": self.extract_findings(paper_content)
        }
        return analysis

    def extract_citations(self, content):
        """Extract citation references."""
        citations = self.citation_pattern.findall(content)
        return list(set(citations))  # Remove duplicates

    def extract_doi(self, content):
        """Extract DOI if present."""
        doi_match = self.doi_pattern.search(content)
        return doi_match.group(0) if doi_match else None

    def extract_authors(self, content):
        """Extract author names."""
        authors = self.author_pattern.findall(content)
        return list(set(authors))

    def extract_key_concepts(self, content):
        """Extract key technical concepts using semantic search."""
        # Use the search system to find related concepts
        concepts_query = "key concepts methodology findings"
        results = self.server.search_notes(concepts_query, top_k=10)

        concepts = []
        for result in results:
            # Extract noun phrases and technical terms
            text = result.get('content', '')
            # Simple noun phrase extraction (can be enhanced with NLP)
            words = text.split()
            concepts.extend([word for word in words if len(word) > 4])

        return list(set(concepts))[:20]  # Top 20 unique concepts

    def identify_methodology(self, content):
        """Identify research methodology."""
        methodology_keywords = [
            "experiment", "survey", "case study", "qualitative",
            "quantitative", "mixed methods", "longitudinal",
            "cross-sectional", "meta-analysis", "systematic review"
        ]

        found_methods = []
        content_lower = content.lower()

        for method in methodology_keywords:
            if method in content_lower:
                found_methods.append(method)

        return found_methods

    def extract_findings(self, content):
        """Extract key findings and results."""
        # Look for result/discussion sections
        result_patterns = [
            r'results?.*?(?=conclusion|discussion|future work|$)',
            r'findings.*?(?=conclusion|discussion|future work|$)',
            r'conclusion.*?(?=future work|references|$)',
            r'discussion.*?(?=future work|references|$)'
        ]

        findings = []
        content_lower = content.lower()

        for pattern in result_patterns:
            matches = re.findall(pattern, content_lower, re.DOTALL | re.IGNORECASE)
            findings.extend(matches)

        return findings[:5]  # Top 5 findings

    def semantic_similarity_search(self, query, top_k=10):
        """Perform semantic similarity search across research papers."""
        return self.server.search_notes(query, top_k=top_k)

    def find_related_papers(self, paper_id, top_k=5):
        """Find papers related to a given paper."""
        # Get the original paper content
        original_paper = self.server.get_document_by_id(paper_id)
        if not original_paper:
            return []

        # Extract key concepts from original paper
        analysis = self.analyze_paper(original_paper.get('content', ''))

        # Search for papers with similar concepts
        search_terms = ' '.join(analysis.get('key_concepts', [])[:10])
        related_papers = self.server.search_notes(search_terms, top_k=top_k)

        # Filter out the original paper
        related_papers = [p for p in related_papers if p.get('id') != paper_id]

        return related_papers

    def generate_research_summary(self, paper_ids):
        """Generate a summary of multiple research papers."""
        summaries = []

        for paper_id in paper_ids:
            paper = self.server.get_document_by_id(paper_id)
            if paper:
                analysis = self.analyze_paper(paper.get('content', ''))
                summaries.append({
                    'id': paper_id,
                    'title': paper.get('title', 'Unknown'),
                    'analysis': analysis
                })

        # Aggregate findings across papers
        all_concepts = []
        all_methods = []
        all_findings = []

        for summary in summaries:
            analysis = summary['analysis']
            all_concepts.extend(analysis.get('key_concepts', []))
            all_methods.extend(analysis.get('methodology', []))
            all_findings.extend(analysis.get('findings', []))

        # Find common themes
        concept_counts = defaultdict(int)
        for concept in all_concepts:
            concept_counts[concept] += 1

        method_counts = defaultdict(int)
        for method in all_methods:
            method_counts[method] += 1

        return {
            'paper_summaries': summaries,
            'common_concepts': dict(sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'common_methods': dict(sorted(method_counts.items(), key=lambda x: x[1], reverse=True)),
            'aggregated_findings': list(set(all_findings))[:10]
        }

def main():
    # Initialize research analysis system
    research_system = ResearchAnalysisSystem()

    print("🔬 Research Analysis System Demo")
    print("=" * 50)

    # Example: Analyze a sample research paper
    sample_paper = """
    This paper presents a novel approach to machine learning optimization [1].
    The methodology involves deep neural networks and gradient descent algorithms.
    DOI: 10.1234/example.2023.001

    Authors: John Smith, Jane Doe

    Key findings include improved accuracy by 15% and reduced training time by 30%.
    The experimental results demonstrate significant improvements over baseline methods.
    """

    print("\n📄 Paper Analysis:")
    analysis = research_system.analyze_paper(sample_paper)

    print(f"Citations: {analysis['citations']}")
    print(f"DOI: {analysis['doi']}")
    print(f"Authors: {analysis['authors']}")
    print(f"Methodology: {analysis['methodology']}")
    print(f"Key Concepts: {analysis['key_concepts'][:5]}...")

    # Semantic search example
    print("\n🔍 Semantic Search:")
    search_results = research_system.semantic_similarity_search(
        "machine learning optimization techniques", top_k=3
    )

    for result in search_results:
        print(f"  - {result.get('title', 'Unknown')}: {result.get('score', 0):.3f}")

    print("\n✅ Research analysis complete!")

if __name__ == "__main__":
    main()
```

### **Scenario 3: Personal Knowledge Management**

#### **Personal Configuration**
```python
# config/personal.json
{
  "model_name": "all-MiniLM-L6-v2",
  "chunking_strategy": "heading_based",
  "chunk_size": 150,
  "batch_size": 32,
  "preprocessing": {
    "remove_headers": false,
    "normalize_whitespace": true,
    "extract_links": true,
    "custom_filters": [
      {
        "pattern": "#\\w+",
        "replacement": "",
        "description": "Remove hashtags"
      }
    ]
  },
  "performance": {
    "enable_gpu": false,
    "max_workers": 4,
    "memory_limit_mb": 2048
  }
}
```

#### **Personal Knowledge Manager**
```python
#!/usr/bin/env python3
"""
Personal knowledge management system with advanced organization
and retrieval capabilities.
"""

from config import AdvancedConfig
from notes_mcp_server import NotesMCPServer
from datetime import datetime, timedelta
import json
import os

class PersonalKnowledgeManager:
    def __init__(self, config_path="config/personal.json"):
        self.config = AdvancedConfig()
        self.config.load_config(config_path)
        self.server = NotesMCPServer(self.config)

        # Personal knowledge categories
        self.categories = {
            "work": ["meeting", "project", "deadline", "colleague"],
            "learning": ["course", "tutorial", "book", "research"],
            "personal": ["health", "finance", "travel", "hobby"],
            "ideas": ["innovation", "brainstorm", "concept", "invention"]
        }

        # Initialize personal database
        self.personal_db = "personal_knowledge.json"
        self.load_personal_db()

    def load_personal_db(self):
        """Load personal knowledge database."""
        if os.path.exists(self.personal_db):
            with open(self.personal_db, 'r') as f:
                self.personal_data = json.load(f)
        else:
            self.personal_data = {
                "notes": [],
                "tags": {},
                "categories": {},
                "insights": []
            }

    def save_personal_db(self):
        """Save personal knowledge database."""
        with open(self.personal_db, 'w') as f:
            json.dump(self.personal_data, f, indent=2, default=str)

    def add_personal_note(self, content, tags=None, category=None):
        """Add a personal note with metadata."""
        note = {
            "id": f"personal_{len(self.personal_data['notes'])}",
            "content": content,
            "tags": tags or [],
            "category": category,
            "created": datetime.now(),
            "last_modified": datetime.now()
        }

        self.personal_data["notes"].append(note)

        # Update tag index
        for tag in note["tags"]:
            if tag not in self.personal_data["tags"]:
                self.personal_data["tags"][tag] = []
            self.personal_data["tags"][tag].append(note["id"])

        # Update category index
        if category:
            if category not in self.personal_data["categories"]:
                self.personal_data["categories"][category] = []
            self.personal_data["categories"][category].append(note["id"])

        self.save_personal_db()
        return note["id"]

    def categorize_note(self, note_id, category):
        """Automatically categorize a note based on content."""
        note = self.get_note_by_id(note_id)
        if not note:
            return False

        content_lower = note["content"].lower()

        # Check against category keywords
        for cat, keywords in self.categories.items():
            if any(keyword in content_lower for keyword in keywords):
                note["category"] = cat
                self.save_personal_db()
                return cat

        return None

    def get_note_by_id(self, note_id):
        """Get a note by ID."""
        for note in self.personal_data["notes"]:
            if note["id"] == note_id:
                return note
        return None

    def search_personal_knowledge(self, query, category=None, tags=None, top_k=10):
        """Search personal knowledge with filters."""
        # First search in semantic index
        semantic_results = self.server.search_notes(query, top_k=top_k*2)

        # Filter by category and tags
        filtered_results = []

        for result in semantic_results:
            note_id = result.get("id", "")
            note = self.get_note_by_id(note_id)

            if not note:
                continue

            # Category filter
            if category and note.get("category") != category:
                continue

            # Tags filter
            if tags:
                note_tags = set(note.get("tags", []))
                query_tags = set(tags)
                if not query_tags.issubset(note_tags):
                    continue

            filtered_results.append({
                **result,
                "category": note.get("category"),
                "tags": note.get("tags"),
                "created": note.get("created")
            })

        return filtered_results[:top_k]

    def generate_daily_insight(self):
        """Generate daily insights from personal knowledge."""
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)

        # Get recent notes
        recent_notes = [
            note for note in self.personal_data["notes"]
            if datetime.fromisoformat(note["created"]).date() >= week_ago
        ]

        insights = {
            "total_notes": len(recent_notes),
            "categories": {},
            "top_tags": {},
            "recent_activity": []
        }

        # Analyze categories
        for note in recent_notes:
            cat = note.get("category", "uncategorized")
            insights["categories"][cat] = insights["categories"].get(cat, 0) + 1

            # Analyze tags
            for tag in note.get("tags", []):
                insights["top_tags"][tag] = insights["top_tags"].get(tag, 0) + 1

            # Recent activity
            insights["recent_activity"].append({
                "id": note["id"],
                "category": cat,
                "tags": note["tags"],
                "created": note["created"]
            })

        # Sort insights
        insights["top_tags"] = dict(sorted(
            insights["top_tags"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])

        insights["categories"] = dict(sorted(
            insights["categories"].items(),
            key=lambda x: x[1],
            reverse=True
        ))

        insights["recent_activity"] = insights["recent_activity"][-5:]  # Last 5

        return insights

    def find_connections(self, note_id):
        """Find connections between notes based on content similarity."""
        note = self.get_note_by_id(note_id)
        if not note:
            return []

        # Search for similar content
        similar_notes = self.server.search_notes(
            note["content"][:200],  # First 200 chars
            top_k=5
        )

        # Filter out the original note
        connections = [
            n for n in similar_notes
            if n.get("id") != note_id
        ]

        return connections

def main():
    # Initialize personal knowledge manager
    pkm = PersonalKnowledgeManager()

    print("📚 Personal Knowledge Management Demo")
    print("=" * 50)

    # Add some personal notes
    print("\n📝 Adding Personal Notes:")

    note1_id = pkm.add_personal_note(
        "Had a great meeting with the team about the new ML project. Discussed deployment strategies and timeline.",
        tags=["work", "meeting", "ml"],
        category="work"
    )
    print(f"Added work note: {note1_id}")

    note2_id = pkm.add_personal_note(
        "Started reading 'Deep Learning' book. Chapter 3 about neural networks is fascinating.",
        tags=["learning", "book", "deep-learning"],
        category="learning"
    )
    print(f"Added learning note: {note2_id}")

    note3_id = pkm.add_personal_note(
        "Great idea for a personal project: AI-powered note organizer using semantic search.",
        tags=["ideas", "ai", "project"],
        category="ideas"
    )
    print(f"Added idea note: {note3_id}")

    # Auto-categorize notes
    print("\n🏷️  Auto-Categorization:")
    for note_id in [note1_id, note2_id, note3_id]:
        category = pkm.categorize_note(note_id, None)
        if category:
            print(f"Note {note_id} categorized as: {category}")

    # Search personal knowledge
    print("\n🔍 Personal Knowledge Search:")
    search_results = pkm.search_personal_knowledge(
        "machine learning project",
        category="work",
        top_k=3
    )

    for result in search_results:
        print(f"  - {result.get('title', 'Unknown')}: {result.get('score', 0):.3f}")
        print(f"    Category: {result.get('category')}, Tags: {result.get('tags')}")

    # Generate daily insights
    print("\n📊 Daily Insights:")
    insights = pkm.generate_daily_insight()
    print(f"Total notes this week: {insights['total_notes']}")
    print(f"Categories: {insights['categories']}")
    print(f"Top tags: {insights['top_tags']}")

    # Find connections
    print("\n🔗 Finding Connections:")
    connections = pkm.find_connections(note1_id)
    print(f"Notes connected to {note1_id}:")
    for conn in connections:
        print(f"  - {conn.get('title', 'Unknown')}: {conn.get('score', 0):.3f}")

    print("\n✅ Personal knowledge management demo complete!")

if __name__ == "__main__":
    main()
```

## 🧪 **Testing & Validation Examples**

### **Example 4: Configuration Testing Suite**

```python
#!/usr/bin/env python3
"""
Comprehensive testing suite for advanced configuration system.
"""

import unittest
from config import AdvancedConfig
from notes_mcp_server import NotesMCPServer
import tempfile
import os
import json

class TestAdvancedConfiguration(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = AdvancedConfig()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_environment_detection(self):
        """Test automatic environment detection."""
        # Test development environment
        os.environ['MCP_ENVIRONMENT'] = 'development'
        config = AdvancedConfig()
        self.assertEqual(config.environment.value, 'development')

        # Test production environment
        os.environ['MCP_ENVIRONMENT'] = 'production'
        config = AdvancedConfig()
        self.assertEqual(config.environment.value, 'production')

    def test_model_switching(self):
        """Test dynamic model switching."""
        # Start with default model
        initial_model = self.config.model_name

        # Switch to different model
        self.config.set_model("all-mpnet-base-v2")
        self.assertEqual(self.config.model_name, "all-mpnet-base-v2")

        # Switch back
        self.config.set_model(initial_model)
        self.assertEqual(self.config.model_name, initial_model)

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        is_valid, errors = self.config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Invalid configuration (simulate)
        self.config.chunk_size = -1  # Invalid
        is_valid, errors = self.config.validate()
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_configuration_persistence(self):
        """Test saving and loading configuration."""
        # Modify configuration
        original_model = self.config.model_name
        self.config.set_model("all-mpnet-base-v2")
        self.config.set_batch_size(256)

        # Save configuration
        config_path = os.path.join(self.temp_dir, "test_config.json")
        self.config.save_config(config_path)

        # Load configuration in new instance
        new_config = AdvancedConfig()
        new_config.load_config(config_path)

        # Verify settings were preserved
        self.assertEqual(new_config.model_name, "all-mpnet-base-v2")
        self.assertEqual(new_config.batch_size, 256)

    def test_performance_settings(self):
        """Test performance-related settings."""
        # Test GPU settings
        self.config.enable_gpu()
        self.assertTrue(self.config.performance.enable_gpu)

        self.config.disable_gpu()
        self.assertFalse(self.config.performance.enable_gpu)

        # Test batch size
        self.config.set_batch_size(128)
        self.assertEqual(self.config.batch_size, 128)

        # Test memory limits
        self.config.set_memory_limit_mb(4096)
        self.assertEqual(self.config.performance.memory_limit_mb, 4096)

    def test_chunking_strategies(self):
        """Test different chunking strategies."""
        strategies = ["fixed", "sentence", "heading", "semantic", "hybrid"]

        for strategy in strategies:
            with self.subTest(strategy=strategy):
                self.config.set_chunking_strategy(strategy)
                self.assertEqual(self.config.chunking_strategy, strategy)

    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline configuration."""
        # Test custom filters
        custom_filters = [
            {
                "pattern": "test_pattern",
                "replacement": "replacement",
                "description": "Test filter"
            }
        ]

        self.config.preprocessing.custom_filters = custom_filters
        self.assertEqual(len(self.config.preprocessing.custom_filters), 1)
        self.assertEqual(self.config.preprocessing.custom_filters[0]["pattern"], "test_pattern")

    def test_security_settings(self):
        """Test security-related settings."""
        # Test authentication
        self.config.security.enable_auth = True
        self.assertTrue(self.config.security.enable_auth)

        # Test SSL
        self.config.security.ssl_enabled = True
        self.assertTrue(self.config.security.ssl_enabled)

        # Test rate limiting
        self.config.security.rate_limit = "1000/hour"
        self.assertEqual(self.config.security.rate_limit, "1000/hour")

    def test_monitoring_settings(self):
        """Test monitoring and logging settings."""
        # Test metrics
        self.config.monitoring.enable_metrics = True
        self.assertTrue(self.config.monitoring.enable_metrics)

        # Test log level
        self.config.monitoring.log_level = "DEBUG"
        self.assertEqual(self.config.monitoring.log_level, "DEBUG")

        # Test metrics port
        self.config.monitoring.metrics_port = 9090
        self.assertEqual(self.config.monitoring.metrics_port, 9090)

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.config = AdvancedConfig()
        self.server = NotesMCPServer(self.config)

    def test_server_initialization(self):
        """Test server initialization with configuration."""
        self.assertIsNotNone(self.server)
        self.assertEqual(self.server.config.model_name, self.config.model_name)

    def test_search_functionality(self):
        """Test search functionality with configuration."""
        # This would require actual notes to search
        # For now, just test that the method exists and doesn't crash
        try:
            results = self.server.search_notes("test query", top_k=1)
            self.assertIsInstance(results, list)
        except Exception as e:
            # Expected if no index exists
            self.assertIn("index", str(e).lower())

def run_tests():
    """Run the test suite."""
    print("🧪 Running Advanced Configuration Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestAdvancedConfiguration))
    suite.addTest(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print results
    print("\n📊 Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        for failure in result.failures:
            print(f"FAIL: {failure[0]}")
            print(f"  {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(f"  {error[1]}")

if __name__ == "__main__":
    run_tests()
```

## 🚀 **Deployment Examples**

### **Example 5: Docker Deployment with Advanced Config**

#### **Dockerfile**
```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV MCP_ENVIRONMENT=production
ENV MCP_MODEL_NAME=all-mpnet-base-v2
ENV MCP_ENABLE_GPU=false
ENV MCP_BATCH_SIZE=128
ENV MCP_MAX_WORKERS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/config /app/notes /app/logs

# Copy configuration files
COPY config/production.json /app/config/
COPY config/models.json /app/config/
COPY config/preprocessing.json /app/config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8181/health || exit 1

# Expose port
EXPOSE 8181

# Start the application
CMD ["python", "notes_mcp_server.py"]
```

#### **Docker Compose**
```yaml
version: '3.8'

services:
  markdown-mcp:
    build: .
    ports:
      - "8181:8181"
    volumes:
      - ./notes:/app/notes:ro
      - ./config:/app/config:ro
      - ./logs:/app/logs
    environment:
      - MCP_ENVIRONMENT=production
      - MCP_MODEL_NAME=all-mpnet-base-v2
      - MCP_ENABLE_GPU=false
      - MCP_BATCH_SIZE=128
      - MCP_MAX_WORKERS=8
      - MCP_ADMIN_TOKEN=${MCP_ADMIN_TOKEN}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8181/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # Optional: Monitoring service
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  # Optional: Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus

volumes:
  grafana_data:
```

#### **Production Deployment Script**
```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

echo "🚀 Deploying Advanced Markdown-MCP System"
echo "========================================="

# Configuration
ENVIRONMENT=${1:-production}
VERSION=$(git rev-parse --short HEAD)
IMAGE_NAME="markdown-mcp:${VERSION}"

echo "Environment: ${ENVIRONMENT}"
echo "Version: ${VERSION}"
echo "Image: ${IMAGE_NAME}"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Save configuration
echo "💾 Saving configuration..."
python -c "
from config import AdvancedConfig
config = AdvancedConfig()
config.load_environment('${ENVIRONMENT}')
config.save_config('config/deployed_config.json')
"

# Deploy with docker-compose
echo "🐳 Deploying with Docker Compose..."
export MCP_ENVIRONMENT=${ENVIRONMENT}
export MCP_ADMIN_TOKEN=$(openssl rand -hex 32)

docker-compose up -d

# Wait for health check
echo "🏥 Waiting for service to be healthy..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if curl -f http://localhost:8181/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ $elapsed -ge $timeout ]; then
    echo "❌ Service failed to become healthy within ${timeout} seconds"
    exit 1
fi

# Run post-deployment tests
echo "🧪 Running post-deployment tests..."
python -c "
from config import AdvancedConfig
from notes_mcp_server import NotesMCPServer

config = AdvancedConfig()
config.load_environment('${ENVIRONMENT}')
server = NotesMCPServer(config)

# Test basic functionality
try:
    results = server.search_notes('test', top_k=1)
    print('✅ Search functionality working')
except Exception as e:
    print(f'❌ Search test failed: {e}')
    exit(1)

print('✅ All post-deployment tests passed!')
"

echo "🎉 Deployment completed successfully!"
echo "🌐 Service available at: http://localhost:8181"
echo "📊 Metrics available at: http://localhost:9090"
echo "📈 Grafana available at: http://localhost:3000"
```

## 📊 **Performance Benchmarking**

### **Example 6: Performance Benchmark Suite**

```python
#!/usr/bin/env python3
"""
Performance benchmarking suite for the advanced configuration system.
"""

import time
import psutil
import GPUtil
from config import AdvancedConfig
from notes_mcp_server import NotesMCPServer
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
import json

class PerformanceBenchmark:
    def __init__(self):
        self.config = AdvancedConfig()
        self.server = NotesMCPServer(self.config)
        self.results = {}

    def measure_system_resources(self):
        """Measure current system resource usage."""
        resources = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }

        # GPU information if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                resources.update({
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': gpu.memoryUtil * 100,
                    'gpu_temperature': gpu.temperature
                })
        except:
            pass

        return resources

    def benchmark_model_inference(self, queries: List[str], model_name: str):
        """Benchmark model inference performance."""
        print(f"🔬 Benchmarking model: {model_name}")

        # Switch model
        self.config.set_model(model_name)

        results = {
            'model': model_name,
            'queries': [],
            'total_time': 0,
            'avg_time': 0,
            'min_time': float('inf'),
            'max_time': 0
        }

        for query in queries:
            # Measure resources before
            resources_before = self.measure_system_resources()

            # Perform search
            start_time = time.time()
            search_results = self.server.search_notes(query, top_k=10)
            end_time = time.time()

            # Measure resources after
            resources_after = self.measure_system_resources()

            query_time = end_time - start_time
            results['queries'].append({
                'query': query,
                'time': query_time,
                'results_count': len(search_results),
                'resources_before': resources_before,
                'resources_after': resources_after
            })

            results['total_time'] += query_time
            results['min_time'] = min(results['min_time'], query_time)
            results['max_time'] = max(results['max_time'], query_time)

        results['avg_time'] = results['total_time'] / len(queries)
        return results

    def benchmark_chunking_strategies(self, text: str, strategies: List[str]):
        """Benchmark different chunking strategies."""
        print("📊 Benchmarking chunking strategies")

        results = {}

        for strategy in strategies:
            print(f"  Testing strategy: {strategy}")

            self.config.set_chunking_strategy(strategy)

            start_time = time.time()
            # Simulate chunking process
            chunks = self.server.chunk_text(text)
            end_time = time.time()

            chunking_time = end_time - start_time

            results[strategy] = {
                'time': chunking_time,
                'chunk_count': len(chunks),
                'avg_chunk_size': sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                'chunks': chunks[:5]  # Sample first 5 chunks
            }

        return results

    def benchmark_batch_sizes(self, queries: List[str], batch_sizes: List[int]):
        """Benchmark different batch sizes."""
        print("🔢 Benchmarking batch sizes")

        results = {}

        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")

            self.config.set_batch_size(batch_size)

            start_time = time.time()
            # Process queries in batches
            all_results = []
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_results = []
                for query in batch_queries:
                    results = self.server.search_notes(query, top_k=5)
                    batch_results.extend(results)
                all_results.extend(batch_results)
            end_time = time.time()

            total_time = end_time - start_time

            results[batch_size] = {
                'total_time': total_time,
                'avg_time_per_query': total_time / len(queries),
                'total_results': len(all_results)
            }

        return results

    def run_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark."""
        print("🚀 Running Comprehensive Performance Benchmark")
        print("=" * 60)

        # Test queries
        test_queries = [
            "machine learning algorithms",
            "neural network optimization",
            "data preprocessing techniques",
            "model evaluation metrics",
            "deep learning architectures",
            "computer vision applications",
            "natural language processing",
            "reinforcement learning",
            "supervised learning methods",
            "unsupervised learning techniques"
        ]

        # Benchmark models
        models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
        model_results = {}
        for model in models:
            try:
                model_results[model] = self.benchmark_model_inference(test_queries, model)
            except Exception as e:
                print(f"❌ Error benchmarking {model}: {e}")
                model_results[model] = {'error': str(e)}

        # Benchmark chunking strategies
        sample_text = """
        Machine learning is a subset of artificial intelligence that involves training algorithms
        to recognize patterns in data and make predictions or decisions without being explicitly
        programmed. Deep learning, a subset of machine learning, uses neural networks with multiple
        layers to model complex patterns in data. Neural networks are inspired by the structure
        of the human brain and consist of interconnected nodes called neurons.

        The process of training a neural network involves feeding it large amounts of data and
        adjusting the connections between neurons to minimize errors in predictions. This is
        typically done using optimization algorithms like gradient descent. The quality of the
        training data and the architecture of the network significantly impact the performance
        of the model.

        Applications of machine learning include image recognition, natural language processing,
        recommendation systems, autonomous vehicles, and medical diagnosis. As computational
        power increases and more data becomes available, the capabilities of machine learning
        systems continue to expand, leading to breakthroughs in various fields.
        """

        chunking_strategies = ["fixed", "sentence", "heading", "semantic", "hybrid"]
        chunking_results = self.benchmark_chunking_strategies(sample_text, chunking_strategies)

        # Benchmark batch sizes
        batch_sizes = [1, 4, 8, 16, 32, 64]
        batch_results = self.benchmark_batch_sizes(test_queries, batch_sizes)

        # Compile results
        benchmark_results = {
            'timestamp': time.time(),
            'system_info': self.measure_system_resources(),
            'model_benchmarks': model_results,
            'chunking_benchmarks': chunking_results,
            'batch_benchmarks': batch_results,
            'configuration': {
                'model_name': self.config.model_name,
                'chunking_strategy': self.config.chunking_strategy,
                'batch_size': self.config.batch_size,
                'enable_gpu': self.config.performance.enable_gpu
            }
        }

        # Save results
        with open('benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)

        return benchmark_results

    def generate_report(self, results: Dict):
        """Generate performance report."""
        print("\n📊 Performance Benchmark Report")
        print("=" * 60)

        # Model comparison
        print("\n🤖 Model Performance Comparison:")
        model_data = results.get('model_benchmarks', {})
        for model, data in model_data.items():
            if 'error' not in data:
                print(f"  {model}:")
                print(".2f")
                print(".2f")
                print(".2f")
                print(f"    Queries tested: {len(data.get('queries', []))}")

        # Chunking strategy comparison
        print("\n📝 Chunking Strategy Comparison:")
        chunking_data = results.get('chunking_benchmarks', {})
        for strategy, data in chunking_data.items():
            print(f"  {strategy}:")
            print(".2f")
            print(f"    Chunks created: {data.get('chunk_count', 0)}")
            print(".0f")

        # Batch size comparison
        print("\n🔢 Batch Size Performance:")
        batch_data = results.get('batch_benchmarks', {})
        for batch_size, data in batch_data.items():
            print(f"  Batch size {batch_size}:")
            print(".2f")
            print(".2f")

        # System resources
        print("\n💻 System Resources:")
        sys_info = results.get('system_info', {})
        print(".1f")
        print(".1f")
        print(".0f")
        if 'gpu_memory_percent' in sys_info:
            print(".1f")

        print("\n💾 Results saved to: benchmark_results.json")

def main():
    benchmark = PerformanceBenchmark()

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()

    # Generate report
    benchmark.generate_report(results)

    print("\n✅ Benchmarking complete!")

if __name__ == "__main__":
    main()
```

---

*These practical examples demonstrate the full power and flexibility of the advanced configuration system. Each scenario shows how to leverage different features for specific use cases, from enterprise deployments to personal knowledge management. The testing and benchmarking examples ensure the system performs optimally across different configurations.*
