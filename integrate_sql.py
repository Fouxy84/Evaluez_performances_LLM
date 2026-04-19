"""
Main integration script for SQL Tool with RAG Agent.

This script:
1. Initializes the database
2. Loads Excel data
3. Creates the SQL Tool
4. Integrates with the RAG evaluator
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from load_excel_to_db import ExcelIngestionPipeline
from sql_tool import create_sql_tool, SQLTool
from utils.database import init_db, SessionLocal

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class RAGWithSQLIntegration:
    """RAG system integrated with SQL database for structured queries."""

    def __init__(self):
        """Initialize the system."""
        self.db = SessionLocal()
        self.sql_tool = None
        self.pipeline = None

    def setup_database(self, force_reinit: bool = False):
        """
        Set up the database schema and load data.
        
        Args:
            force_reinit: Force re-initialization of schema
        """
        logger.info("Setting up database...")
        
        # Initialize schema
        init_db()
        
        # Load data from Excel
        self.pipeline = ExcelIngestionPipeline(self.db)
        results = self.pipeline.ingest_all("inputs")
        
        logger.info(f"✓ Database setup complete")
        logger.info(f"  - Teams: {len(results['teams'])}")
        logger.info(f"  - Players: {len(results['players'])}")
        logger.info(f"  - Matches: {len(results['matches'])}")
        logger.info(f"  - Stats: {len(results['stats'])}")
        
        if results['errors']:
            logger.warning(f"  - Errors: {len(results['errors'])}")

    def create_sql_tool(self):
        """Create and configure the SQL tool."""
        logger.info("Creating SQL tool...")
        self.sql_tool = SQLTool(self.db)
        logger.info("✓ SQL tool created")
        return self.sql_tool

    def detect_query_type(self, question: str) -> Optional[str]:
        """
        Detect the type of query based on question content.
        
        Returns:
            Query type: 'sql', 'rag', or 'hybrid'
        """
        question_lower = question.lower()
        
        # Keywords indicating SQL queries
        sql_keywords = [
            "how many", "what is", "compare", "best", "top", "win",
            "record", "average", "stats", "points", "assists", "season",
            "percentage", "shooting", "leader", "ranking", "vs"
        ]
        
        # Keywords indicating contextual/RAG queries
        rag_keywords = [
            "explain", "why", "context", "background", "meaning",
            "analyse", "discuss", "debate", "opinion", "argument"
        ]
        
        sql_score = sum(1 for kw in sql_keywords if kw in question_lower)
        rag_score = sum(1 for kw in rag_keywords if kw in question_lower)
        
        if sql_score > rag_score:
            return "sql"
        elif rag_score > sql_score:
            return "rag"
        else:
            return "hybrid"

    def answer_question(self, question: str) -> str:
        """
        Answer a question using SQL or RAG as appropriate.
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        query_type = self.detect_query_type(question)
        logger.info(f"Query type: {query_type}")
        
        if query_type == "sql" and self.sql_tool:
            logger.info("Using SQL tool...")
            try:
                result = self.sql_tool.player_stats("Curry")  # Placeholder
                if result is not None:
                    return f"SQL Result:\n{self.sql_tool.format_results(result)}"
            except Exception as e:
                logger.error(f"SQL query error: {e}")
        
        if query_type == "rag" or query_type == "hybrid":
            logger.info("Using RAG approach...")
            # Integration with RAG system would go here
            return f"RAG Answer: [Integration with RAG system pending]"
        
        return "Unable to answer question."

    def get_system_info(self) -> str:
        """Get information about the current system setup."""
        info = """
╔════════════════════════════════════════════════════════════════════════╗
║                     RAG + SQL SYSTEM INFORMATION                        ║
╚════════════════════════════════════════════════════════════════════════╝

Components:
✓ Database Schema (SQLAlchemy)
  - Teams, Players, Matches, Stats, Reports tables
  - Fully relational design with foreign keys
  
✓ Excel Data Pipeline
  - Automatic data validation via Pydantic
  - Flexible column mapping for different Excel formats
  - Error tracking and logging

✓ SQL Query Tool
  - Few-shot learning for SQL generation
  - Predefined templates for common queries
  - Safe query execution with validation
  - Result formatting and caching

✓ Query Type Detection
  - Automatic routing: SQL vs RAG vs Hybrid
  - Keyword-based classification
  - Extensible for custom patterns

Integration Points:
- SQL Tool: For structured, numerical queries
  (stats, comparisons, rankings, records)
- RAG System: For contextual, analytical queries
  (explanations, discussions, analysis)

Database Location: ./nba_stats.db
Input Data Location: ./inputs/
Output Results: ./data/

Ready to process user queries!
        """
        return info


def demonstrate_sql_tool():
    """Demonstrate SQL tool functionality."""
    print("\n" + "=" * 80)
    print("SQL TOOL DEMONSTRATION")
    print("=" * 80)
    
    sql_tool = SQLTool(SessionLocal())
    
    print("\n1. Few-Shot Examples for SQL Generation:")
    print(sql_tool.get_few_shot_context("player_stats"))
    
    print("\n2. Available Query Templates:")
    for name in sql_tool.QUERY_TEMPLATES.keys():
        print(f"   - {name}")


def main():
    """Main entry point."""
    import sys
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║              RAG + SQL DATABASE INTEGRATION FOR NBA STATS              ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    system = RAGWithSQLIntegration()
    
    # Setup database
    system.setup_database()
    
    # Create SQL tool
    system.create_sql_tool()
    
    # Print system info
    print(system.get_system_info())
    
    # Demonstrate SQL tool
    demonstrate_sql_tool()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
1. Run Excel data loading:
   python load_excel_to_db.py

2. Test SQL queries:
   python sql_tool.py

3. View database documentation:
   python DATABASE_DOCUMENTATION.py

4. Integrate with RAG agent (in evaluate_ragas.py):
   from sql_tool import create_sql_tool
   sql_tool = create_sql_tool(db)
   agent_tools.append(sql_tool)

5. Check database directly:
   sqlite3 nba_stats.db
   .tables
   .schema
    """)


if __name__ == "__main__":
    main()
