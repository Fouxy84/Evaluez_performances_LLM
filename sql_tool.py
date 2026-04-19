"""
SQL Tool for LangChain to query NBA statistics.

Features:
- Few-shot examples for SQL generation
- Query validation and safety checks
- Result formatting and caching
- Database query execution
"""

from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
from sqlalchemy.sql import text
from sqlalchemy.orm import Session
import pandas as pd
import logging
from utils.database import SessionLocal
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)


@dataclass
class SimpleTool:
    name: str
    func: Callable[[str], str]
    description: str


class SQLTool:
    """Tool for executing SQL queries on NBA statistics database."""

    # Few-shot SQL examples for different question types
    FEW_SHOT_EXAMPLES = {
        "player_stats": {
            "question": "What are Steph Curry's stats?",
            "sql": """
            SELECT 
                p.name,
                m.date,
                t.name AS opponent,
                s.points,
                s.assists,
                s.total_rebounds,
                s.three_pointers_made,
                s.three_point_percentage
            FROM stats s
            JOIN players p ON s.player_id = p.id
            JOIN matches m ON s.match_id = m.id
            JOIN teams t ON (CASE 
                WHEN m.home_team_id = p.team_id THEN m.away_team_id
                ELSE m.home_team_id
            END) = t.id
            WHERE LOWER(p.name) LIKE LOWER(:player_name)
            ORDER BY m.date DESC
            LIMIT 10;
            """
        },
        "player_comparison": {
            "question": "Compare Curry and Durant's 3-point shooting",
            "sql": """
            SELECT 
                p.name,
                ROUND(AVG(s.three_pointers_made), 2) AS avg_3pm,
                ROUND(AVG(s.three_pointers_attempted), 2) AS avg_3pa,
                ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
                COUNT(*) AS games
            FROM stats s
            JOIN players p ON s.player_id = p.id
            WHERE LOWER(p.name) LIKE LOWER(:p1) OR LOWER(p.name) LIKE LOWER(:p2)
            GROUP BY p.id, p.name
            ORDER BY avg_3p_pct DESC;
            """
        },
        "team_performance": {
            "question": "How many wins does Lakers have this season?",
            "sql": """
            SELECT 
                t.name,
                COUNT(CASE WHEN m.home_team_id = t.id AND m.home_score > m.away_score THEN 1
                         WHEN m.away_team_id = t.id AND m.away_score > m.home_score THEN 1 END) AS wins,
                COUNT(CASE WHEN m.home_team_id = t.id AND m.home_score < m.away_score THEN 1
                         WHEN m.away_team_id = t.id AND m.away_score < m.home_score THEN 1 END) AS losses
            FROM teams t
            LEFT JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            WHERE LOWER(t.name) LIKE LOWER(:team_name) AND m.season = :season
            GROUP BY t.id, t.name;
            """
        },
        "home_away_comparison": {
            "question": "Compare home and away scoring averages",
            "sql": """
            SELECT 
                p.name,
                'Home' AS game_location,
                ROUND(AVG(s.points), 1) AS avg_points,
                ROUND(AVG(s.assists), 1) AS avg_assists,
                COUNT(*) AS games
            FROM stats s
            JOIN players p ON s.player_id = p.id
            JOIN matches m ON s.match_id = m.id
            WHERE m.home_team_id = p.team_id AND LOWER(p.name) LIKE LOWER(:player_name)
            GROUP BY p.id, p.name
            UNION ALL
            SELECT 
                p.name,
                'Away' AS game_location,
                ROUND(AVG(s.points), 1) AS avg_points,
                ROUND(AVG(s.assists), 1) AS avg_assists,
                COUNT(*) AS games
            FROM stats s
            JOIN players p ON s.player_id = p.id
            JOIN matches m ON s.match_id = m.id
            WHERE m.away_team_id = p.team_id AND LOWER(p.name) LIKE LOWER(:player_name)
            GROUP BY p.id, p.name
            ORDER BY game_location;
            """
        },
        "top_performers": {
            "question": "Who are the top scorers this season?",
            "sql": """
            SELECT 
                p.name,
                t.name AS team,
                ROUND(AVG(s.points), 1) AS avg_points,
                ROUND(AVG(s.assists), 1) AS avg_assists,
                ROUND(AVG(s.total_rebounds), 1) AS avg_rebounds,
                COUNT(*) AS games
            FROM stats s
            JOIN players p ON s.player_id = p.id
            JOIN matches m ON s.match_id = m.id
            JOIN teams t ON p.team_id = t.id
            WHERE m.season = ?
            GROUP BY p.id, p.name, t.id, t.name
            ORDER BY avg_points DESC
            LIMIT 15;
            """
        }
    }

    QUERY_TEMPLATES = {
        "player_season_stats": """
        SELECT 
            p.name,
            ROUND(AVG(s.points), 1) AS avg_points,
            ROUND(AVG(s.assists), 1) AS avg_assists,
            ROUND(AVG(s.total_rebounds), 1) AS avg_rebounds,
            ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
            ROUND(AVG(s.field_goal_percentage), 1) AS avg_fg_pct,
            COUNT(*) AS games_played
        FROM stats s
        JOIN players p ON s.player_id = p.id
        JOIN matches m ON s.match_id = m.id
        WHERE LOWER(p.name) LIKE LOWER(:player_name)
            AND m.season = :season
        GROUP BY p.id, p.name
        """,

        "player_vs_opponent": """
        SELECT 
            p.name,
            t.name AS opponent,
            COUNT(*) AS games,
            ROUND(AVG(s.points), 1) AS avg_points,
            ROUND(AVG(s.assists), 1) AS avg_assists,
            MAX(s.points) AS max_points
        FROM stats s
        JOIN players p ON s.player_id = p.id
        JOIN matches m ON s.match_id = m.id
        JOIN teams t ON (CASE 
            WHEN m.home_team_id = p.team_id THEN m.away_team_id
            ELSE m.home_team_id
        END) = t.id
        WHERE LOWER(p.name) LIKE LOWER(:player_name) AND LOWER(t.name) LIKE LOWER(:opponent_name)
        GROUP BY p.id, p.name, t.id, t.name
        """,

        "best_3pt_shooters": """
        SELECT 
            p.name,
            t.name AS team,
            ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
            ROUND(AVG(s.three_pointers_made), 1) AS avg_3pm,
            COUNT(*) AS games
        FROM stats s
        JOIN players p ON s.player_id = p.id
        JOIN teams t ON p.team_id = t.id
        JOIN matches m ON s.match_id = m.id
        WHERE m.season = :season AND s.three_pointers_attempted >= 3
        GROUP BY p.id, p.name, t.id, t.name
        HAVING COUNT(*) >= 10
        ORDER BY avg_3p_pct DESC
        LIMIT 20
        """,

        "team_comparison": """
        SELECT 
            t.name AS team,
            COUNT(CASE WHEN (m.home_team_id = t.id AND m.home_score > m.away_score)
                    OR (m.away_team_id = t.id AND m.away_score > m.home_score) THEN 1 END) AS wins,
            COUNT(CASE WHEN (m.home_team_id = t.id AND m.home_score < m.away_score)
                    OR (m.away_team_id = t.id AND m.away_score < m.home_score) THEN 1 END) AS losses,
            ROUND(AVG(CASE WHEN m.home_team_id = t.id THEN m.home_score
                          ELSE m.away_score END), 1) AS avg_points_for,
            ROUND(AVG(CASE WHEN m.home_team_id = t.id THEN m.away_score
                          ELSE m.home_score END), 1) AS avg_points_against
        FROM teams t
        LEFT JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
        WHERE m.season = :season
        GROUP BY t.id, t.name
        ORDER BY wins DESC
        """
    }

    def __init__(self, db: Optional[Session] = None):
        """Initialize SQL tool with database session."""
        self.db = db or SessionLocal()
        self.is_sqlite = "sqlite" in str(self.db.bind.url)

    def safe_query(self, sql: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        Execute SQL query safely with validation.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with results or None if error
        """
        try:
            # Basic validation
            if not sql or len(sql.strip()) == 0:
                logger.error("Empty SQL query")
                return None

            forbidden_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
            if any(keyword in sql.upper() for keyword in forbidden_keywords):
                logger.error("Query contains forbidden keywords")
                return None

            # Execute
            result = self.db.execute(text(sql), params or {})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df if not df.empty else None

        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None

    def get_few_shot_context(self, category: str) -> str:
        """
        Get few-shot examples for SQL generation.
        
        Args:
            category: Type of query (player_stats, comparison, etc.)
            
        Returns:
            Formatted few-shot examples
        """
        if category not in self.FEW_SHOT_EXAMPLES:
            category = list(self.FEW_SHOT_EXAMPLES.keys())[0]

        example = self.FEW_SHOT_EXAMPLES[category]
        return f"""
Example:
Question: {example['question']}
SQL: {example['sql']}
"""

    def player_stats(self, player_name: str, season: int = 2024) -> Optional[pd.DataFrame]:
        """Get player season statistics."""
        return self.safe_query(
            self.QUERY_TEMPLATES["player_season_stats"],
            {"player_name": f"%{player_name}%", "season": season}
        )

    def player_comparison(self, player1: str, player2: str, season: int = 2024) -> Optional[pd.DataFrame]:
        """Compare two players' statistics."""
        query = """
        SELECT 
            p.name,
            ROUND(AVG(s.points), 1) AS avg_points,
            ROUND(AVG(s.assists), 1) AS avg_assists,
            ROUND(AVG(s.total_rebounds), 1) AS avg_rebounds,
            ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
            COUNT(*) AS games
        FROM stats s
        JOIN players p ON s.player_id = p.id
        JOIN matches m ON s.match_id = m.id
        WHERE (LOWER(p.name) LIKE LOWER(:p1) OR LOWER(p.name) LIKE LOWER(:p2)) AND m.season = :season
        GROUP BY p.id, p.name
        """
        return self.safe_query(query, {
            "p1": f"%{player1}%",
            "p2": f"%{player2}%",
            "season": season
        })

    def top_scorers(self, season: int = 2024, limit: int = 15) -> Optional[pd.DataFrame]:
        """Get top scorers for the season."""
        query = self.QUERY_TEMPLATES["top_performers"]
        return self.safe_query(query, {"season": season})

    def team_stats(self, team_name: str, season: int = 2024) -> Optional[pd.DataFrame]:
        """Get team statistics."""
        query = """
        SELECT 
            t.name AS team,
            COUNT(CASE WHEN (m.home_team_id = t.id AND m.home_score > m.away_score)
                    OR (m.away_team_id = t.id AND m.away_score > m.home_score) THEN 1 END) AS wins,
            COUNT(CASE WHEN (m.home_team_id = t.id AND m.home_score < m.away_score)
                    OR (m.away_team_id = t.id AND m.away_score < m.home_score) THEN 1 END) AS losses
        FROM teams t
        LEFT JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id) AND m.season = :season
        WHERE LOWER(t.name) LIKE LOWER(:team_name)
        GROUP BY t.id, t.name
        """
        return self.safe_query(query, {"team_name": f"%{team_name}%", "season": season})

    def format_results(self, df: Optional[pd.DataFrame], max_rows: int = 10) -> str:
        """Format DataFrame results as readable text."""
        if df is None or df.empty:
            return "No results found."

        df_display = df.head(max_rows)
        return df_display.to_string(index=False)


def create_sql_tool(db: Optional[Session] = None) -> Tool:
    """Create a LangChain Tool for SQL queries."""
    sql_tool = SQLTool(db)

    def query_nba_stats(question: str) -> str:
        """
        Query NBA statistics database.
        
        Use this tool to answer questions about:
        - Player statistics (points, assists, rebounds, etc.)
        - Team performance (wins, losses, records)
        - Comparisons (player vs player, home vs away)
        - Top performers (leaders in various stats)
        
        Examples:
        - "What are Steph Curry's average points?"
        - "Compare Kevin Durant and LeBron James"
        - "Who are the top 3-point shooters?"
        - "How many wins does the Lakers have?"
        """
        # Simple keyword-based routing
        question_lower = question.lower()

        if "compare" in question_lower or "vs" in question_lower or "versus" in question_lower or "et" in question_lower:
            # Extract player names (simplified)
            words = question_lower.split()
            player1 = player2 = None
            for i, word in enumerate(words):
                if word in ["and", "et", "vs", "versus"]:
                    if i > 0:
                        player1 = words[i-1]
                    if i < len(words) - 1:
                        player2 = words[i+1]
            if player1 and player2:
                result = sql_tool.player_comparison(player1, player2)
                return sql_tool.format_results(result)

        elif "top" in question_lower or "leader" in question_lower or "meilleur" in question_lower or "victoires" in question_lower or "classement" in question_lower:
            result = sql_tool.top_scorers()
            return sql_tool.format_results(result)

        elif "team" in question_lower or "wins" in question_lower or "record" in question_lower:
            # Extract team name (simplified)
            for team_keyword in ["lakers", "celtics", "warriors", "heat", "suns", "warriors", "lakers", "celtics", "heat", "suns"]:
                if team_keyword in question_lower:
                    result = sql_tool.team_stats(team_keyword)
                    return sql_tool.format_results(result)

        else:
            # Default: look for player name
            words = [w for w in question_lower.split() if len(w) > 2]
            if words:
                result = sql_tool.player_stats(words[0])
                return sql_tool.format_results(result)

        return "Could not determine query type. Please rephrase your question."

    return SimpleTool(
        name="sql_nba_stats",
        func=query_nba_stats,
        description="Query NBA statistics database for player and team information"
    )


if __name__ == "__main__":
    tool = SQLTool()
    
    # Example queries
    print("Example queries:")
    print(tool.get_few_shot_context("player_stats"))
    print(tool.get_few_shot_context("player_comparison"))
