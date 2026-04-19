"""
Database documentation and example queries for NBA statistics.

This document explains the database schema, relationships, and provides
examples of typical queries you can run.
"""

DATABASE_SCHEMA = """
╔════════════════════════════════════════════════════════════════════════╗
║                   NBA STATISTICS DATABASE SCHEMA                        ║
╚════════════════════════════════════════════════════════════════════════╝

┌─── TEAMS ─────────────────────────────────────┐
│ id (PK)                  - Team ID             │
│ name (UNIQUE)            - Team name           │
│ abbreviation (UNIQUE)    - Team abbreviation   │
│ city                     - City name           │
└───────────────────────────────────────────────┘
        ↓                               ↓
    (1:N)                          (1:N)
        │                               │
        ├─→ PLAYERS ←───────────────────┤
        │   ┌─────────────────────────────────────────────┐
        │   │ id (PK)              - Player ID             │
        │   │ name (UNIQUE)        - Player name           │
        │   │ team_id (FK)         - Team reference        │
        │   │ position             - PG, SG, SF, PF, C     │
        │   │ jersey_number        - Jersey #              │
        │   │ height               - In cm                 │
        │   │ weight               - In kg                 │
        │   │ birth_date           - Birth date            │
        │   └─────────────────────────────────────────────┘
        │            ↓
        │        (1:N)
        │            │
        │            └─→ STATS
        │                ┌────────────────────────────────────────┐
        │                │ id (PK)                      - Stat ID │
        │                │ player_id (FK)   - Player reference    │
        │                │ match_id (FK)    - Match reference     │
        │                │ SCORING STATS                          │
        │                │   - points, FG%, 3P%, FT%              │
        │                │ REBOUNDING STATS                       │
        │                │   - total_rebounds, OREB, DREB         │
        │                │ ASSIST STATS                           │
        │                │   - assists, turnovers                 │
        │                │ DEFENSE STATS                          │
        │                │   - steals, blocks, fouls              │
        │                │ EFFICIENCY                             │
        │                │   - plus_minus, PER                    │
        │                └────────────────────────────────────────┘
        │                            ↑
        │                        (N:1)
        │                            │
        └────→ MATCHES ◄─────────────┘
            ┌──────────────────────────────────────┐
            │ id (PK)          - Match ID           │
            │ date             - Game date          │
            │ season           - Season year        │
            │ match_type       - regular/playoff    │
            │ home_team_id     - Home team (FK)     │
            │ away_team_id     - Away team (FK)     │
            │ home_score       - Home team score    │
            │ away_score       - Away team score    │
            └──────────────────────────────────────┘

┌─── REPORTS ────────────────────────────────────┐
│ id (PK)              - Report ID                │
│ title                - Report title             │
│ player_id (FK)       - Associated player        │
│ content              - Report text              │
│ query_used           - SQL query used           │
│ report_type          - comparison/trend/perf    │
│ generated_at         - Timestamp                │
└────────────────────────────────────────────────┘
"""

EXAMPLE_QUERIES = {
    "1. Simple Player Stats": """
    -- Get all stats for a single player
    SELECT 
        p.name,
        m.date,
        t.name AS opponent,
        s.points,
        s.assists,
        s.total_rebounds,
        s.field_goal_percentage
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN matches m ON s.match_id = m.id
    JOIN teams t ON (CASE 
        WHEN m.home_team_id = p.team_id THEN m.away_team_id
        ELSE m.home_team_id
    END) = t.id
    WHERE p.name LIKE 'Steph Curry'
    ORDER BY m.date DESC;
    """,

    "2. Season Averages": """
    -- Calculate season averages for a player
    SELECT 
        p.name,
        ROUND(AVG(s.points), 1) AS avg_points,
        ROUND(AVG(s.assists), 1) AS avg_assists,
        ROUND(AVG(s.total_rebounds), 1) AS avg_rebounds,
        ROUND(AVG(s.field_goal_percentage), 1) AS avg_fg_pct,
        ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
        COUNT(*) AS games_played
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN matches m ON s.match_id = m.id
    WHERE p.name LIKE 'Kevin Durant' AND m.season = 2024
    GROUP BY p.id, p.name;
    """,

    "3. Player Comparison": """
    -- Compare statistics of two players
    SELECT 
        p.name,
        ROUND(AVG(s.points), 1) AS avg_points,
        ROUND(AVG(s.assists), 1) AS avg_assists,
        ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
        COUNT(*) AS games
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN matches m ON s.match_id = m.id
    WHERE (p.name LIKE 'Steph Curry' OR p.name LIKE 'Kevin Durant') 
        AND m.season = 2024
    GROUP BY p.id, p.name
    ORDER BY avg_points DESC;
    """,

    "4. Top 3-Point Shooters": """
    -- Find best 3-point shooters (with minimum attempts)
    SELECT 
        p.name,
        t.name AS team,
        ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
        ROUND(AVG(s.three_pointers_made), 1) AS avg_3pm,
        ROUND(AVG(s.three_pointers_attempted), 1) AS avg_3pa,
        COUNT(*) AS games
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN teams t ON p.team_id = t.id
    JOIN matches m ON s.match_id = m.id
    WHERE m.season = 2024 AND s.three_pointers_attempted >= 3
    GROUP BY p.id, p.name, t.id, t.name
    HAVING COUNT(*) >= 10
    ORDER BY avg_3p_pct DESC
    LIMIT 20;
    """,

    "5. Home vs Away Performance": """
    -- Compare player performance at home vs away
    SELECT 
        p.name,
        'Home' AS location,
        ROUND(AVG(s.points), 1) AS avg_points,
        ROUND(AVG(s.assists), 1) AS avg_assists,
        COUNT(*) AS games
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN matches m ON s.match_id = m.id
    WHERE p.name LIKE 'LeBron James' AND m.home_team_id = p.team_id
    GROUP BY p.id, p.name
    
    UNION ALL
    
    SELECT 
        p.name,
        'Away' AS location,
        ROUND(AVG(s.points), 1) AS avg_points,
        ROUND(AVG(s.assists), 1) AS avg_assists,
        COUNT(*) AS games
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN matches m ON s.match_id = m.id
    WHERE p.name LIKE 'LeBron James' AND m.away_team_id = p.team_id
    GROUP BY p.id, p.name
    ORDER BY location;
    """,

    "6. Team Wins and Losses": """
    -- Get team record for a season
    SELECT 
        t.name AS team,
        COUNT(CASE WHEN (m.home_team_id = t.id AND m.home_score > m.away_score)
                    OR (m.away_team_id = t.id AND m.away_score > m.home_score) 
              THEN 1 END) AS wins,
        COUNT(CASE WHEN (m.home_team_id = t.id AND m.home_score < m.away_score)
                    OR (m.away_team_id = t.id AND m.away_score < m.home_score) 
              THEN 1 END) AS losses
    FROM teams t
    LEFT JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
                        AND m.season = 2024
    GROUP BY t.id, t.name
    ORDER BY wins DESC;
    """,

    "7. Scoring Matchups (Player vs Team)": """
    -- How does a player score against specific teams?
    SELECT 
        p.name,
        t.name AS opponent,
        ROUND(AVG(s.points), 1) AS avg_points,
        ROUND(AVG(s.assists), 1) AS avg_assists,
        MAX(s.points) AS max_points,
        COUNT(*) AS games
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN matches m ON s.match_id = m.id
    JOIN teams t ON (CASE 
        WHEN m.home_team_id = p.team_id THEN m.away_team_id
        ELSE m.home_team_id
    END) = t.id
    WHERE p.name LIKE 'Stephen Curry' 
        AND m.season = 2024
    GROUP BY p.id, p.name, t.id, t.name
    ORDER BY avg_points DESC;
    """,

    "8. Assists Leaders": """
    -- Top assist makers
    SELECT 
        p.name,
        t.name AS team,
        ROUND(AVG(s.assists), 1) AS avg_assists,
        ROUND(AVG(s.points), 1) AS avg_points,
        COUNT(*) AS games
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN teams t ON p.team_id = t.id
    JOIN matches m ON s.match_id = m.id
    WHERE m.season = 2024
    GROUP BY p.id, p.name, t.id, t.name
    HAVING COUNT(*) >= 10
    ORDER BY avg_assists DESC
    LIMIT 15;
    """,

    "9. Multi-Criteria Aggregation": """
    -- Complex stats: Efficiency by season and category
    SELECT 
        m.season,
        p.position,
        COUNT(DISTINCT p.id) AS num_players,
        ROUND(AVG(s.points), 1) AS avg_points,
        ROUND(AVG(s.assists), 1) AS avg_assists,
        ROUND(AVG(s.total_rebounds), 1) AS avg_rebounds,
        ROUND(AVG(s.field_goal_percentage), 1) AS avg_fg_pct
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN matches m ON s.match_id = m.id
    WHERE m.season = 2024 AND p.position IS NOT NULL
    GROUP BY m.season, p.position
    ORDER BY m.season, avg_points DESC;
    """,

    "10. Best Performing Months": """
    -- How did a player perform each month?
    SELECT 
        p.name,
        EXTRACT(YEAR FROM m.date) AS year,
        EXTRACT(MONTH FROM m.date) AS month,
        ROUND(AVG(s.points), 1) AS avg_points,
        ROUND(AVG(s.assists), 1) AS avg_assists,
        COUNT(*) AS games
    FROM stats s
    JOIN players p ON s.player_id = p.id
    JOIN matches m ON s.match_id = m.id
    WHERE p.name LIKE 'Michael Jordan'
    GROUP BY p.id, p.name, year, month
    ORDER BY year DESC, month DESC;
    """
}

DATABASE_TIPS = """
╔════════════════════════════════════════════════════════════════════════╗
║                       DATABASE USAGE TIPS                              ║
╚════════════════════════════════════════════════════════════════════════╝

1. DATA INSERTION:
   ┌─ Excel Loading
   │  Run: python load_excel_to_db.py
   │  This will:
   │  - Read all Excel files from ./inputs/
   │  - Validate data with Pydantic
   │  - Insert into SQLite database
   └─

2. COMMON QUERY PATTERNS:
   ┌─ Player Lookups
   │  Use: p.name LIKE '%partial_name%' (case-insensitive)
   │  Example: LIKE '%Curry%'
   ├─ Team Lookups
   │  Use: t.name ILIKE 'Team Name'
   ├─ Date Ranges
   │  Use: m.date BETWEEN '2024-01-01' AND '2024-12-31'
   └─

3. OPTIMIZATION TIPS:
   ┌─ Index creation (for large datasets)
   │  CREATE INDEX idx_player_name ON players(name);
   │  CREATE INDEX idx_match_date ON matches(date);
   │  CREATE INDEX idx_stat_player ON stats(player_id);
   ├─ Use LIMIT to prevent huge result sets
   ├─ Filter by season early to reduce joins
   └─

4. AGGREGATION FUNCTIONS:
   ┌─ AVG()  - Average value
   ├─ SUM()  - Total sum
   ├─ COUNT() - Count rows
   ├─ MAX()  - Maximum value
   ├─ MIN()  - Minimum value
   └─

5. DEBUGGING:
   ┌─ Check database file: ls -la nba_stats.db
   ├─ SQLite shell: sqlite3 nba_stats.db
   ├─ List tables: .tables
   ├─ Schema: .schema table_name
   └─
"""

if __name__ == "__main__":
    print(DATABASE_SCHEMA)
    print("\n" + "=" * 80 + "\n")
    
    for title, query in EXAMPLE_QUERIES.items():
        print(f"\\n{title}")
        print("-" * 80)
        print(query)
        print()
    
    print(DATABASE_TIPS)
