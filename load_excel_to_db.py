"""
Pipeline for ingesting Excel data into the database.

Handles:
- Excel file reading
- Data validation via Pydantic
- Database insertion with conflict handling
- Error logging and reporting
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from db_schema import Player, Team, Match, Stat, Report, Base
from data_validators import PlayerIn, TeamIn, MatchIn, StatIn
from utils.database import SessionLocal, engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ExcelIngestionPipeline:
    """Pipeline for ingesting NBA data from Excel files."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize the pipeline with a database session."""
        self.db = db or SessionLocal()
        self.stats_cache = []  # Cache stats for batch insertion
        self.errors = []

    def init_database(self):
        """Initialize database tables."""
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database initialized")

    def load_teams_from_excel(self, file_path: str) -> List[Team]:
        """Load teams from Excel file."""
        logger.info(f"Loading teams from {file_path}...")
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            df = df.fillna("")

            teams = []
            for _, row in df.iterrows():
                # Flexible column mapping
                name = (
                    row.get("Team") or row.get("Équipe") or
                    row.get("team_name") or row.get("name") or ""
                ).strip()

                if not name:
                    continue

                abbr = (
                    row.get("Abbreviation") or row.get("Abbr") or
                    row.get("abbreviation") or ""
                ).strip()[:3]

                try:
                    team_data = TeamIn(
                        name=name,
                        abbreviation=abbr or name[:3],
                        city=row.get("City") or row.get("Ville") or ""
                    )
                    team = self.db.query(Team).filter_by(name=name).first()
                    if not team:
                        team = Team(**team_data.model_dump())
                        self.db.add(team)
                        teams.append(team)
                except Exception as e:
                    self.errors.append(f"Team error: {e}")
                    logger.error(f"Team validation error: {e}")

            self.db.commit()
            logger.info(f"✓ Loaded {len(teams)} teams")
            return teams

        except Exception as e:
            logger.error(f"Error loading teams: {e}")
            self.errors.append(str(e))
            return []

    def load_players_from_excel(self, file_path: str) -> List[Player]:
        """Load players from Excel file."""
        logger.info(f"Loading players from {file_path}...")
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            df = df.fillna("")

            players = []
            for _, row in df.iterrows():
                name = (
                    row.get("Player") or row.get("Joueur") or
                    row.get("name") or ""
                ).strip()

                if not name or name in ["", "Unnamed: 0"]:
                    continue

                team_name = row.get("Team") or row.get("Équipe") or row.get("team") or ""
                team = None
                if team_name:
                    team = self.db.query(Team).filter_by(name=team_name).first()

                try:
                    player_data = PlayerIn(
                        name=name,
                        team_name=team_name,
                        position=(row.get("Position") or "").strip().upper(),
                        jersey_number=int(row.get("Jersey", 0)) if row.get("Jersey") else None,
                        height=float(row.get("Height", 0)) if row.get("Height") else None,
                        weight=float(row.get("Weight", 0)) if row.get("Weight") else None,
                    )

                    player = self.db.query(Player).filter_by(name=name).first()
                    if not player:
                        player = Player(
                            **{k: v for k, v in player_data.model_dump().items() if k != "team_name"},
                            team_id=team.id if team else None
                        )
                        self.db.add(player)
                        players.append(player)

                except Exception as e:
                    self.errors.append(f"Player {name} error: {e}")
                    logger.error(f"Player validation error for {name}: {e}")

            self.db.commit()
            logger.info(f"✓ Loaded {len(players)} players")
            return players

        except Exception as e:
            logger.error(f"Error loading players: {e}")
            self.errors.append(str(e))
            return []

    def load_matches_from_excel(self, file_path: str) -> List[Match]:
        """Load matches from Excel file."""
        logger.info(f"Loading matches from {file_path}...")
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            df = df.fillna("")

            matches = []
            for _, row in df.iterrows():
                try:
                    home_team_name = row.get("Home Team") or row.get("Équipe domicile") or ""
                    away_team_name = row.get("Away Team") or row.get("Équipe extérieure") or ""

                    if not home_team_name or not away_team_name:
                        continue

                    home_team = self.db.query(Team).filter_by(name=home_team_name).first()
                    away_team = self.db.query(Team).filter_by(name=away_team_name).first()

                    if not home_team or not away_team:
                        self.errors.append(f"Teams not found: {home_team_name} vs {away_team_name}")
                        continue

                    match_data = MatchIn(
                        date=pd.to_datetime(row.get("Date")).date(),
                        season=int(row.get("Season", 2024)),
                        match_type=row.get("Type", "regular"),
                        home_team_name=home_team_name,
                        away_team_name=away_team_name,
                        home_score=int(row.get("Home Score", 0)) if row.get("Home Score") else None,
                        away_score=int(row.get("Away Score", 0)) if row.get("Away Score") else None
                    )

                    match = Match(
                        date=match_data.date,
                        season=match_data.season,
                        match_type=match_data.match_type,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        home_score=match_data.home_score,
                        away_score=match_data.away_score
                    )
                    self.db.add(match)
                    matches.append(match)

                except Exception as e:
                    self.errors.append(f"Match error: {e}")
                    logger.error(f"Match validation error: {e}")

            self.db.commit()
            logger.info(f"✓ Loaded {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"Error loading matches: {e}")
            self.errors.append(str(e))
            return []

    def load_stats_from_excel(self, file_path: str) -> List[Stat]:
        """Load player statistics from Excel file."""
        logger.info(f"Loading stats from {file_path}...")
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            df = df.fillna(0)

            stats_list = []
            for _, row in df.iterrows():
                try:
                    player_name = (row.get("Player") or row.get("Joueur") or "").strip()
                    if not player_name:
                        continue

                    player = self.db.query(Player).filter_by(name=player_name).first()
                    if not player:
                        logger.warning(f"Player not found: {player_name}")
                        continue

                    # Try to find match by date and teams
                    match_date = pd.to_datetime(row.get("Date")).date() if row.get("Date") else None
                    if not match_date:
                        continue

                    match = self.db.query(Match).filter_by(date=match_date).first()
                    if not match:
                        logger.warning(f"Match not found for {match_date}")
                        continue

                    stat = Stat(
                        player_id=player.id,
                        match_id=match.id,
                        points=int(row.get("Points", 0)) or 0,
                        field_goals_made=int(row.get("FGM", 0)) or 0,
                        field_goals_attempted=int(row.get("FGA", 0)) or 0,
                        field_goal_percentage=float(row.get("FG%", 0)) or None,
                        three_pointers_made=int(row.get("3PM", 0)) or 0,
                        three_pointers_attempted=int(row.get("3PA", 0)) or 0,
                        three_point_percentage=float(row.get("3P%", 0)) or None,
                        free_throws_made=int(row.get("FTM", 0)) or 0,
                        free_throws_attempted=int(row.get("FTA", 0)) or 0,
                        free_throw_percentage=float(row.get("FT%", 0)) or None,
                        total_rebounds=int(row.get("REB", 0)) or 0,
                        offensive_rebounds=int(row.get("OREB", 0)) or 0,
                        defensive_rebounds=int(row.get("DREB", 0)) or 0,
                        assists=int(row.get("AST", 0)) or 0,
                        turnovers=int(row.get("TOV", 0)) or 0,
                        steals=int(row.get("STL", 0)) or 0,
                        blocks=int(row.get("BLK", 0)) or 0,
                        fouls_personal=int(row.get("PF", 0)) or 0,
                        plus_minus=int(row.get("+/-", 0)) if row.get("+/-") else None,
                        efficiency_rating=float(row.get("PER", 0)) or None,
                        minutes_played=float(row.get("MIN", 0)) or None
                    )
                    self.db.add(stat)
                    stats_list.append(stat)

                except Exception as e:
                    self.errors.append(f"Stat error: {e}")
                    logger.error(f"Stat validation error: {e}")

            self.db.commit()
            logger.info(f"✓ Loaded {len(stats_list)} stats")
            return stats_list

        except Exception as e:
            logger.error(f"Error loading stats: {e}")
            self.errors.append(str(e))
            return []

    def ingest_all(self, inputs_dir: str = "inputs") -> Dict:
        """Ingest all Excel files from the inputs directory."""
        logger.info(f"Starting ingestion from {inputs_dir}...")
        inputs_path = Path(inputs_dir)

        results = {
            "teams": [],
            "players": [],
            "matches": [],
            "stats": [],
            "errors": []
        }

        # Load in order
        for file_path in inputs_path.glob("*.xlsx"):
            file_name = file_path.name.lower()
            logger.info(f"\nProcessing {file_name}...")

            if "team" in file_name or "equipe" in file_name:
                results["teams"].extend(self.load_teams_from_excel(str(file_path)))
            elif "player" in file_name or "joueur" in file_name:
                results["players"].extend(self.load_players_from_excel(str(file_path)))
            elif "match" in file_name or "game" in file_name:
                results["matches"].extend(self.load_matches_from_excel(str(file_path)))
            else:
                # Try to infer from content
                results["players"].extend(self.load_players_from_excel(str(file_path)))
                results["stats"].extend(self.load_stats_from_excel(str(file_path)))

        results["errors"] = self.errors
        logger.info("\n✓ Ingestion complete")
        self._print_summary(results)
        return results

    def _print_summary(self, results: Dict):
        """Print ingestion summary."""
        print("\n" + "=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)
        print(f"Teams: {len(results['teams'])}")
        print(f"Players: {len(results['players'])}")
        print(f"Matches: {len(results['matches'])}")
        print(f"Stats: {len(results['stats'])}")
        print(f"Errors: {len(results['errors'])}")
        if results['errors']:
            print("\nErrors:")
            for err in results['errors'][:10]:
                print(f"  - {err}")
        print("=" * 50)


if __name__ == "__main__":
    pipeline = ExcelIngestionPipeline()
    pipeline.init_database()
    results = pipeline.ingest_all()
