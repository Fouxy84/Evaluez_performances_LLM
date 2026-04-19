"""
Data validation schemas using Pydantic.

Used for validating Excel data before insertion into the database.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date


class TeamIn(BaseModel):
    """Validation model for Team data."""
    name: str
    abbreviation: str
    city: Optional[str] = None

    class Config:
        from_attributes = True


class PlayerIn(BaseModel):
    """Validation model for Player data."""
    name: str
    team_name: Optional[str] = None
    position: Optional[str] = None
    jersey_number: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    birth_date: Optional[date] = None

    @validator('position')
    def validate_position(cls, v):
        if v and v.upper() not in ['PG', 'SG', 'SF', 'PF', 'C']:
            raise ValueError(f'Position must be one of: PG, SG, SF, PF, C')
        return v.upper() if v else None

    class Config:
        from_attributes = True


class MatchIn(BaseModel):
    """Validation model for Match data."""
    date: date
    season: Optional[int] = None
    match_type: Optional[str] = "regular"
    home_team_name: str
    away_team_name: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    @validator('match_type')
    def validate_match_type(cls, v):
        if v and v.lower() not in ['regular', 'playoff', 'finals']:
            raise ValueError('match_type must be: regular, playoff, or finals')
        return v.lower() if v else 'regular'

    class Config:
        from_attributes = True


class StatIn(BaseModel):
    """Validation model for Stat data."""
    player_name: str
    match_date: date
    home_team_name: str
    away_team_name: str
    
    points: int = 0
    field_goals_made: int = 0
    field_goals_attempted: int = 0
    field_goal_percentage: Optional[float] = None
    three_pointers_made: int = 0
    three_pointers_attempted: int = 0
    three_point_percentage: Optional[float] = None
    free_throws_made: int = 0
    free_throws_attempted: int = 0
    free_throw_percentage: Optional[float] = None
    
    total_rebounds: int = 0
    offensive_rebounds: int = 0
    defensive_rebounds: int = 0
    
    assists: int = 0
    turnovers: int = 0
    steals: int = 0
    blocks: int = 0
    fouls_personal: int = 0
    
    plus_minus: Optional[int] = None
    efficiency_rating: Optional[float] = None
    minutes_played: Optional[float] = None

    @validator('points', 'assists', 'rebounds', pre=True)
    def coerce_int(cls, v):
        if v is None:
            return 0
        return int(v) if v else 0

    class Config:
        from_attributes = True


class ReportIn(BaseModel):
    """Validation model for Report data."""
    title: str
    player_name: Optional[str] = None
    content: str
    report_type: str = "analysis"

    class Config:
        from_attributes = True
