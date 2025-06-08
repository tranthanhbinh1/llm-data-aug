from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class PromptCandidate(BaseModel):
    """Represents a candidate prompt in the genetic algorithm."""

    prompt: str = Field(..., description="The prompt text")
    fitness: Optional[float] = Field(None, description="Fitness score (0.0 to 1.0)")
    reflection: Optional[str] = Field(
        None, description="Evaluation reflection/reasoning"
    )
    generation: int = Field(0, description="Generation number in the genetic algorithm")
    parent_ids: List[str] = Field(
        default_factory=list, description="IDs of parent prompts"
    )

    @field_validator("fitness")
    def validate_fitness(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Fitness must be between 0.0 and 1.0")
        return v

    @field_validator("generation")
    def validate_generation(cls, v):
        if v < 0:
            raise ValueError("Generation must be non-negative")
        return v

    class Config:
        json_encoders = {
            # Custom encoders if needed
        }


class OptimizationResult(BaseModel):
    """Result of a prompt optimization run."""

    best_prompt: str = Field(..., description="The best prompt found")
    best_score: float = Field(
        ..., ge=0.0, le=1.0, description="Best fitness score achieved"
    )
    initial_prompt: str = Field(..., description="The starting prompt")
    improvement_request: str = Field(..., description="What was requested to improve")
    total_iterations: int = Field(..., ge=0, description="Total iterations run")
    total_candidates_evaluated: int = Field(
        ..., ge=0, description="Total candidates evaluated"
    )
    execution_time_seconds: float = Field(
        ..., ge=0.0, description="Total execution time"
    )
    convergence_iteration: Optional[int] = Field(
        None, description="Iteration where convergence was reached"
    )
    all_candidates: List[PromptCandidate] = Field(
        default_factory=list, description="All candidates evaluated"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the optimization was run"
    )

    @field_validator("convergence_iteration")
    def validate_convergence_iteration(cls, v, values):
        if (
            v is not None
            and "total_iterations" in values
            and v > values["total_iterations"]
        ):
            raise ValueError(
                "Convergence iteration cannot be greater than total iterations"
            )
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class OptimizationConfig(BaseModel):
    """Configuration for prompt optimization."""

    population_size: int = Field(
        5, ge=1, description="Number of candidates per generation"
    )
    num_iterations: int = Field(5, ge=1, description="Maximum number of generations")
    num_elites: int = Field(2, ge=0, description="Number of top candidates to preserve")
    threshold: float = Field(
        1.0, ge=0.0, le=1.0, description="Fitness threshold for early stopping"
    )
    tournament_size: int = Field(3, ge=1, description="Tournament selection size")
    num_evaluation_samples: int = Field(
        3, ge=1, description="Self-consistency samples for LLM evaluation"
    )
    model: str = Field("gemini-2.0-flash", description="LLM model for operations")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_retries: int = Field(3, ge=0, description="Retry attempts for failed API calls")

    @field_validator("num_elites")
    def validate_num_elites(cls, v, values):
        if "population_size" in values and v >= values["population_size"]:
            raise ValueError("Number of elites must be less than population size")
        return v

    @field_validator("tournament_size")
    def validate_tournament_size(cls, v, values):
        if "population_size" in values and v > values["population_size"]:
            raise ValueError("Tournament size cannot be greater than population size")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()

    class Config:
        validate_assignment = True  # Validate on assignment
