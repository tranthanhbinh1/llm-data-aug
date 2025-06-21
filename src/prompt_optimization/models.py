from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, model_validator, ConfigDict


class PromptCandidate(BaseModel):
    """Represents a candidate prompt in the genetic algorithm."""

    prompt: str = Field(..., description="The prompt text")
    fitness: Optional[float] = Field(
        None, description="Fitness score (0.0 to 1.0)", ge=0.0, le=1.0
    )
    reflection: Optional[str] = Field(
        None, description="Evaluation reflection/reasoning"
    )
    generation: int = Field(
        0, description="Generation number in the genetic algorithm", ge=0
    )
    parent_ids: List[str] = Field(
        default_factory=list, description="IDs of parent prompts"
    )


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

    @model_validator(mode="after")
    def validate_convergence_iteration(self):
        if (
            self.convergence_iteration
            and self.total_iterations
            and self.convergence_iteration > self.total_iterations
        ):
            raise ValueError(
                "Convergence iteration cannot be greater than total iterations"
            )
        return self

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


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

    @model_validator(mode="after")
    def validate_num_elites(self):
        if self.population_size and self.num_elites >= self.population_size:
            raise ValueError("Number of elites must be less than population size")
        return self

    @model_validator(mode="after")
    def validate_tournament_size(self):
        if self.population_size and self.tournament_size > self.population_size:
            raise ValueError("Tournament size cannot be greater than population size")
        return self

    model_config = ConfigDict(validate_assignment=True)
