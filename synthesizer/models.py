from pydantic import BaseModel, Field


class UserReview(BaseModel):
    reviews: list[str] = Field(description="The user reviews", alias="Review")
    sentiments: list[int] = Field(
        description="The sentiments of the reviews, 0 for negative, 1 for positive, 2 for neutral",
        alias="Sentiment",
    )
    n: int = Field(
        description="The number of reviews to generate",
        default=10,
    )
