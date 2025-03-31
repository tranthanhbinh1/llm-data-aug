from pydantic import BaseModel, Field
from enum import StrEnum


class UserReview(BaseModel):
    review: str = Field(description="The user reviews")
    sentiment: int = Field(
        description="The sentiments of the reviews, 0 for negative, 1 for positive, 2 for neutral",
    )


class UserReviews(BaseModel):
    reviews: list[UserReview] = Field(
        description="The user reviews", min_length=15, max_length=15
    )


class AugmentedUserReview(BaseModel):
    review: str = Field(
        description="The augmented user review, it should be semantically similar to the original review"
    )
    sentiment: int = Field(
        description="The sentiments of the reviews, 0 for negative, 1 for positive, 2 for neutral",
    )


class AugmentedUserReviews(BaseModel):
    reviews: list[AugmentedUserReview] = Field(
        description="The augmented user reviews", min_length=6, max_length=6
    )


class SentimentPrompt(StrEnum):
    NEGATIVE = "Hãy tạo ra {batch_size} bình luận tương tự nói về trải nghiệm: Trong vai là một khách hàng vừa trải qua một một trải nghiệm tồi tệ, ko hài lòng về chất lượng dịch vụ của McDonald."
    POSITIVE = "Hãy tạo ra {batch_size} bình luận tương tự nói về trải nghiệm: Trong vai là một khách hàng vừa trải qua một một trải nghiệm tốt, hài lòng về chất lượng dịch vụ của McDonald."
    NEUTRAL = "Hãy tạo ra {batch_size} bình luận tương tự nói về trải nghiệm: Trong vai là một khách hàng vừa trải qua một một trải nghiệm trung bình, không tốt cũng không xấu về chất lượng dịch vụ của McDonald."
    AUG_GPT_PROMPT = "Từ đánh giá/mẫu gốc được cung cấp, hãy tạo ra {batch_size} phiên bản diễn đạt lại mà vẫn giữ nguyên ý nghĩa gốc."
