from pydantic import BaseModel, Field, field_validator
from enum import StrEnum

from src.enums import Sentiment


class AugmentedSentence(BaseModel):
    sentence: str = Field(
        description="The augmented sentence, it should be semantically similar to the original sentence"
    )
    sentiment: Sentiment = Field(
        description="The sentiment of the generated sentence, it should be the same as the original sentence. It can be positive, negative or neutral.",
    )


class AugmentedSentencesBatch(BaseModel):
    sentences: list[AugmentedSentence] = Field(
        description="The augmented sentences", min_length=6, max_length=6
    )


class SentimentPrompt(StrEnum):
    # NEGATIVE = "Hãy tạo ra {batch_size} bình luận tương tự nói về trải nghiệm: Trong vai là một khách hàng vừa trải qua một một trải nghiệm tồi tệ, ko hài lòng về chất lượng dịch vụ của McDonald."
    # POSITIVE = "Hãy tạo ra {batch_size} bình luận tương tự nói về trải nghiệm: Trong vai là một khách hàng vừa trải qua một một trải nghiệm tốt, hài lòng về chất lượng dịch vụ của McDonald."
    # NEUTRAL = "Hãy tạo ra {batch_size} bình luận tương tự nói về trải nghiệm: Trong vai là một khách hàng vừa trải qua một một trải nghiệm trung bình, không tốt cũng không xấu về chất lượng dịch vụ của McDonald."
    AUG_GPT_PROMPT = "Từ đánh giá/mẫu gốc được cung cấp, hãy tạo ra {batch_size} phiên bản diễn đạt lại mà vẫn giữ nguyên ý nghĩa gốc."
