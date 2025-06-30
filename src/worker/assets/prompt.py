"""
Prompt optimization asset for Dagster pipeline.
"""

import dagster as dg
import asyncio
from src.enums import Sentiment
from src.worker.helpers import PromptHelper
from src.worker.resource import LLMResource


class PromptAssetConfig(dg.Config):
    initial_prompt: str = """
    Bạn là một trợ lý AI hữu ích. Nhiệm vụ của bạn là tạo ra các đánh giá một địa điểm ăn uống (McDonald's) bằng tiếng Việt chân thực và đa dạng.
    """
    improvement_request: str = """
    Prompt hiện tại đã tạo ra được các đánh giá. Để nâng cao chất lượng dữ liệu và giúp việc chấm điểm (scoring) hiệu quả hơn, hãy tối ưu hóa prompt để các đánh giá được tạo ra đáp ứng các tiêu chí sau:
    1. Tăng tính cụ thể và chi tiết (Increase Specificity and Detail):
    - Đánh giá cần đề cập đến các món ăn, đồ uống, hoặc combo cụ thể của McDonald's tại Việt Nam (ví dụ: Big Mac, khoai tây chiên, McFlurry Oreo, gà rán).
    - Phản ánh về nhiều khía cạnh khác nhau của trải nghiệm, không chỉ đồ ăn, ví dụ: thái độ nhân viên, tốc độ phục vụ, sự sạch sẽ của quán, không gian (chật/rộng, ồn ào/yên tĩnh), giá cả.
    2. Mở rộng sự đa dạng về cảm xúc và góc nhìn (Expand Diversity of Sentiment and Perspective):
    - Tạo ra một tỷ lệ rõ ràng hơn giữa các loại đánh giá: tích cực, tiêu cực, và trung lập.
    - Bắt buộc phải có những đánh giá mang cảm xúc pha trộn (ví dụ: "Gà rán ngon nhưng khoai tây chiên lại bị ỉu" hoặc "Phục vụ nhanh nhưng giá hơi cao so với sinh viên").
    - Mô phỏng các góc nhìn từ những nhóm khách hàng điển hình khác nhau: sinh viên (quan tâm giá cả, không gian học bài), gia đình có con nhỏ (quan tâm khu vui chơi, thực đơn cho trẻ em), nhân viên văn phòng (quan tâm tốc độ ăn trưa, combo tiện lợi).
    3. Nâng cao tính chân thực trong ngôn ngữ (Enhance Authenticity of Language):
    - Sử dụng ngôn ngữ tự nhiên, văn nói hàng ngày, có thể bao gồm cả những từ cảm thán (vd: "ôi", "chà", "trời ơi") hoặc từ lóng phổ biến.
    - Tránh sử dụng những câu văn quá trang trọng, máy móc hoặc lặp lại cùng một cấu trúc. Các đánh giá phải có độ dài ngắn khác nhau.
    """
    sentiment: Sentiment = Sentiment.NEUTRAL


@dg.asset(
    group_name="generation",
    description="Optimized prompt for synthetic data generation",
    metadata={
        "asset_type": "prompt",
        "output_format": "text",
    },
)
def prompt_asset(
    context: dg.AssetExecutionContext,
    llm: LLMResource,
    config: PromptAssetConfig,
) -> str:
    """Generate optimized prompt using genetic algorithm optimization."""
    # Get configuration from asset context
    initial_prompt = config.initial_prompt
    improvement_request = config.improvement_request
    sentiment = config.sentiment

    # Generate cache key and check for existing prompt
    cache_key = PromptHelper.get_cache_key(
        initial_prompt, improvement_request, sentiment
    )
    output_path = PromptHelper.get_output_path(cache_key)

    if PromptHelper.check_cached_prompt(output_path):
        prompt = PromptHelper.load_cached_prompt(output_path)
        context.log.info(f"Using cached prompt from {output_path}")
        result_metadata = {"cached": True, "cache_key": cache_key}
    else:
        # Run optimization asynchronously
        result = asyncio.run(
            PromptHelper.optimize_prompt(
                api_key=llm.api_key,
                initial_prompt=initial_prompt,
                improvement_request=improvement_request,
                # config=config,
            )
        )
        prompt = result["prompt"]
        PromptHelper.save_prompt(prompt, output_path)

        result_metadata = {
            "cached": False,
            "cache_key": cache_key,
            "score": result["score"],
            "iterations": result["iterations"],
            "candidates_evaluated": result["candidates_evaluated"],
            "execution_time": result["execution_time"],
        }

    # Add metadata to context
    context.add_output_metadata(metadata=result_metadata)

    context.log.info(f"Optimized prompt: {prompt}")

    context.log_event(
        dg.AssetObservation(
            asset_key=dg.AssetKey("prompt_asset"),
            description="Prompt optimization result",
            metadata=result_metadata,
        )
    )

    return prompt
