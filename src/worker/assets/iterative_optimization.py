"""
Iterative prompt optimization asset that integrates genetic algorithm with similarity evaluation.
"""

import dagster as dg
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from src.enums import Sentiment
from src.worker.helpers import DataHelper, ScoreHelper
from src.worker.resource import LLMResource, SynthesizerResource
from src.prompt_optimization import PromptOptimizer, OptimizationConfig
from src.constants import PROJECT_ROOT


class IterativeOptimizationConfig(dg.Config):
    """Configuration for iterative optimization with similarity feedback."""

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
    max_optimization_rounds: int = 10
    similarity_threshold: float = 0.8
    population_size: int = 3
    num_iterations: int = 2
    num_elites: int = 1


@dg.asset(
    group_name="optimization",
    description="Iterative prompt optimization with similarity evaluation feedback",
    metadata={
        "asset_type": "iterative_optimization",
        "output_format": "json",
    },
)
def iterative_optimization_asset(
    context: dg.AssetExecutionContext,
    llm: LLMResource,
    synthesizer: SynthesizerResource,
    config: IterativeOptimizationConfig,
) -> str:
    """
    Perform iterative prompt optimization with similarity evaluation feedback.

    This asset combines genetic algorithm optimization with real similarity evaluation
    to create a feedback loop that improves prompts based on actual data quality.
    """

    context.log.info(
        "🚀 Starting iterative prompt optimization with similarity feedback"
    )
    context.log.info(
        f"📊 Configuration: max_rounds={config.max_optimization_rounds}, "
        f"threshold={config.similarity_threshold}, "
        f"population_size={config.population_size}, "
        f"sentiment={config.sentiment.value}"
    )

    # Generate cache key for this optimization run
    import hashlib

    cache_content = f"{config.initial_prompt}|{config.improvement_request}|{config.sentiment}|{config.max_optimization_rounds}|{config.similarity_threshold}"
    cache_key = hashlib.sha256(cache_content.encode()).hexdigest()[:16]

    context.log.info(f"🔑 Generated cache key: {cache_key}")

    # Setup output paths
    output_dir = Path(PROJECT_ROOT) / "graphs" / "iterative_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"optimization_result_{cache_key}.json"

    if output_path.exists():
        context.log.info(f"💾 Using cached optimization result from {output_path}")
        with open(output_path, "r") as f:
            result = json.load(f)
        context.log.info(
            f"📈 Cached result: {result['total_rounds']} rounds, "
            f"final score: {result['final_similarity_score']:.4f}, "
            f"converged: {result['converged']}"
        )
        result_metadata = {"cached": True, "cache_key": cache_key}
        result_metadata.update(result)
    else:
        context.log.info("🔄 Running new iterative optimization...")
        # Run iterative optimization
        result = asyncio.run(
            _run_iterative_optimization(
                context=context,
                llm=llm,
                synthesizer=synthesizer,
                config=config,
            )
        )

        # Save result
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        context.log.info(f"💾 Saved optimization result to {output_path}")

        result_metadata = {"cached": False, "cache_key": cache_key}
        result_metadata.update(result)

    # Add metadata to context
    context.add_output_metadata(metadata=result_metadata)

    context.log.info("✅ Iterative optimization completed successfully")

    return str(output_path)


async def _run_iterative_optimization(
    context: dg.AssetExecutionContext,
    llm: LLMResource,
    synthesizer: SynthesizerResource,
    config: IterativeOptimizationConfig,
) -> Dict[str, Any]:
    """
    Run the iterative optimization process with similarity feedback.
    """
    context.log.info("🔧 Initializing iterative optimization process")

    # Initialize optimization configuration
    optimization_config = OptimizationConfig(
        population_size=config.population_size,
        num_iterations=config.num_iterations,
        num_elites=config.num_elites,
        threshold=config.similarity_threshold,
        tournament_size=min(3, config.population_size),
        num_evaluation_samples=2,
        model="gemini-2.0-flash",
        max_retries=3,
    )

    context.log.info(
        f"⚙️ Genetic Algorithm Config: population={optimization_config.population_size}, "
        f"iterations_per_round={optimization_config.num_iterations}, "
        f"elites={optimization_config.num_elites}"
    )

    # Create custom evaluator that uses similarity evaluation
    def create_similarity_evaluator():
        async def similarity_evaluator(candidate, initial_prompt, improvement_request):
            """Custom evaluator that uses actual similarity evaluation."""
            try:
                context.log.info(
                    f"🔍 Evaluating candidate prompt (length: {len(candidate.prompt)} chars)"
                )

                # Generate synthetic data using the candidate prompt
                data_path = DataHelper.generate_synthetic_data(
                    auggpt_runner=synthesizer.get_synthesizer_instance(),
                    prompt=candidate.prompt,
                    sentiment=config.sentiment,
                )

                context.log.info(f"📝 Generated synthetic data at: {data_path}")

                # Evaluate similarity
                score_data = ScoreHelper.evaluate_similarity(
                    data_path=data_path,
                    prompt=candidate.prompt,
                )

                # Set fitness based on similarity score
                candidate.fitness = score_data["similarity_score"]
                candidate.reflection = (
                    f"Similarity score: {score_data['similarity_score']:.4f}"
                )

                context.log.info(
                    f"📊 Candidate evaluation complete - Similarity score: {candidate.fitness:.4f}"
                )

                return candidate

            except Exception as e:
                context.log.error(f"❌ Similarity evaluation failed: {e}")
                candidate.fitness = 0.0
                candidate.reflection = f"Evaluation failed: {str(e)}"
                return candidate

        return similarity_evaluator

    # Run optimization with similarity feedback
    optimizer = PromptOptimizer(api_key=llm.api_key, config=optimization_config)

    optimization_rounds = []
    current_prompt = config.initial_prompt
    best_similarity_score = 0.0

    context.log.info(
        f"🎯 Starting optimization with initial prompt (length: {len(current_prompt)} chars)"
    )
    context.log.info(f"📝 Initial prompt preview: {current_prompt[:200]}...")

    for round_num in range(config.max_optimization_rounds):
        context.log.info(
            f"🔄 Starting optimization round {round_num + 1}/{config.max_optimization_rounds}"
        )
        context.log.info(
            f"📈 Current best similarity score: {best_similarity_score:.4f}"
        )

        # Run genetic algorithm optimization with similarity evaluator
        context.log.info("🧬 Running genetic algorithm with similarity evaluator...")
        result = await optimizer.optimize(
            initial_prompt=current_prompt,
            improvement_request=config.improvement_request,
            custom_evaluator=create_similarity_evaluator(),
        )

        round_data = {
            "round": round_num + 1,
            "initial_prompt": current_prompt,
            "best_prompt": result.best_prompt,
            "best_score": result.best_score,
            "iterations": result.total_iterations,
            "candidates_evaluated": result.total_candidates_evaluated,
            "execution_time": result.execution_time_seconds,
        }

        optimization_rounds.append(round_data)

        context.log.info(f"✅ Round {round_num + 1} completed:")
        context.log.info(f"   📊 Best similarity score: {result.best_score:.4f}")
        context.log.info(f"   🔄 GA iterations: {result.total_iterations}")
        context.log.info(
            f"   🧪 Candidates evaluated: {result.total_candidates_evaluated}"
        )
        context.log.info(f"   ⏱️ Execution time: {result.execution_time_seconds:.2f}s")

        # Check if we've reached the similarity threshold
        if result.best_score >= config.similarity_threshold:
            context.log.info(
                f"🎯 SUCCESS! Similarity threshold {config.similarity_threshold} reached!"
            )
            context.log.info(
                f"🏆 Final score: {result.best_score:.4f} >= {config.similarity_threshold}"
            )
            break

        # Check if we're making progress
        improvement = result.best_score - best_similarity_score
        if result.best_score > best_similarity_score:
            best_similarity_score = result.best_score
            current_prompt = result.best_prompt  # Use best prompt for next round
            context.log.info(f"📈 IMPROVEMENT! Score increased by {improvement:.4f}")
            context.log.info(f"🔄 Using improved prompt for next round")
            context.log.info(f"📝 New prompt preview: {result.best_prompt[:200]}...")
        else:
            context.log.info(
                f"📊 No improvement this round (score: {result.best_score:.4f} vs best: {best_similarity_score:.4f})"
            )
            context.log.info("🔄 Continuing with current best prompt")

    # Compile final result
    final_result = {
        "final_prompt": optimization_rounds[-1]["best_prompt"],
        "final_similarity_score": optimization_rounds[-1]["best_score"],
        "total_rounds": len(optimization_rounds),
        "converged": optimization_rounds[-1]["best_score"]
        >= config.similarity_threshold,
        "optimization_rounds": optimization_rounds,
        "config": {
            "max_rounds": config.max_optimization_rounds,
            "similarity_threshold": config.similarity_threshold,
            "sentiment": config.sentiment.value,
        },
    }

    # Final summary
    context.log.info("🏁 ITERATIVE OPTIMIZATION COMPLETE!")
    context.log.info(f"📊 Final Results:")
    context.log.info(f"   🔄 Total rounds: {len(optimization_rounds)}")
    context.log.info(
        f"   📈 Final similarity score: {final_result['final_similarity_score']:.4f}"
    )
    context.log.info(f"   🎯 Converged: {final_result['converged']}")
    context.log.info(
        f"   📝 Final prompt length: {len(final_result['final_prompt'])} chars"
    )

    if final_result["converged"]:
        context.log.info(
            f"🎉 SUCCESS! Reached similarity threshold of {config.similarity_threshold}"
        )
    else:
        remaining_improvement = (
            config.similarity_threshold - final_result["final_similarity_score"]
        )
        context.log.info(
            f"⚠️ Did not reach threshold. Need {remaining_improvement:.4f} more similarity score."
        )

    # Log improvement over rounds
    if len(optimization_rounds) > 1:
        initial_score = optimization_rounds[0]["best_score"]
        final_score = optimization_rounds[-1]["best_score"]
        total_improvement = final_score - initial_score
        context.log.info(
            f"📈 Total improvement: {total_improvement:.4f} (from {initial_score:.4f} to {final_score:.4f})"
        )

    return final_result
