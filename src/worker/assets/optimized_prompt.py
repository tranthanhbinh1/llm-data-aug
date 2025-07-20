import dagster as dg
import asyncio
from typing import Dict, Any
from pydantic import field_validator, Field

from src.enums import Sentiment
from src.worker.evaluators import (
    LightweightEvaluator,
    HeavyweightEvaluator,
)
from src.worker.resource import LLMResource, SynthesizerResource
from src.prompt_optimization import PromptOptimizer, OptimizationConfig


class OptimizationInitializationConfig(dg.Config):
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
    population_size: int = 5
    num_iterations: int = 5
    max_optimization_rounds: int = Field(default=3, min_length=2)
    lightweight_evaluation_rounds: int = 1
    trainer_score_weight: float = 0.7

@dg.op
def initialize_optimization_op(
    context: dg.OpExecutionContext, config: OptimizationInitializationConfig
) -> Dict[str, Any]:
    """Initialize optimization parameters and state."""

    optimization_state = {
        "current_prompt": config.initial_prompt,
        "improvement_request": config.improvement_request,
        "sentiment": config.sentiment,
        "round": 0,
        "best_score": 0.0,
        "last_trainer_score": 0.0,
        "max_optimization_rounds": config.max_optimization_rounds,
        "lightweight_evaluation_rounds": config.lightweight_evaluation_rounds,
        "trainer_score_weight": config.trainer_score_weight,
        "optimization_config": {
            "population_size": config.population_size,
            "num_iterations": config.num_iterations,
            "num_elites": 1,
            "threshold": 0.7,
            "tournament_size": 3,
            "num_evaluation_samples": 2,
            "model": "gemini-2.0-flash",
            "max_retries": 3,
        },
        "history": [],
        "trainer_scores_history": [],
    }

    context.log.info("Initialized optimization state")
    return optimization_state


def determine_evaluator_strategy(
    context: dg.OpExecutionContext,
    current_round: int,
    lightweight_evaluation_rounds: int,
) -> str:
    """Determine which evaluator strategy to use for this round."""

    # Use heavyweight evaluation every N rounds
    should_use_heavyweight = (current_round % (lightweight_evaluation_rounds + 1)) == 0

    evaluator_type = "heavyweight" if should_use_heavyweight else "lightweight"

    context.log.info(f"🎯 Round {current_round}: Using {evaluator_type} evaluation")

    return evaluator_type


@dg.op
def run_full_optimization_cycle_op(
    context: dg.OpExecutionContext,
    optimization_state: Dict[str, Any],
    llm: LLMResource,
    synthesizer: SynthesizerResource,
) -> Dict[str, Any]:
    """Run the complete optimization cycle with multiple rounds and pluggable evaluators."""

    async def _run_single_round(evaluator_type: str):
        """Run a single optimization round with the specified evaluator."""
        
        # Setup optimization config
        config_dict = optimization_state["optimization_config"]
        optimization_config = OptimizationConfig(**config_dict)

        # Create the appropriate evaluator
        if evaluator_type == "heavyweight":
            evaluator = HeavyweightEvaluator(context, synthesizer)
        else:
            evaluator = LightweightEvaluator(context, synthesizer)

        context.log.info(
            f"🔧 Using {evaluator.evaluation_type} evaluator"
        )

        # Create the evaluator function for genetic algorithm
        def create_evaluator():
            async def ga_evaluator(candidate, initial_prompt, improvement_request):
                try:
                    # Use the pluggable evaluator
                    fitness_score = await evaluator.evaluate(
                        candidate.prompt, optimization_state
                    )

                    candidate.fitness = fitness_score
                    candidate.reflection = (
                        f"{evaluator.evaluation_type}: {fitness_score:.4f}"
                    )

                    return candidate

                except Exception as e:
                    context.log.error(
                        f"{evaluator.evaluation_type} evaluation failed: {e}"
                    )
                    candidate.fitness = 0.0
                    candidate.reflection = f"Failed: {str(e)}"
                    return candidate

            return ga_evaluator

        # Run optimization
        optimizer = PromptOptimizer(api_key=llm.api_key, config=optimization_config)

        result = await optimizer.optimize(
            initial_prompt=optimization_state["current_prompt"],
            improvement_request=optimization_state["improvement_request"],
            custom_evaluator=create_evaluator(),
        )

        return result, evaluator_type

    async def _run_optimization_cycle():
        """Run the complete optimization cycle with multiple rounds."""
        
        max_rounds = optimization_state["max_optimization_rounds"]
        lightweight_rounds = optimization_state["lightweight_evaluation_rounds"]
        threshold = 0.8
        
        context.log.info(f"🚀 Starting optimization cycle: {max_rounds} max rounds")
        
        for round_num in range(1, max_rounds + 1):
            # Determine evaluator strategy for this round
            evaluator_type = determine_evaluator_strategy(
                context, round_num, lightweight_rounds
            )
            
            # Run optimization round
            result, actual_evaluator_type = await _run_single_round(evaluator_type)
            
            # Update state
            optimization_state["round"] = round_num
            
            round_data = {
                "round": round_num,
                "prompt": result.best_prompt,
                "score": result.best_score,
                "evaluator_type": actual_evaluator_type,
                "iterations": result.total_iterations,
                "candidates_evaluated": result.total_candidates_evaluated,
            }
            
            optimization_state["history"].append(round_data)
            
            # Track trainer scores separately
            if actual_evaluator_type == "heavyweight":
                optimization_state["last_trainer_score"] = result.best_score
                optimization_state["trainer_scores_history"].append(
                    {
                        "round": round_num,
                        "trainer_score": result.best_score,
                    }
                )
            
            # Update the current prompt if improved
            if result.best_score > optimization_state["best_score"]:
                optimization_state["current_prompt"] = result.best_prompt
                optimization_state["best_score"] = result.best_score
            
            context.log.info(
                f"✅ Round {round_num} ({actual_evaluator_type}) - Score: {result.best_score:.4f}"
            )
            
            # Check convergence
            if result.best_score >= threshold:
                context.log.info(f"🎯 Convergence achieved! Score {result.best_score:.4f} >= {threshold}")
                break
                
        context.log.info(f"🏁 Optimization cycle complete after {optimization_state['round']} rounds")
        return optimization_state

    # Run the full optimization cycle
    return asyncio.run(_run_optimization_cycle())



@dg.op
def finalize_optimization_op(
    context: dg.OpExecutionContext,
    optimization_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Finalize optimization and return results."""

    trainer_scores_history = optimization_state.get("trainer_scores_history", [])
    last_trainer_score = optimization_state.get("last_trainer_score", 0.0)

    final_result = {
        "final_prompt": optimization_state["current_prompt"],
        "final_score": optimization_state["best_score"],
        "final_trainer_score": last_trainer_score,
        "total_rounds": optimization_state["round"],
        "converged": optimization_state["best_score"] >= 0.8,
        "history": optimization_state["history"],
        "trainer_scores_history": trainer_scores_history,
        "config": {
            "max_optimization_rounds": optimization_state["max_optimization_rounds"],
            "lightweight_evaluation_rounds": optimization_state[
                "lightweight_evaluation_rounds"
            ],
            "trainer_score_weight": optimization_state["trainer_score_weight"],
            "sentiment": optimization_state["sentiment"],
        },
    }

    # Calculate evaluation type distribution
    evaluation_types = [
        round_data.get("evaluator_type", "unknown")
        for round_data in optimization_state["history"]
    ]
    lightweight_count = evaluation_types.count("lightweight")
    heavyweight_count = evaluation_types.count("heavyweight")

    context.log.info("🏁 OPTIMIZATION COMPLETE!")
    context.log.info(f"📊 Final Results:")
    context.log.info(f"   🔄 Total rounds: {optimization_state['round']}")
    context.log.info(f"   📈 Final score: {optimization_state['best_score']:.4f}")
    context.log.info(f"   🏋️ Final trainer score: {last_trainer_score:.4f}")
    context.log.info(f"   🎯 Converged: {final_result['converged']}")
    context.log.info(f"   ⚡ Lightweight evaluations: {lightweight_count}")
    context.log.info(f"   🏋️ Heavyweight evaluations: {heavyweight_count}")
    context.log.info(
        f"   📝 Final prompt length: {len(optimization_state['current_prompt'])} chars"
    )

    return final_result


# Create the graph-backed asset
@dg.graph_asset
def optimization_result():
    """
    Graph-backed asset that produces optimized prompts using pluggable evaluators.

    Returns:
        Dict containing the optimization results including the final prompt,
        similarity score, trainer scores, convergence status, and complete history.

    Features:
    - Multi-round optimization cycle
    - Pluggable evaluators (lightweight vs. heavyweight)
    - Trainer integration for heavyweight evaluation
    - Configurable evaluation intervals
    - Automatic convergence detection
    - Comprehensive logging and tracking
    """
    # Initialize optimization state with configuration
    initial_state = initialize_optimization_op()

    # Run the complete optimization cycle with multiple rounds
    optimized_state = run_full_optimization_cycle_op(initial_state)

    # Finalize and return comprehensive results
    return finalize_optimization_op(optimized_state)
