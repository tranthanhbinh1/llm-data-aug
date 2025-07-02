"""
Sensor to trigger next optimization cycle after trainer evaluation completes.
This creates the complete iterative loop: optimization -> trainer evaluation -> optimization -> ...
"""

import dagster as dg
import json
import os
from typing import Optional

from src.worker.jobs.optimization_job import iterative_optimization_job


# Configuration constants with environment variable support
DEFAULT_MAX_TOTAL_CYCLES = 50
DEFAULT_MINIMUM_INTERVAL_SECONDS = (
    120  # Longer interval to allow trainer eval to complete
)


def get_max_total_cycles() -> int:
    """Get maximum total cycles from environment or use default."""
    try:
        return int(os.getenv("MAX_TOTAL_CYCLES", DEFAULT_MAX_TOTAL_CYCLES))
    except ValueError:
        return DEFAULT_MAX_TOTAL_CYCLES


def get_cycle_enabled() -> bool:
    """Check if continuous cycling is enabled via environment variable."""
    return os.getenv("ENABLE_CONTINUOUS_CYCLES", "false").lower() in (
        "true",
        "1",
        "yes",
    )


@dg.sensor(
    job=iterative_optimization_job,
    minimum_interval_seconds=DEFAULT_MINIMUM_INTERVAL_SECONDS,
    default_status=dg.DefaultSensorStatus.STOPPED,  # Start disabled by default
)
def optimization_cycle_sensor(
    context: dg.SensorEvaluationContext,
) -> Optional[dg.RunRequest]:
    """
    Monitor trainer evaluation job completions and trigger next optimization cycle.

    This sensor creates the complete iterative loop:
    1. Optimization runs N times (monitored by trainer_evaluation_sensor)
    2. Trainer evaluation is triggered automatically
    3. This sensor detects trainer evaluation completion
    4. Triggers next optimization cycle
    5. Repeat until max cycles reached or manually stopped

    Configuration:
    - ENABLE_CONTINUOUS_CYCLES: Set to "true" to enable (default: false)
    - MAX_TOTAL_CYCLES: Maximum cycles before stopping (default: 50)
    """
    context.log.info(
        "🔄 Optimization cycle sensor triggered - checking for completed trainer evaluations"
    )

    # Check if continuous cycling is enabled
    if not get_cycle_enabled():
        context.log.info(
            "📴 Continuous cycling is disabled (ENABLE_CONTINUOUS_CYCLES=false)"
        )
        context.log.info(
            "💡 To enable: export ENABLE_CONTINUOUS_CYCLES=true && dagster sensor start optimization_cycle_sensor"
        )
        return None

    # Get configuration
    max_cycles = get_max_total_cycles()
    context.log.info(f"⚙️ Configuration: max_cycles={max_cycles}")

    # Parse cursor to track total cycles and last processed trainer evaluation
    cursor_data = {"total_cycles": 0, "last_trainer_eval_storage_id": 0}
    if context.cursor:
        try:
            cursor_data = json.loads(context.cursor)
            context.log.info(
                f"📚 Loaded cursor: total_cycles={cursor_data['total_cycles']}, "
                f"last_eval_id={cursor_data['last_trainer_eval_storage_id']}"
            )
        except json.JSONDecodeError:
            context.log.warning("⚠️ Failed to parse cursor, resetting to default")
            cursor_data = {"total_cycles": 0, "last_trainer_eval_storage_id": 0}
    else:
        context.log.info("📚 No cursor found, starting fresh")

    # Check if we've reached maximum cycles
    if cursor_data["total_cycles"] >= max_cycles:
        context.log.info(
            f"🏁 Maximum cycles reached: {cursor_data['total_cycles']}/{max_cycles}"
        )
        context.log.info(
            "🛑 Stopping continuous cycling - manually restart sensor to continue"
        )
        return None

    # Check for currently running jobs to avoid conflicts
    context.log.info("🔒 Checking for currently running jobs...")

    # Check for running optimization jobs
    running_optimization_jobs = context.instance.get_run_records(
        dg.RunsFilter(
            job_name="iterative_optimization_job",
            statuses=[
                dg.DagsterRunStatus.QUEUED,
                dg.DagsterRunStatus.NOT_STARTED,
                dg.DagsterRunStatus.STARTING,
                dg.DagsterRunStatus.STARTED,
            ],
        )
    )

    # Check for running trainer evaluation jobs
    running_trainer_jobs = context.instance.get_run_records(
        dg.RunsFilter(
            job_name="trainer_evaluation_job",
            statuses=[
                dg.DagsterRunStatus.QUEUED,
                dg.DagsterRunStatus.NOT_STARTED,
                dg.DagsterRunStatus.STARTING,
                dg.DagsterRunStatus.STARTED,
            ],
        )
    )

    if running_optimization_jobs:
        context.log.info(
            f"⏸️ Skipping cycle trigger - {len(running_optimization_jobs)} optimization job(s) running"
        )
        return None

    if running_trainer_jobs:
        context.log.info(
            f"⏸️ Skipping cycle trigger - {len(running_trainer_jobs)} trainer evaluation job(s) running"
        )
        return None

    context.log.info("✅ No conflicting jobs running - proceeding with check")

    # Get recent successful trainer evaluation runs
    context.log.info(
        "🔎 Searching for completed trainer evaluations since cursor timestamp"
    )

    # Get successful trainer evaluation runs
    successful_trainer_runs = context.instance.get_run_records(
        dg.RunsFilter(
            job_name="trainer_evaluation_job",
            statuses=[dg.DagsterRunStatus.SUCCESS],
        ),
        limit=50,
    )

    # Filter for runs completed after our last cursor
    last_cursor_id = cursor_data["last_trainer_eval_storage_id"]
    new_trainer_completions = []

    for run_record in successful_trainer_runs:
        # Use run ID as a simple cursor mechanism (run IDs are sequential)
        if run_record.dagster_run.run_id > str(last_cursor_id):
            new_trainer_completions.append(run_record)

    context.log.info(
        f"📊 Found {len(new_trainer_completions)} newly completed trainer evaluation runs"
    )

    if not new_trainer_completions:
        context.log.info(
            "📈 No new trainer evaluation completions - sensor check complete"
        )
        return None

    # Log details about found completions
    for i, run_record in enumerate(new_trainer_completions):
        context.log.info(
            f"   📋 Completion {i + 1}: run_id={run_record.dagster_run.run_id}, "
            f"status={run_record.dagster_run.status}, "
            f"end_time={run_record.dagster_run.end_time}"
        )

    # Trigger next optimization cycle
    latest_completion = new_trainer_completions[-1]  # Most recent completion
    new_cycle_count = cursor_data["total_cycles"] + 1

    context.log.info("🎯 TRIGGERING NEXT OPTIMIZATION CYCLE:")
    context.log.info(f"   🔄 Cycle number: {new_cycle_count}/{max_cycles}")
    context.log.info(
        f"   🔗 Based on trainer evaluation: {latest_completion.dagster_run.run_id}"
    )

    # Update cursor
    new_cursor_data = {
        "total_cycles": new_cycle_count,
        "last_trainer_eval_storage_id": latest_completion.dagster_run.run_id,
    }
    context.update_cursor(json.dumps(new_cursor_data))
    context.log.info(f"📚 Updated cursor for cycle {new_cycle_count}")

    # Create run key
    run_key = (
        f"optimization_cycle_{new_cycle_count}_{latest_completion.dagster_run.run_id}"
    )

    context.log.info(f"🚀 STARTING OPTIMIZATION CYCLE {new_cycle_count}")
    context.log.info(f"   📋 Run key: {run_key}")
    context.log.info(f"   🔄 Remaining cycles: {max_cycles - new_cycle_count}")

    return dg.RunRequest(
        run_key=run_key,
        tags={
            "cycle_number": str(new_cycle_count),
            "max_cycles": str(max_cycles),
            "trigger_type": "automatic_cycle",
            "based_on_trainer_eval": latest_completion.dagster_run.run_id,
            "remaining_cycles": str(max_cycles - new_cycle_count),
        },
    )
