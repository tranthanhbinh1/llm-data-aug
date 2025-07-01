"""
Sensor to trigger trainer evaluation after X optimization rounds.
Now monitors iterative_optimization_asset materializations.
"""

import dagster as dg
import json
import os
from typing import Optional

from src.worker.jobs.trainer_evaluation_job import trainer_evaluation_job


# Configuration constants with environment variable support
DEFAULT_OPTIMIZATION_ROUNDS_THRESHOLD = 5
DEFAULT_MINIMUM_INTERVAL_SECONDS = 60


def get_optimization_threshold() -> int:
    """Get optimization rounds threshold from environment or use default."""
    try:
        return int(
            os.getenv(
                "OPTIMIZATION_ROUNDS_THRESHOLD", DEFAULT_OPTIMIZATION_ROUNDS_THRESHOLD
            )
        )
    except ValueError:
        return DEFAULT_OPTIMIZATION_ROUNDS_THRESHOLD


@dg.sensor(
    job=trainer_evaluation_job,
    minimum_interval_seconds=DEFAULT_MINIMUM_INTERVAL_SECONDS,
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def trainer_evaluation_sensor(
    context: dg.SensorEvaluationContext,
) -> Optional[dg.RunRequest]:
    """
    Monitor iterative_optimization_asset materializations and trigger trainer evaluation after X rounds.

    Uses cursor to track processed materializations and count optimization rounds.
    Threshold can be configured via OPTIMIZATION_ROUNDS_THRESHOLD environment variable.
    Includes concurrency control to prevent overlapping trainer evaluations.
    """
    context.log.info(
        "🔍 Trainer evaluation sensor triggered - checking for new optimization runs"
    )

    # Get threshold from environment variable or use default
    threshold = get_optimization_threshold()
    context.log.info(
        f"📊 Using optimization threshold: {threshold} (from env var or default)"
    )

    # Check if there are any currently running trainer evaluation jobs
    context.log.info("🔒 Checking for currently running trainer evaluation jobs...")
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

    if running_trainer_jobs:
        context.log.info(
            f"🚫 Skipping trainer evaluation trigger - {len(running_trainer_jobs)} trainer job(s) already running"
        )
        for job in running_trainer_jobs:
            context.log.info(
                f"   📋 Running job: {job.dagster_run.run_id} (status: {job.dagster_run.status})"
            )
        return None
    else:
        context.log.info(
            "✅ No trainer evaluation jobs currently running - proceeding with check"
        )

    # Parse cursor to get last processed storage_id
    cursor_data = {"last_storage_id": 0, "count_since_last_eval": 0}
    if context.cursor:
        try:
            cursor_data = json.loads(context.cursor)
            context.log.info(
                f"📚 Loaded cursor data: last_storage_id={cursor_data['last_storage_id']}, "
                f"count_since_last_eval={cursor_data['count_since_last_eval']}"
            )
        except json.JSONDecodeError:
            context.log.warning("⚠️ Failed to parse cursor, resetting to default")
            cursor_data = {"last_storage_id": 0, "count_since_last_eval": 0}
    else:
        context.log.info("📚 No cursor found, starting with default values")

    # Get new iterative_optimization_asset materializations since last cursor
    context.log.info(
        f"🔎 Searching for new iterative_optimization_asset materializations after storage_id: {cursor_data['last_storage_id']}"
    )
    optimization_events = context.instance.get_event_records(
        dg.EventRecordsFilter(
            event_type=dg.DagsterEventType.ASSET_MATERIALIZATION,
            asset_key=dg.AssetKey("iterative_optimization_asset"),
            after_cursor=cursor_data["last_storage_id"],
        ),
        limit=100,  # Reasonable limit to prevent memory issues
    )

    context.log.info(f"📊 Found {len(optimization_events)} new optimization events")

    if not optimization_events:
        # No new materializations
        context.log.info("📈 No new optimization runs found - sensor check complete")
        return None

    # Log details about found events
    for i, event in enumerate(optimization_events):
        context.log.info(
            f"   📋 Event {i + 1}: storage_id={event.storage_id}, "
            f"timestamp={event.timestamp}"
        )

    # Update count and cursor
    new_count = cursor_data["count_since_last_eval"] + len(optimization_events)
    new_cursor_data = {
        "last_storage_id": optimization_events[-1].storage_id,
        "count_since_last_eval": new_count,
    }

    # Update cursor
    context.update_cursor(json.dumps(new_cursor_data))
    context.log.info(
        f"📚 Updated cursor: new_count={new_count}, last_storage_id={new_cursor_data['last_storage_id']}"
    )

    context.log.info(
        f"📊 Optimization run count update: "
        f"Found {len(optimization_events)} new runs, "
        f"Total since last evaluation: {new_count}, "
        f"Threshold: {threshold}"
    )

    # Check if we should trigger trainer evaluation
    if new_count >= threshold:
        context.log.info(
            f"🎯 THRESHOLD REACHED! {new_count} >= {threshold} - triggering trainer evaluation"
        )

        # Reset count for next cycle
        reset_cursor_data = {
            "last_storage_id": optimization_events[-1].storage_id,
            "count_since_last_eval": 0,
        }
        context.update_cursor(json.dumps(reset_cursor_data))
        context.log.info("🔄 Reset optimization count to 0 for next cycle")

        # Get the latest optimization for the evaluation
        latest_optimization_event = optimization_events[-1]
        run_key = f"trainer_eval_{latest_optimization_event.storage_id}"

        context.log.info(f"🚀 TRIGGERING TRAINER EVALUATION:")
        context.log.info(f"   📋 Run key: {run_key}")
        context.log.info(
            f"   📊 Triggered after {new_count} iterative optimization runs"
        )
        context.log.info(
            f"   🔗 Based on optimization event: {latest_optimization_event.storage_id}"
        )

        return dg.RunRequest(
            run_key=run_key,
            tags={
                "optimization_runs": str(new_count),
                "trigger_event_id": str(latest_optimization_event.storage_id),
                "evaluation_type": "scheduled",
                "threshold_used": str(threshold),
            },
        )

    # Not enough rounds yet
    remaining = threshold - new_count
    context.log.info(f"⏳ Threshold not reached: {new_count}/{threshold}")
    context.log.info(
        f"📈 Need {remaining} more iterative optimization runs before triggering evaluation"
    )
    context.log.info("✅ Sensor check complete - no action taken")
    return None
