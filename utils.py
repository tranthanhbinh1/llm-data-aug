import os

import pandas as pd
from sklearn.metrics import classification_report


def save_classification_report(
    true_labels, predictions, target_names, project_root, data_scenario
):
    # Save the classification report
    report = classification_report(true_labels, predictions, target_names=target_names)
    print(type(report))
    print(report)
    classification_report_df = pd.DataFrame(report).transpose()
    classification_report_df.to_csv(
        os.path.join(
            project_root,
            "reports",
            f"{data_scenario.name.lower()}_phobert_v1_classification_report.csv",
        )
    )
