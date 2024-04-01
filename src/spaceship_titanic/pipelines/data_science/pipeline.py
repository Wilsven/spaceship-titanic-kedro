"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.data_science.nodes import (
    compare_candidate_models,
    evaluate_candidate_models,
    split_data,
    tune_candidate_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["modeling_data", "params:tuning", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=tune_candidate_models,
                inputs=["X_train", "y_train", "params:tuning"],
                outputs="data",  # forcing the running order
                name="tune_candidate_models_node",
            ),
            node(
                func=evaluate_candidate_models,
                inputs=["data", "X_test", "y_test", "params:tuning"],
                outputs="candidate_model",  # forcing the running order
                name="evaluate_candidate_models_node",
            ),
            node(
                func=compare_candidate_models,
                inputs=[
                    "candidate_model",
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:training",
                    "params:tuning",
                ],
                outputs=None,
                name="compare_candidate_models_node",
            ),
        ]
    )
