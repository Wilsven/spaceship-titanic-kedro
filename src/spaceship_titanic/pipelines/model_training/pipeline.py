"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.model_training.nodes import retrain_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=retrain_model,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:training"],
                outputs=None,
                name="retrain_model_node",
            )
        ]
    )
