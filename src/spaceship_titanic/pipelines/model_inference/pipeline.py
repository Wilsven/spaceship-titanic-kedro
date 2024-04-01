"""
This is a boilerplate pipeline 'model_inference'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.model_inference.nodes import (
    predict_inference,
    prepare_inference_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_inference_data,
                inputs=["test", "params:preprocessing"],
                outputs="inference_data",
                name="prepare_inference_data_node",
            ),
            node(
                func=predict_inference,
                inputs=["inference_data", "params:training"],
                outputs="submission_data",
                name="predict_inference_node",
            ),
        ]
    )
