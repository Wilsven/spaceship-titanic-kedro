"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.data_processing.nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["train", "params:preprocessing"],
                outputs="modeling_data",
                name="preprocess_data_node",
            )
        ]
    )
