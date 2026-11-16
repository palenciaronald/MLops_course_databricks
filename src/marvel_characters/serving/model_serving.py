"""Model serving module for Marvel characters."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)


class ModelServing:
    """Manages model serving in Databricks for Marvel characters."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving Manager.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        # model_name has the form <catalog>.<schema>.<model>; inference logs go to the same schema
        self.catalog_name, self.schema_name, _ = model_name.split(".")

    def get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: Latest version of the model as a string
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
        enable_inference_table: bool = False,
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks for Marvel characters.

        :param version: Model version to serve (default: "latest")
        :param workload_size: Size of the serving workload (default: "Small")
        :param scale_to_zero: Whether to enable scale-to-zero (default: True)
        :param enable_inference_table: Log requests/responses with an AI Gateway inference table (default: False).
            Not available in every workspace (e.g. Free Edition), so it is opt-in.
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        # Log every request/response to <catalog>.<schema>.custom_model_payload (read by monitoring.py)
        ai_gateway = None
        if enable_inference_table:
            ai_gateway = AiGatewayConfig(
                inference_table_config=AiGatewayInferenceTableConfig(
                    catalog_name=self.catalog_name,
                    schema_name=self.schema_name,
                    table_name_prefix="custom_model",
                    enabled=True,
                )
            )

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=served_entities),
                ai_gateway=ai_gateway,
            )
        else:
            if ai_gateway is not None:
                self.workspace.serving_endpoints.put_ai_gateway(
                    name=self.endpoint_name, inference_table_config=ai_gateway.inference_table_config
                )
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
