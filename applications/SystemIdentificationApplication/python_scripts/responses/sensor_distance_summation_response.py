from typing import Optional

import KratosMultiphysics as Kratos
import KratosMultiphysics.OptimizationApplication as KratosOA
import KratosMultiphysics.SystemIdentificationApplication as KratosSI
from KratosMultiphysics.OptimizationApplication.responses.response_function import ResponseFunction
from KratosMultiphysics.OptimizationApplication.responses.response_function import SupportedSensitivityFieldVariableTypes
from KratosMultiphysics.OptimizationApplication.utilities.union_utilities import SupportedSensitivityFieldVariableTypes
from KratosMultiphysics.OptimizationApplication.utilities.model_part_utilities import ModelPartOperation
from KratosMultiphysics.OptimizationApplication.utilities.optimization_problem import OptimizationProblem
from KratosMultiphysics.OptimizationApplication.utilities.component_data_view import ComponentDataView

def Factory(model: Kratos.Model, parameters: Kratos.Parameters, optimization_problem: OptimizationProblem) -> ResponseFunction:
    if not parameters.Has("name"):
        raise RuntimeError(f"SensorDistanceSummationResponse instantiation requires a \"name\" in parameters [ parameters = {parameters}].")
    if not parameters.Has("settings"):
        raise RuntimeError(f"SensorDistanceSummationResponse instantiation requires a \"settings\" in parameters [ parameters = {parameters}].")
    return SensorDistanceSummationResponse(parameters["name"].GetString(), model, parameters["settings"], optimization_problem)


class SensorDistanceSummationResponse(ResponseFunction):
    def __init__(self, name: str, model: Kratos.Model, parameters: Kratos.Parameters, optimization_problem: OptimizationProblem) -> None:
        super().__init__(name)

        default_settings = Kratos.Parameters("""{
            "evaluated_model_part_names" : [
                "PLEASE_PROVIDE_A_MODEL_PART_NAME"
            ]
        }""")
        parameters.ValidateAndAssignDefaults(default_settings)

        self.model = model

        evaluated_model_part_names = parameters["evaluated_model_part_names"].GetStringArray()
        if len(evaluated_model_part_names) == 0:
            raise RuntimeError(f"No model parts were provided for SensorDistanceSummationResponse. [ response name = \"{self.GetName()}\"]")

        self.model_part_operation = ModelPartOperation(self.model, ModelPartOperation.OperationType.UNION, f"response_{self.GetName()}", evaluated_model_part_names, False)
        self.model_part: Optional[Kratos.ModelPart] = None
        self.optimization_problem = optimization_problem

    def GetImplementedPhysicalKratosVariables(self) -> 'list[SupportedSensitivityFieldVariableTypes]':
        return [KratosSI.SENSOR_STATUS]

    def Initialize(self) -> None:
        self.model_part = self.model_part_operation.GetModelPart()
        data = ComponentDataView("sensors", self.optimization_problem).GetUnBufferedData()
        if not data.HasValue(f"{self.model_part.FullName()}_distance_matrix"):
            distance_matrix = KratosSI.DistanceMatrix()
            nodal_positions = Kratos.Expression.NodalExpression(self.model_part)
            Kratos.Expression.NodalPositionExpressionIO.Read(nodal_positions, Kratos.Configuration.Current)
            distance_matrix.Update(nodal_positions)
            data.SetValue(f"{self.model_part.FullName()}_distance_matrix", distance_matrix)

    def Check(self) -> None:
        pass

    def Finalize(self) -> None:
        pass

    def GetInfluencingModelPart(self) -> Kratos.ModelPart:
        if self.model_part is None:
            raise RuntimeError("Please call SensorDistanceSummationResponse::Initialize first.")
        return self.model_part

    def CalculateValue(self) -> float:
        distance_matrix: KratosSI.DistanceMatrix = ComponentDataView("sensors", self.optimization_problem).GetUnBufferedData().GetValue(f"{self.model_part.FullName()}_distance_matrix")
        return KratosSI.SensorDistanceSummationResponseUtils.CalculateValue(self.model_part, distance_matrix)

    def CalculateGradient(self, physical_variable_collective_expressions: 'dict[SupportedSensitivityFieldVariableTypes, KratosOA.CollectiveExpression]') -> None:
        # make everything zeros
        for physical_variable, collective_expression in physical_variable_collective_expressions.items():
            for container_expression in collective_expression.GetContainerExpressions():
                Kratos.Expression.LiteralExpressionIO.SetDataToZero(container_expression, physical_variable)

        distance_matrix: KratosSI.DistanceMatrix = ComponentDataView("sensors", self.optimization_problem).GetUnBufferedData().GetValue(f"{self.model_part.FullName()}_distance_matrix")
        physical_variable_collective_expressions[KratosSI.SENSOR_STATUS].GetContainerExpressions()[0].SetExpression(KratosSI.SensorDistanceSummationResponseUtils.CalculateGradient(self.model_part, distance_matrix).GetExpression())

    def __str__(self) -> str:
        return f"Response [type = {self.__class__.__name__}, name = {self.GetName()}, model part name = {self.model_part.FullName()}]"