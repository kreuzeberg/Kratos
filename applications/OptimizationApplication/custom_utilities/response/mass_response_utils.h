//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   license: OptimizationApplication/license.txt
//
//  Main author:     Reza Najian Asl,
//                   Suneth Warnakulasuriya
//

#pragma once

// System includes
#include <variant>
#include <vector>

// Project includes
#include "includes/define.h"
#include "includes/model_part.h"

// Application includes

namespace Kratos
{

///@name Kratos Classes
///@{

class KRATOS_API(OPTIMIZATION_APPLICATION) MassResponseUtils
{
public:
    ///@name Type definitions
    ///@{

    using IndexType = std::size_t;

    using GeometryType = ModelPart::ElementType::GeometryType;

    using GradientFieldVariableTypes = std::variant<const Variable<double>*, const Variable<array_1d<double, 3>>*>;

    ///@}
    ///@name Static operations
    ///@{

    static void Check(const ModelPart& rModelPart);

    static double CalculateValue(const ModelPart& rModelPart);

    static void CalculateGradient(
        const std::vector<GradientFieldVariableTypes>& rListOfGradientVariables,
        const std::vector<ModelPart*>& rListOfGradientRequiredModelParts,
        const std::vector<ModelPart*>& rListOfGradientComputedModelParts);

    ///@}
private:
    ///@name Private operations
    ///@{

    static bool HasVariableInProperties(
        const ModelPart& rModelPart,
        const Variable<double>& rVariable);

    static void CalculateMassGeometricalPropertyGradient(
        ModelPart& rModelPart,
        const Variable<double>& rGeometricalPropertyGradientVariable,
        const Variable<double>& rGeometricalCoflictingPropertyGradientVariable,
        const Variable<double>& rOutputGradientVariable);

    static void CalculateMassShapeGradient(
        ModelPart& rModelPart,
        const Variable<array_1d<double, 3>>& rOutputGradientVariable);

    static void CalculateMassDensityGradient(
        ModelPart& rModelPart,
        const Variable<double>& rOutputGradientVariable);

    static void CalculateMassThicknessGradient(
        ModelPart& rModelPart,
        const Variable<double>& rOutputGradientVariable);

    static void CalculateMassCrossAreaGradient(
        ModelPart& rModelPart,
        const Variable<double>& rOutputGradientVariable);

    ///@}
};

///@}
}