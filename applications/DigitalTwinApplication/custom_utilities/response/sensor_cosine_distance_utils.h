//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         DigitalTwinApplication/license.txt
//
//  Main authors:    Suneth Warnakulasuriya
//

#pragma once

// System includes

// External includes

// Project includes
#include "includes/define.h"
#include "expression/container_expression.h"

// Application includes

namespace Kratos {
///@name Kratos Classes
///@{

class SensorCosineDistanceResponseUtils
{
public:
    ///@name Type definitions
    ///@{

    KRATOS_CLASS_POINTER_DEFINITION(SensorCosineDistanceResponseUtils);

    ///@}
    ///@name Life cycle
    ///@{

    SensorCosineDistanceResponseUtils(
        ModelPart& rSensorModelPart,
        const double P);

    ///@}
    ///@name Public operations
    ///@{

    void Initialize(const std::vector<ContainerExpression<ModelPart::ElementsContainerType>::Pointer>& rMasksList);

    double CalculateValue() const;

    ContainerExpression<ModelPart::NodesContainerType> CalculateGradient() const;

    ///@}

private:
    ///@name Private member variables
    ///@{

    ModelPart* mpSensorModelPart;

    double mP;

    Matrix mDistances;

    ///@}
};

///@} // Kratos Classes

} /* namespace Kratos.*/