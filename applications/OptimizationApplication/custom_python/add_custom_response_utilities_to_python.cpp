//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   license: OptimizationApplication/license.txt
//
//  Main author:     Suneth Warnakulasuriya
//

// System includes

// External includes

// Project includes

// Application includes
#include "custom_utilities/response/mass_response_utils.h"
#include "custom_utilities/response/linear_strain_energy_response_utils.h"

// Include base h
#include "add_custom_response_utilities_to_python.h"

namespace Kratos {
namespace Python {

void  AddCustomResponseUtilitiesToPython(pybind11::module& m)
{
    namespace py = pybind11;

    py::class_<MassResponseUtils >(m, "MassResponseUtils")
        .def_static("CalculateMass", &MassResponseUtils::CalculateMass)
        .def_static("CalculateMassShapeSensitivity", &MassResponseUtils::CalculateMassShapeSensitivity)
        .def_static("CalculateMassDensitySensitivity", &MassResponseUtils::CalculateMassDensitySensitivity)
        .def_static("CalculateMassThicknessSensitivity", &MassResponseUtils::CalculateMassThicknessSensitivity)
        .def_static("CalculateMassCrossAreaSensitivity", &MassResponseUtils::CalculateMassCrossAreaSensitivity)
        ;

    py::class_<LinearStrainEnergyResponseUtils >(m, "LinearStrainEnergyResponseUtils")
        .def_static("CalculateStrainEnergy", &LinearStrainEnergyResponseUtils::CalculateStrainEnergy)
        .def_static("CalculateStrainEnergyShapeSensitivity", &LinearStrainEnergyResponseUtils::CalculateStrainEnergyShapeSensitivity)
        .def_static("CalculateStrainEnergyYoungModulusSensitivity", &LinearStrainEnergyResponseUtils::CalculateStrainEnergyYoungModulusSensitivity)
        .def_static("CalculateStrainEnergyNonLinearSensitivity", &LinearStrainEnergyResponseUtils::CalculateStrainEnergyNonLinearSensitivity)
        ;

}

}  // namespace Python.
} // Namespace Kratos
