//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Pooyan Dadvand
//

#pragma once

// System includes

// External includes

// Project includes
#include "voxel_mesher_entity_generation.h"

namespace Kratos {

class GenerateElementsWithCellColor: public VoxelMesherEntityGeneration {
public:
    GenerateElementsWithCellColor(VoxelMeshGeneratorModeler& rModeler, Parameters GenerationParameters):
        VoxelMesherEntityGeneration(rModeler, GenerationParameters)
    {}

    ~GenerateElementsWithCellColor() override = default;

    Parameters GetDefaultParameters() const override;

    void Generate(ModelPart& rModelPart, Parameters parameters) override;
};

}