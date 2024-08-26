//  KRATOS  _____     _ _ _
//         |_   _| __(_) (_)_ __   ___  ___
//           | || '__| | | | '_ \ / _ \/ __|
//           | || |  | | | | | | | (_) \__
//           |_||_|  |_|_|_|_| |_|\___/|___/ APPLICATION
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Riccardo Rossi
//

#pragma once

// System includes

// External includes

// Project includes
#include "includes/define.h"
#include "includes/kratos_application.h"

namespace Kratos {

///@name Kratos Globals
///@{

// Variables definition

///@}
///@name Type Definitions
///@{

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

/**
 * @class KratosTrilinosApplication
 * @ingroup TrilinosApplication
 * @brief This Application is used to interface with Trilinos
 * @details The TrilinosApplication is the application used to interface with Trilinos. It is used to create the linear solvers and preconditioners that are available in Trilinos. The application is designed to be as general as possible, so that it can be used with any of the Trilinos packages. The application is designed to be used with the TrilinosApplicationFactory, which is used to create the linear solvers and preconditioners that are available in Trilinos. The application is designed to be used with the TrilinosApplicationFactory, which is used to create the linear solvers and preconditioners that are available in Trilinos. The application is designed to be used with the TrilinosApplicationFactory, which is used to create the linear solvers and preconditioners that are available in Trilinos. The application is designed to be used with the TrilinosApplicationFactory, which is used to create the linear solvers and preconditioners that are available in Trilinos.
*/
class KRATOS_API(TRILINOS_APPLICATION) KratosTrilinosApplication : public KratosApplication {
   public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of KratosTrilinosApplication
    KRATOS_CLASS_POINTER_DEFINITION(KratosTrilinosApplication);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    KratosTrilinosApplication() : KratosApplication("TrilinosApplication") {}

    /// Destructor.
    ~KratosTrilinosApplication() override {}

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    void Register() override;

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const override { return "KratosTrilinosApplication"; }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override {
        rOStream << Info();
        PrintData(rOStream);
    }

    ///// Print object's data.
    void PrintData(std::ostream& rOStream) const override {
        KRATOS_WATCH("in KratosMeshMovingApplication application");
        KRATOS_WATCH(KratosComponents<VariableData>::GetComponents().size());
        rOStream << "Variables:" << std::endl;
        KratosComponents<VariableData>().PrintData(rOStream);
        rOStream << std::endl;
        rOStream << "Elements:" << std::endl;
        KratosComponents<Element>().PrintData(rOStream);
        rOStream << std::endl;
        rOStream << "Conditions:" << std::endl;
        KratosComponents<Condition>().PrintData(rOStream);
    }

    ///@}
    ///@name Friends
    ///@{

    ///@}
   protected:
    ///@name Protected static Member Variables
    ///@{

    ///@}
    ///@name Protected member Variables
    ///@{

    ///@}
    ///@name Protected Operators
    ///@{

    ///@}
    ///@name Protected Operations
    ///@{

    ///@}
    ///@name Protected  Access
    ///@{

    ///@}
    ///@name Protected Inquiry
    ///@{

    ///@}
    ///@name Protected LifeCycle
    ///@{

    ///@}

   private:
    ///@name Static Member Variables
    ///@{

    //       static const ApplicationCondition  msApplicationCondition;

    ///@}
    ///@name Member Variables
    ///@{

    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{

    ///@}
    ///@name Private  Access
    ///@{

    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Un accessible methods
    ///@{

    /// Assignment operator.
    KratosTrilinosApplication& operator=(
        KratosTrilinosApplication const& rOther);

    /// Copy constructor.
    KratosTrilinosApplication(KratosTrilinosApplication const& rOther);

    ///@}

};  // Class KratosTrilinosApplication

///@}

///@name Type Definitions
///@{

///@}
///@name Input and output
///@{

///@}

}  // namespace Kratos.
