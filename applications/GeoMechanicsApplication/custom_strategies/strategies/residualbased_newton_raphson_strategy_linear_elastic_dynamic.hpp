//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ \.
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Aron Noordam
//


#pragma once

// System includes
#include <iostream>

// External includes

// Project includes

#include "includes/define.h"
#include "includes/kratos_parameters.h"
#include "includes/model_part.h"

//#include "includes/define.h"
#include "solving_strategies/strategies/residualbased_newton_raphson_strategy.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"
#include "utilities/builtin_timer.h"

//default builder and solver
#include "custom_strategies/builder_and_solvers/residualbased_block_builder_and_solver_linear_elastic_dynamic.h"
//#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "solving_strategies/strategies/implicit_solving_strategy.h"

// Application includes
#include "geo_mechanics_application_variables.h"

namespace Kratos
{

///@name Kratos Globals
///@{

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
 * @class GeoMechanicNewtonRaphsonStrategyLinearElasticDynamic
 * @ingroup KratosCore
 * @brief This is the base Newton Raphson strategy
 * @details This strategy iterates until the convergence is achieved (or the maximum number of iterations is surpassed) using a Newton Raphson algorithm
 * @author Riccardo Rossi
 */
template <class TSparseSpace,
          class TDenseSpace,  // = DenseSpace<double>,
          class TLinearSolver //= LinearSolver<TSparseSpace,TDenseSpace>
          >
class GeoMechanicNewtonRaphsonStrategyLinearElasticDynamic
    : public ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{
  public:
    ///@name Type Definitions
    ///@{
    typedef ConvergenceCriteria<TSparseSpace, TDenseSpace> TConvergenceCriteriaType;

    // Counted pointer of ClassName
    KRATOS_CLASS_POINTER_DEFINITION(GeoMechanicNewtonRaphsonStrategyLinearElasticDynamic);

    typedef SolvingStrategy<TSparseSpace, TDenseSpace> SolvingStrategyType;

	using BaseType = ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>;
    //typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;
    using MotherType = ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>;

    typedef GeoMechanicNewtonRaphsonStrategyLinearElasticDynamic<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

    typedef typename BaseType::TBuilderAndSolverType TBuilderAndSolverType;

    typedef typename BaseType::TDataType TDataType;

    typedef TSparseSpace SparseSpaceType;

    typedef typename BaseType::TSchemeType TSchemeType;

    typedef typename BaseType::DofsArrayType DofsArrayType;

    typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

    typedef typename BaseType::TSystemVectorType TSystemVectorType;

    typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

    typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

    typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;

    typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;

    ///@}
    ///@name Life Cycle
    ///@{


    /**
 * Default constructor
 * @param rModelPart The model part of the problem
 * @param pScheme The integration scheme
 * @param pNewLinearSolver The linear solver employed
 * @param pNewConvergenceCriteria The convergence criteria employed
 * @param MaxIterations The maximum number of non-linear iterations to be considered when solving the problem
 * @param CalculateReactions The flag for the reaction calculation
 * @param ReformDofSetAtEachStep The flag that allows to compute the modification of the DOF
 * @param MoveMeshFlag The flag that allows to move the mesh
 */
    explicit GeoMechanicNewtonRaphsonStrategyLinearElasticDynamic(
        ModelPart& rModelPart,
        typename TSchemeType::Pointer pScheme,
        typename TLinearSolver::Pointer pNewLinearSolver,
        typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
        typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
        Parameters& rParameters,
        int MaxIterations = 30,
        bool CalculateReactions = false,
        bool ReformDofSetAtEachStep = false,
        bool MoveMeshFlag = false)
        : ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(
            rModelPart,
            pScheme,
            /*pNewLinearSolver,*/
            pNewConvergenceCriteria,
            pNewBuilderAndSolver,
            MaxIterations,
            CalculateReactions,
            ReformDofSetAtEachStep,
            MoveMeshFlag)
    {
        // new constructor
    }



    /**
     * @brief Operation to predict the solution ... if it is not called a trivial predictor is used in which the
    values of the solution step of interest are assumed equal to the old values
     */
    void Predict() override
    {
        KRATOS_TRY
        const DataCommunicator &r_comm = BaseType::GetModelPart().GetCommunicator().GetDataCommunicator();
        //OPERATIONS THAT SHOULD BE DONE ONCE - internal check to avoid repetitions
        //if the operations needed were already performed this does nothing
        if (mInitializeWasPerformed == false)
            Initialize();


        auto equation_size = GetBuilderAndSolver()->GetDofSet().size();

        ModelPart& model_part = BaseType::GetModelPart();

        double delta_time = model_part.GetProcessInfo()[DELTA_TIME];

        TSystemVectorType& rFirstDerivativeVector = TSystemVectorType(equation_size, 0.0);
		TSystemVectorType& rSecondDerivativeVector = TSystemVectorType(equation_size, 0.0);

        TSystemVectorType& rUpdatedFirstDerivativeVector = TSystemVectorType(equation_size, 0.0);
        TSystemVectorType& rUpdatedSecondDerivativeVector = TSystemVectorType(equation_size, 0.0);

		this->GetFirstAndSecondDerivativeVector(rFirstDerivativeVector, rSecondDerivativeVector, model_part, 0);

        noalias(rUpdatedSecondDerivativeVector) = rFirstDerivativeVector * (1.0 / (mBeta * delta_time)) + rSecondDerivativeVector * (1.0 / (2 * mBeta));
        noalias(rUpdatedFirstDerivativeVector) = rFirstDerivativeVector * (mGamma / mBeta) + rSecondDerivativeVector * (delta_time * (mGamma / (2 * mBeta) - 1));

		this->SetFirstAndSecondDerivativeVector(rUpdatedFirstDerivativeVector, rUpdatedSecondDerivativeVector, model_part);

        // Move the mesh if needed
        if (this->MoveMeshFlag() == true)
            BaseType::MoveMesh();

        KRATOS_CATCH("")
    }

    /**
     * @brief Initialization of member variables and prior operations
     */
    void Initialize() override
    {
        KRATOS_TRY;

        if (mInitializeWasPerformed == false)
        {
            //pointers needed in the solution
            typename TSchemeType::Pointer p_scheme = GetScheme();
            typename TConvergenceCriteriaType::Pointer p_convergence_criteria = mpConvergenceCriteria;

            //Initialize The Scheme - OPERATIONS TO BE DONE ONCE
            if (p_scheme->SchemeIsInitialized() == false)
                p_scheme->Initialize(BaseType::GetModelPart());

            //Initialize The Elements - OPERATIONS TO BE DONE ONCE
            if (p_scheme->ElementsAreInitialized() == false)
                p_scheme->InitializeElements(BaseType::GetModelPart());

            //Initialize The Conditions - OPERATIONS TO BE DONE ONCE
            if (p_scheme->ConditionsAreInitialized() == false)
                p_scheme->InitializeConditions(BaseType::GetModelPart());

            //initialisation of the convergence criteria
            if (p_convergence_criteria->IsInitialized() == false)
                p_convergence_criteria->Initialize(BaseType::GetModelPart());

            mInitializeWasPerformed = true;
        }

        KRATOS_CATCH("");
    }

    /**
     * @brief Performs all the required operations that should be done (for each step) before solving the solution step.
     * @details A member variable should be used as a flag to make sure this function is called only once per step.
     */
    void InitializeSolutionStep() override
    {
        KRATOS_TRY;

        // Pointers needed in the solution
        typename TSchemeType::Pointer p_scheme = GetScheme();
        typename TBuilderAndSolverType::Pointer p_builder_and_solver = GetBuilderAndSolver();
        ModelPart& r_model_part = BaseType::GetModelPart();

        // Set up the system, operation performed just once unless it is required
        // to reform the dof set at each iteration
        BuiltinTimer system_construction_time;
        if (!p_builder_and_solver->GetDofSetIsInitializedFlag() || mReformDofSetAtEachStep)
        {
            // Setting up the list of the DOFs to be solved
            BuiltinTimer setup_dofs_time;
            p_builder_and_solver->SetUpDofSet(p_scheme, r_model_part);
            KRATOS_INFO_IF("ResidualBasedNewtonRaphsonStrategy", BaseType::GetEchoLevel() > 0) << "Setup Dofs Time: "
                << setup_dofs_time << std::endl;

            // Shaping correctly the system
            BuiltinTimer setup_system_time;
            p_builder_and_solver->SetUpSystem(r_model_part);
            KRATOS_INFO_IF("ResidualBasedNewtonRaphsonStrategy", BaseType::GetEchoLevel() > 0) << "Setup System Time: "
                << setup_system_time << std::endl;

            // Setting up the Vectors involved to the correct size
            BuiltinTimer system_matrix_resize_time;
            p_builder_and_solver->ResizeAndInitializeVectors(p_scheme, mpA, mpDx, mpb,
                                                                r_model_part);
            KRATOS_INFO_IF("ResidualBasedNewtonRaphsonStrategy", BaseType::GetEchoLevel() > 0) << "System Matrix Resize Time: "
                << system_matrix_resize_time << std::endl;
        }

        KRATOS_INFO_IF("ResidualBasedNewtonRaphsonStrategy", BaseType::GetEchoLevel() > 0) << "System Construction Time: "
            << system_construction_time << std::endl;

        TSystemMatrixType& rA  = *mpA;
        TSystemVectorType& rDx = *mpDx;
        TSystemVectorType& rb  = *mpb;

        // Initial operations ... things that are constant over the Solution Step
        p_builder_and_solver->InitializeSolutionStep(r_model_part, rA, rDx, rb);

        // Initial operations ... things that are constant over the Solution Step
        p_scheme->InitializeSolutionStep(r_model_part, rA, rDx, rb);

        // Initialisation of the convergence criteria
        if (mpConvergenceCriteria->GetActualizeRHSflag())
        {
            TSparseSpace::SetToZero(rb);
            p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
        }

        mpConvergenceCriteria->InitializeSolutionStep(r_model_part, p_builder_and_solver->GetDofSet(), rA, rDx, rb);

        if (mpConvergenceCriteria->GetActualizeRHSflag()) {
            TSparseSpace::SetToZero(rb);
        }

        KRATOS_CATCH("");
    }

    /**
     * @brief Performs all the required operations that should be done (for each step) after solving the solution step.
     * @details A member variable should be used as a flag to make sure this function is called only once per step.
     */
    void FinalizeSolutionStep() override
    {
        KRATOS_TRY;

        ModelPart& r_model_part = BaseType::GetModelPart();

        typename TSchemeType::Pointer p_scheme = GetScheme();
        typename TBuilderAndSolverType::Pointer p_builder_and_solver = GetBuilderAndSolver();

        TSystemMatrixType& rA  = *mpA;
        TSystemVectorType& rDx = *mpDx;
        TSystemVectorType& rb  = *mpb;

        //Finalisation of the solution step,
        //operations to be done after achieving convergence, for example the
        //Final Residual Vector (mb) has to be saved in there
        //to avoid error accumulation

        p_scheme->FinalizeSolutionStep(r_model_part, rA, rDx, rb);
        p_builder_and_solver->FinalizeSolutionStep(r_model_part, rA, rDx, rb);
        mpConvergenceCriteria->FinalizeSolutionStep(r_model_part, p_builder_and_solver->GetDofSet(), rA, rDx, rb);

        //Cleaning memory after the solution
        p_scheme->Clean();

        if (mReformDofSetAtEachStep == true) //deallocate the systemvectors
        {
            this->Clear();
        }

        KRATOS_CATCH("");
    }


    /**
     * @brief Solves the current step. This function returns true if a solution has been found, false otherwise.
     */
    bool SolveSolutionStep() override
    {
        // Pointers needed in the solution
        ModelPart& r_model_part = BaseType::GetModelPart();
        typename TSchemeType::Pointer p_scheme = GetScheme();
        typename TBuilderAndSolverType::Pointer p_builder_and_solver = GetBuilderAndSolver();
        auto& r_dof_set = p_builder_and_solver->GetDofSet();
        std::vector<Vector> NonconvergedSolutions;

        if (mStoreNonconvergedSolutionsFlag) {
            Vector initial;
            GetCurrentSolution(r_dof_set,initial);
            NonconvergedSolutions.push_back(initial);
        }

        TSystemMatrixType& rA  = *mpA;
        TSystemVectorType& rDx = *mpDx;
        TSystemVectorType& rb  = *mpb;

        TSystemVectorType& rDx_tot = TSystemVectorType(r_dof_set.size(), 0.0);

        //initializing the parameters of the Newton-Raphson cycle
        unsigned int iteration_number = 1;
        r_model_part.GetProcessInfo()[NL_ITERATION_NUMBER] = iteration_number;
        bool residual_is_updated = false;
        p_scheme->InitializeNonLinIteration(r_model_part, rA, rDx, rb);
        mpConvergenceCriteria->InitializeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);
        bool is_converged = mpConvergenceCriteria->PreCriteria(r_model_part, r_dof_set, rA, rDx, rb);

        // Function to perform the building and the solving phase.
        if (BaseType::mStiffnessMatrixIsBuilt == false) {
            TSparseSpace::SetToZero(rA);
            TSparseSpace::SetToZero(rDx);
            TSparseSpace::SetToZero(rb);

            p_builder_and_solver->BuildAndSolve(p_scheme, r_model_part, rA, rDx, rb);

        } else {
            TSparseSpace::SetToZero(rDx);  // Dx = 0.00;
            TSparseSpace::SetToZero(rb);

            p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
        }

        // Debugging info
        EchoInfo(iteration_number);

        // Updating the results stored in the database

        this->UpdateSolutionStepValue(rDx, rDx_tot);

        p_scheme->FinalizeNonLinIteration(r_model_part, rA, rDx, rb);
        mpConvergenceCriteria->FinalizeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

        if (mStoreNonconvergedSolutionsFlag) {
            Vector first;
            GetCurrentSolution(r_dof_set,first);
            NonconvergedSolutions.push_back(first);
        }

        if (is_converged) {
            if (mpConvergenceCriteria->GetActualizeRHSflag()) {
                TSparseSpace::SetToZero(rb);

                p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
            }

            is_converged = mpConvergenceCriteria->PostCriteria(r_model_part, r_dof_set, rA, rDx, rb);
        }

		

        //Iteration Cycle... performed only for non linear RHS
        while (is_converged == false &&
               iteration_number++ < mMaxIterationNumber)
        {
            //setting the number of iteration
            r_model_part.GetProcessInfo()[NL_ITERATION_NUMBER] = iteration_number;

            p_scheme->InitializeNonLinIteration(r_model_part, rA, rDx, rb);
            mpConvergenceCriteria->InitializeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

            is_converged = mpConvergenceCriteria->PreCriteria(r_model_part, r_dof_set, rA, rDx, rb);

            //call the linear system solver to find the correction mDx for the
            //it is not called if there is no system to solve
            if (SparseSpaceType::Size(rDx) != 0)
            {
               
                TSparseSpace::SetToZero(rDx);
                TSparseSpace::SetToZero(rb);

                p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
               
            }
            else
            {
                KRATOS_WARNING("NO DOFS") << "ATTENTION: no free DOFs!! " << std::endl;
            }

            // Debugging info
            EchoInfo(iteration_number);

            // Updating the results stored in the database
            this->UpdateSolutionStepValue(rDx, rDx_tot);

            p_scheme->FinalizeNonLinIteration(r_model_part, rA, rDx, rb);
            mpConvergenceCriteria->FinalizeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

            if (mStoreNonconvergedSolutionsFlag == true){
                Vector ith;
                GetCurrentSolution(r_dof_set,ith);
                NonconvergedSolutions.push_back(ith);
            }

            residual_is_updated = false;

            if (is_converged == true)
            {
                if (mpConvergenceCriteria->GetActualizeRHSflag() == true)
                {
                    TSparseSpace::SetToZero(rb);

                    p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
                    residual_is_updated = true;
                }

                is_converged = mpConvergenceCriteria->PostCriteria(r_model_part, r_dof_set, rA, rDx, rb);
            }
        }

        if (is_converged)
        {
            this->UpdateSolutionStepDerivative(rDx_tot, r_model_part);
           
		}

        //plots a warning if the maximum number of iterations is exceeded
        if (iteration_number >= mMaxIterationNumber) {
            MaxIterationsExceeded();
        } else {
            KRATOS_INFO_IF("ResidualBasedNewtonRaphsonStrategy", this->GetEchoLevel() > 0)
                << "Convergence achieved after " << iteration_number << " / "
                << mMaxIterationNumber << " iterations" << std::endl;
        }


        //calculate reactions if required
        if (mCalculateReactionsFlag)
            p_builder_and_solver->CalculateReactions(p_scheme, r_model_part, rA, rDx_tot, rb);

        if (mStoreNonconvergedSolutionsFlag) {
            mNonconvergedSolutionsMatrix = Matrix( r_dof_set.size(), NonconvergedSolutions.size() );
            for (std::size_t i = 0; i < NonconvergedSolutions.size(); ++i) {
                block_for_each(r_dof_set, [&](const auto& r_dof) {
                    mNonconvergedSolutionsMatrix(r_dof.EquationId(), i) = NonconvergedSolutions[i](r_dof.EquationId());
                });
            }
        }

        return is_converged;
    }



    /**
     * @brief This method provides the defaults parameters to avoid conflicts between the different constructors
     * @return The default parameters
     */
    Parameters GetDefaultParameters() const override
    {
        Parameters default_parameters = Parameters(R"(
        {
            "name"                                : "newton_raphson_strategy",
            "use_old_stiffness_in_first_iteration": false,
            "max_iteration"                       : 10,
            "reform_dofs_at_each_step"            : false,
            "compute_reactions"                   : false,
            "builder_and_solver_settings"         : {},
            "convergence_criteria_settings"       : {},
            "linear_solver_settings"              : {},
            "scheme_settings"                     : {}
        })");

        // Getting base class default parameters
        const Parameters base_default_parameters = BaseType::GetDefaultParameters();
        default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
        return default_parameters;
    }

    /**
     * @brief Returns the name of the class as used in the settings (snake_case format)
     * @return The name of the class
     */
    static std::string Name()
    {
        return "newton_raphson_strategy_linear_elastic_dynamic";
    }

    ///@}
    ///@name Operators

    ///@{

    ///@}
    ///@name Operations
    ///@{

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
    std::string Info() const override
    {
        return "ResidualBasedNewtonRaphsonStrategyLinearElasticDynamic";
    }


    ///@}
    ///@name Friends
    ///@{

    ///@}

  private:
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
  protected:
    ///@name Static Member Variables
    ///@{

    ///@}
    ///@name Member Variables
    ///@{


    double mBeta = 0.25;
	double mGamma = 0.5;



    ///@}
    ///@name Private Operators
    ///@{


  

    /**
     * @brief This method prints information after reach the max number of iterations
     */
    virtual void MaxIterationsExceeded()
    {
        KRATOS_INFO_IF("ResidualBasedNewtonRaphsonStrategyLinearElasticDynamic", this->GetEchoLevel() > 0)
            << "ATTENTION: max iterations ( " << mMaxIterationNumber
            << " ) exceeded!" << std::endl;
    }

    

    void GetFirstAndSecondDerivativeVector(TSystemVectorType& rFirstDerivativeVector,
        TSystemVectorType& rSecondDerivativeVector,
        ModelPart& rModelPart, IndexType i)
    {
        block_for_each(rModelPart.Nodes(), [&rFirstDerivativeVector, &rSecondDerivativeVector,i, this](Node& rNode) {
            if (rNode.IsActive()) {
                GetDerivativesForVariable(DISPLACEMENT_X, rNode, rFirstDerivativeVector, rSecondDerivativeVector, i);
                GetDerivativesForVariable(DISPLACEMENT_Y, rNode, rFirstDerivativeVector, rSecondDerivativeVector, i);

                const std::vector<const Variable<double>*> optional_variables = {
                    &ROTATION_X, &ROTATION_Y, &ROTATION_Z, &DISPLACEMENT_Z };

                for (const auto p_variable : optional_variables) {
                    GetDerivativesForOptionalVariable(*p_variable, rNode, rFirstDerivativeVector,
                        rSecondDerivativeVector, i);
                }
            }
            });
    }

    void SetFirstAndSecondDerivativeVector(TSystemVectorType& rFirstDerivativeVector,
        TSystemVectorType& rSecondDerivativeVector,
        ModelPart& rModelPart)
    {
        block_for_each(rModelPart.Nodes(), [&rFirstDerivativeVector, &rSecondDerivativeVector, this](Node& rNode) {
            if (rNode.IsActive()) {
                SetDerivativesForVariable(DISPLACEMENT_X, rNode, rFirstDerivativeVector, rSecondDerivativeVector);
                SetDerivativesForVariable(DISPLACEMENT_Y, rNode, rFirstDerivativeVector, rSecondDerivativeVector);

                const std::vector<const Variable<double>*> optional_variables = {
                    &ROTATION_X, &ROTATION_Y, &ROTATION_Z, &DISPLACEMENT_Z };

                for (const auto p_variable : optional_variables) {
                    SetDerivativesForOptionalVariable(*p_variable, rNode, rFirstDerivativeVector,
                        rSecondDerivativeVector);
                }
            }
            });
    }

    void GetDerivativesForOptionalVariable(const Variable<double>& rVariable,
        const Node& rNode,
        TSystemVectorType& rFirstDerivativeVector,
        TSystemVectorType& rSecondDerivativeVector, IndexType i) const
    {
        if (rNode.HasDofFor(rVariable)) {
            GetDerivativesForVariable(rVariable, rNode, rFirstDerivativeVector, rSecondDerivativeVector, i);
        }
    }

    void SetDerivativesForOptionalVariable(const Variable<double>& rVariable,
        Node& rNode,
        const TSystemVectorType& rFirstDerivativeVector,
        const TSystemVectorType& rSecondDerivativeVector)
    {
        if (rNode.HasDofFor(rVariable)) {
            SetDerivativesForVariable(rVariable, rNode, rFirstDerivativeVector, rSecondDerivativeVector);
        }
    }

    void GetDerivativesForVariable(const Variable<double>& rVariable,
        const Node& rNode,
        TSystemVectorType& rFirstDerivativeVector,
        TSystemVectorType& rSecondDerivativeVector, IndexType i) const
    {
        const auto& first_derivative = rVariable.GetTimeDerivative();
        const auto& second_derivative = first_derivative.GetTimeDerivative();

        const auto equation_id = rNode.GetDof(rVariable).EquationId();
        rFirstDerivativeVector[equation_id] = rNode.FastGetSolutionStepValue(first_derivative, i);
        rSecondDerivativeVector[equation_id] = rNode.FastGetSolutionStepValue(second_derivative, i);
    }

	void SetDerivativesForVariable(const Variable<double>& rVariable,
		Node& rNode,
		const TSystemVectorType& rFirstDerivativeVector,
		const TSystemVectorType& rSecondDerivativeVector)
	{
		const auto& first_derivative = rVariable.GetTimeDerivative();
		const auto& second_derivative = first_derivative.GetTimeDerivative();

		const auto equation_id = rNode.GetDof(rVariable).EquationId();
		rNode.FastGetSolutionStepValue(first_derivative) = rFirstDerivativeVector[equation_id];
		rNode.FastGetSolutionStepValue(second_derivative) = rSecondDerivativeVector[equation_id];
	}

    void UpdateSolutionStepValue(TSystemVectorType& rDx, TSystemVectorType& rDx_tot){

		rDx_tot += rDx;

        typename TBuilderAndSolverType::Pointer p_builder_and_solver = GetBuilderAndSolver();
		DofsArrayType& rDofSet = p_builder_and_solver->GetDofSet();


        block_for_each(rDofSet, [&rDx](auto& dof) {
            if (dof.IsFree()) {
                dof.GetSolutionStepValue() += TSparseSpace::GetValue(rDx, dof.EquationId());
            }
            });

	}


    void UpdateSolutionStepDerivative( TSystemVectorType& rDx_tot, ModelPart& rModelPart) {


        typename TBuilderAndSolverType::Pointer p_builder_and_solver = GetBuilderAndSolver();
        const DofsArrayType& rDofSet = p_builder_and_solver->GetDofSet();

        TSystemVectorType& rFirstDerivativeVector = TSystemVectorType(rDofSet.size(), 0.0);
		TSystemVectorType& rSecondDerivativeVector = TSystemVectorType(rDofSet.size(), 0.0);


		const double delta_time = rModelPart.GetProcessInfo()[DELTA_TIME];


        // get values from previous time step as the derivatives are already updated in the Predict step
        this->GetFirstAndSecondDerivativeVector(rFirstDerivativeVector, rSecondDerivativeVector, rModelPart, 1);

        const TSystemVectorType& rDeltaFirstDerivativeVector = rDx_tot * (mGamma / (mBeta * delta_time)) - rFirstDerivativeVector * (mGamma/mBeta) + rSecondDerivativeVector * (delta_time * (1-mGamma / (2 * mBeta)));

		const TSystemVectorType& rDeltaSecondDerivativeVector = rDx_tot * (1 / (mBeta * delta_time * delta_time)) - rFirstDerivativeVector * (1 / (mBeta * delta_time)) - rSecondDerivativeVector * (1 / (2 * mBeta));


		rFirstDerivativeVector += rDeltaFirstDerivativeVector;
		rSecondDerivativeVector += rDeltaSecondDerivativeVector;

		this->SetFirstAndSecondDerivativeVector(rFirstDerivativeVector, rSecondDerivativeVector, rModelPart);


    }

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

    /**
     * Copy constructor.
     */

    GeoMechanicNewtonRaphsonStrategyLinearElasticDynamic(const GeoMechanicNewtonRaphsonStrategyLinearElasticDynamic&Other){};

    ///@}

}; /* Class GeoMechanicNewtonRaphsonStrategyLinearElasticDynamic */

///@}

///@name Type Definitions
///@{

///@}

} /* namespace Kratos. */