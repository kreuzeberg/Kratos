# ==============================================================================
#  KratosOptimizationApplication
#
#  License:         BSD License
#                   license: OptimizationApplication/license.txt
#
#  Main authors:    Suneth Warnakulasuriya
#
# ==============================================================================

from importlib import import_module
from pathlib import Path

import KratosMultiphysics as Kratos
from KratosMultiphysics.model_parameters_factory import KratosModelParametersFactory
from KratosMultiphysics.OptimizationApplication.execution_policies.execution_policy import ExecutionPolicy
from KratosMultiphysics.OptimizationApplication.utilities.logger_utilities import FileLogger
from KratosMultiphysics.OptimizationApplication.optimization_routine import OptimizationRoutine

def RetrieveObject(model: Kratos.Model, parameters: Kratos.Parameters, object_type = object):
    default_settings = Kratos.Parameters("""{
        "module"  : "",
        "type"    : "",
        "settings": {}
    }""")
    parameters.ValidateAndAssignDefaults(default_settings)

    class_name = parameters["type"].GetString()
    python_file_name = ''.join(['_' + c.lower() if c.isupper() else c for c in class_name]).lstrip('_')

    module_prefix = parameters["module"].GetString()
    module_name = module_prefix
    if module_name != "":
        module_name += "."
    module_name += python_file_name

    try:
        module = import_module(module_name)
        retrieved_object = getattr(module, class_name)(model, parameters["settings"])
        if not isinstance(retrieved_object, object_type):
            raise ImportError(f"The retrieved object is of type \"{retrieved_object.__class__.__name__}\" which is not of the required type \"{object_type.__name__}\".")
        else:
            return retrieved_object
    except AttributeError:
        raise AttributeError(f"The python file \"{str(Path(module.__file__))}\" does not have a class named \"{class_name}\".")
    except (ImportError, ModuleNotFoundError):
        if module_prefix != "":
            try:
                module = import_module(module_prefix)
            except (ImportError, ModuleNotFoundError):
                raise RuntimeError(f"The module \"{module_prefix}\" cannot be imported.")
            if hasattr(module, "__path__"):
                list_of_available_options = []
                for item in Path(module.__path__[0]).iterdir():
                    if item.is_file() and item.name.endswith(".py"):
                        try:
                            current_module = import_module(module_prefix + "." + item.name[:-3])
                            current_class_name = "".join([c.title() for c in item.name[:-3].split("_")])
                            if hasattr(current_module, current_class_name):
                                if issubclass(getattr(current_module, current_class_name), object_type):
                                    list_of_available_options.append(current_class_name)
                        except:
                            pass
                raise RuntimeError(f"Given type with name \"{class_name}\" is not found. Followings are available options:\n\t" + "\n\t".join(list_of_available_options))
            else:
                raise RuntimeError(f"Invalid module found under \"{module_prefix}\".")
        else:
            raise RuntimeError("Invalid \"module\" and/or \"type\" fields.")

class ExecutionPolicyWrapper(OptimizationRoutine):
    def __init__(self, model: Kratos.Model, parameters: Kratos.Parameters):
        super().__init__(model, parameters)

        default_parameters = Kratos.Parameters("""{
            "name"                     : "",
            "execution_policy_settings": {
                "module"  : "KratosMultiphysics.OptimizationApplication.execution_policies",
                "type"    : "PleaseProvideClassName",
                "settings": {}
            },
            "pre_operations"           : [],
            "post_operations"          : [],
            "log_in_file"              : false,
            "echo_level"               : 0
        }""")
        parameters.ValidateAndAssignDefaults(default_parameters)

        self.__log_in_file = parameters["log_in_file"].GetBool()
        self.__name = parameters["name"].GetString()
        self.__echo_level = parameters["echo_level"].GetInt()
        self.__is_executed = False

        if self.__name == "":
            raise RuntimeError(f"Execution policies should be given a non-empty name. Followings are the corresponding execution policy settings:\n{parameters}")

        factory = KratosModelParametersFactory(model)

        # create operations
        self.__list_of_pre_operations = factory.ConstructListOfItems(parameters["pre_operations"])
        self.__list_of_post_operations = factory.ConstructListOfItems(parameters["post_operations"])

        # create execution policy
        parameters["execution_policy_settings"].ValidateAndAssignDefaults(default_parameters["execution_policy_settings"])
        self.__execution_policy = RetrieveObject(model, parameters["execution_policy_settings"], ExecutionPolicy)

    def Initialize(self):
        self.__execution_policy.Initialize()

    def InitializeSolutionStep(self):
        self.__is_executed = False
        self.__execution_policy.InitializeSolutionStep()

    def Execute(self):
        if not self.__is_executed:
            if self.__log_in_file:
                with FileLogger(self.__name + ".log"):
                    self.__ExecuteWithoutFileLogger()
            else:
                self.__ExecuteWithoutFileLogger()
        else:
            if self.__echo_level > 1:
                Kratos.Logger.PrintInfo(self.__class__.__name__, f"Skipping execution of {self.__name} because it has already executed for this iteration.")

    def FinalizeSolutionStep(self):
        self.__execution_policy.FinalizeSolutionStep()

    def Finalize(self):
        self.__execution_policy.Finalize()

    def GetExecutionPolicy(self) -> ExecutionPolicy:
        return self.__execution_policy

    def GetName(self):
        return self.__name

    def __ExecuteWithoutFileLogger(self):
        self.__RunPreExecutionPolicyOperations()
        self.__execution_policy.Execute()
        self.__RunPostExecutionPolicyOperations()

    def __RunPreExecutionPolicyOperations(self):
        for operation in self.__list_of_pre_operations:
            operation.Execute()

    def __RunPostExecutionPolicyOperations(self):
        for operation in self.__list_of_post_operations:
            operation.Execute()
