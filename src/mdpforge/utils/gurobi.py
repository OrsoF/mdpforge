def gurobi_model_creation(*, threads: int = 1, memory_limit_gb: float = 16.0):
    from gurobipy import GRB, Model

    model = Model("MDP")
    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.Threads, threads)
    model.setParam("LogToConsole", 0)
    model.setParam("MemLimit", memory_limit_gb)
    return model
