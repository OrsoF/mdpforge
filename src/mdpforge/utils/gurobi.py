def gurobi_model_creation():
    from gurobipy import GRB, Model

    model = Model("MDP")
    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam("LogToConsole", 0)
    model.setParam("MemLimit", 16)
    return model
