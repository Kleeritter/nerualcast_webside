import optuna,optuna_dashboard

def objective(trial:optuna.Trial):
    x=trial.suggest_float("x",-10,10)
    return x**2-4*x+4

study=optuna.create_study()
study.optimize(objective,n_trials=100)