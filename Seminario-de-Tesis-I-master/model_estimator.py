import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from model_evolutive import EvolutiveSearchCV

class EstimatorSelection:
    
    # Constructor de clase
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Error en la definición de los parámetros del modelo %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.results = {}
        self.timeModel = {}
        
    # Fit del modelo    
    def fitModel(self, searchType, X, y, cv=KFold(n_splits=10), n_jobs=1, verbose=1, scoring=None, refit=False,
            population_size=50, gene_mutation_prob=0.10, gene_crossover_prob=0.5, tournament_size=3, generations_number=10):
        
        for key in self.keys:
            model = self.models[key]
            params = self.params[key]
            start = time.time()
            
            if searchType == 'Exh':
                print("Ejecutando la búsqueda exhaustiva para el modelo %s ...." % key)
                gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, verbose=verbose, scoring=scoring, refit=refit)
                self.grid_searches[key] = gs.fit(X, y)
                self.grid_searches[key] = self.grid_searches[key].cv_results_
            elif searchType == 'Rdn':
                print("Ejecutando la búsqueda aleatoria para el modelo %s ..." % key)
                gs = RandomizedSearchCV(model, params, cv=cv, n_jobs=n_jobs, verbose=verbose, scoring=scoring, refit=refit)
                self.grid_searches[key] = gs.fit(X, y)
                self.grid_searches[key] = self.grid_searches[key].cv_results_
            elif searchType == 'Evol':
                print("Ejecutando la búsqueda evolutiva para el modelo %s ..." % key)
                gs = EvolutiveSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                                   verbose=verbose, scoring=scoring, refit=refit, 
                                   population_size=population_size, 
                                   gene_mutation_prob=gene_mutation_prob, 
                                   gene_crossover_prob=gene_crossover_prob, 
                                   tournament_size=tournament_size, 
                                   generations_number=generations_number)
                gs.fit(X, y)
                self.grid_searches[key] = gs.cv_results_()
            
            else:
                print("Modelo no especificado o no válido. Seleccionar uno de los siguientes: \n \
                Exh: Búsqueda exhaustiva. \n \
                Rdn: Búsqueda aleatoria. \n \
                Evol: Búsqueda evolutiva. \n")
       
            end = time.time()
            self.timeModel[key] = [end-start]
        
    # Score del modelo
    def scoreModel(self):
        for estimator in self.keys:
            d = {}
            d['.Accuracy'] = self.grid_searches[estimator]['mean_test_score']
            d['.Error'] = self.grid_searches[estimator]['std_test_score']
            df1 = pd.DataFrame(d)
            d = self.grid_searches[estimator]['params']
            df2 = pd.DataFrame(list(d))
            self.results[estimator] = pd.concat([df1, df2], axis=1).fillna(' ')
            
        return pd.concat(self.results).fillna(' ')