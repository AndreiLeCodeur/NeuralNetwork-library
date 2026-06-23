import numpy as np
class Parameter() :
    def __init__(self,shape) :
        self.shape=shape
        np.random.seed(42)
        self.grad=np.zeros(self.shape)
        self.size=self.grad.size
        self.data=np.random.randn(self.size).reshape(self.shape)

class Generic() : 
    def __init__(self) :
        self.list_params=[] # Liste des paramètres
        self.save=None # Objet pour sauver des infos dans le forward
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        self.save=np.copy(x)
        y=None
        return y
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params
        hat_x=None
        return hat_x
    
class Arctan() : 
    def __init__(self) :
        self.list_params=[] # Liste des paramètres
        self.save=None # Objet pour sauver des infos dans le forward
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        self.save=np.copy(x)
        y=np.arctan(x)
        return y
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params
        hat_x=(1/(1+(self.save**2)))*hat_y
        return hat_x
    
class Sigmoid() : 
    def __init__(self) :
        self.list_params=[] # Liste des paramètres
        self.save=None # Objet pour sauver des infos dans le forward

    def __s(self, x):
        return 1/(1+np.exp(-1*x))
    
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        self.save=np.copy(x)
        y=self.__s(x)
        return y
    
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params
        x = self.save
        s = self.__s(x)
        hat_x= s*(1-s)*hat_y
        return hat_x
    
class Dense() : 
    def __init__(self, nb_entree, nb_sortie) :
        A = Parameter((nb_sortie, nb_entree))
        b = Parameter((nb_sortie, 1))
        self.list_params=[A,b] # Liste des paramètres
        self.save=None # Objet pour sauver des infos dans le forward
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        A,b = [p.data for p in self.list_params]
        self.save=np.copy(x)
        y=A@x + b
        return y
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params
        A,b = [p.data for p in self.list_params]
        self.list_params[0].grad = hat_y @ (self.save.T)
        self.list_params[1].grad = np.sum(hat_y, axis=1, keepdims=True)
        hat_x= A.T @ hat_y
        return hat_x
    
class Loss_L2() : 
    def __init__(self, D) :
        self.list_params=[] # Liste des paramètres
        self.D = D
        self.save=None # Objet pour sauver des infos dans le forward
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        self.save=np.copy(x)
        y=0.5 * (np.linalg.norm(x-self.D)**2)
        return y
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params
        hat_x= (self.save - self.D)*hat_y
        return hat_x
    

    
class Sequential() : 
    def __init__(self, list_layers) :
        self.list_layers = list_layers # contient toutes les couches
        self.list_params = [p for L in self.list_layers for p in L.list_params]
        self.save=None # Objet pour sauver des infos dans le forward
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        self.save=np.copy(x)
        for L in self.list_layers :
            x = L.forward(x)
        return x
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params
        i=len(self.list_params)-1
        for L in reversed(self.list_layers) :
            hat_y=L.backward(hat_y)
        
        return hat_y
    
class Softmax() : 
    def __init__(self) :
        self.list_params=[] # Liste des paramètres
        self.save=None # Objet pour sauver des infos dans le forward
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        y=np.exp(x)
        self.save=np.copy(y)/np.sum(y,axis=0)
        return y/np.sum(y,axis=0)
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params
        y = self.save
        hat_x=y*hat_y - y*np.sum(y*hat_y, axis=0)
        return hat_x
    
class KL():
    def __init__(self,O) :
        self.list_params=[] # Liste des paramètres
        self.O = O
        self.save=None # Objet pour sauver des infos dans le forward
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        self.save = np.copy(x)
        return float(np.sum(-self.O * np.log(x)))
    
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params

        return - (self.O/self.save)*hat_y