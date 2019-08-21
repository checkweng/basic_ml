from sklearn.datasets import load_iris 

import pickle 

data = load_iris()

train_df = pd.DataFrame(data.data,columns = ['A','B','C','D'])

pickle.dump(data,open('d:/data.pkl','wb')) 

data = pickle.load(open("d:/data.pkl","rb"))
