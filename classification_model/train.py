import pipeline as pipe
import data_managers

titanic_data = data_managers.load_data(split= True)
print(titanic_data)