import numpy as np
import matplotlib.pyplot as plt


plt.style.available

"""training the model"""

x_train = np.array([1.0,2.0])
y_train= np.array([300.0,500.0])

print(f"The input data(sq feet) is: {x_train}")
print(f"The target data(cost) is: {y_train}")
m= len(x_train)
print(f"the number of training examples are:{m} ")

"""for the i no of terms"""


i= 0

for i in range(2):
    x_i = x_train[i]
    y_i = y_train[i]

    print(f"(x)^{i},(y)^{i} = {x_i},{y_i}")

"""ploting the data"""


plt.scatter(x_train, y_train, marker ='x', c='y')
plt.title("Housing prices")
plt.ylabel("prices in 1000 dolalrs")
plt.xlabel("size in sq feet(1000)")
plt.show()



"""now since this is linear regession we work fo rthe values of w and b for the equation of line"""
"""We know that f(x)wb = wx +b and that w and bare the perimeters or the weights
w is the slope
b is the y intercept

"""

w= 200.0
b= 100.0

print(f"w : {w}")
print(f"b : {b}")

"""We can manually type for each M , but if the data set is large we will need a for loop
Here we predict the y value/ target
"""

def model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

"""now we call it"""

fx_wb = model_output(x_train,w,b)

plt.plot(x_train, fx_wb, label = 'the prediction', c ='g' )
plt.scatter(x_train, y_train, marker='x', label = 'the actual value', c='b')


plt.ylabel("cost in 1000 dollars")
plt.title("housing prices")
plt.xlabel("size in 1000 sq feet")

plt.legend()
plt.show()

"""now that the model si complete we can go with some other value of prediciton"""

w= 200
b= 100
x_i= 1.5 
cost_1500sqft = w * x_i + b
print(f"the cost of the house of 1200 sqft is : {cost_1500sqft} thousand dollars")


