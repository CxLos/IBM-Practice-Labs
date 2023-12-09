
# print("Yay week 3!")
# import statistics
# import math
 
import matplotlib.pyplot as plt
# %matplotlib inline  
# print('matplotlib: {}'.format(matplotlib.pyplot__version__))

# age = 1
# # age = 19

# if age > 18:
#     print("you can enter" )
# else:
#     print("go see Meat Loaf" )
    
# print("move on")

# album_year = 2005

# if(album_year > 1999) or (album_year < 2010):
#     print("Album was made in the 2000's")
# else:
#     print("Album was not made in the 2000's")

# dates = [1982,1980,1973]
# N = len(dates)

# for i in range(N):
#     print(dates[i])   

# dates = [1982, 1980, 1973, 2000]

# i = 0
# year = dates[0]

# while(year != 1973):    
#     print(year)
#     i = i + 1
#     year = dates[i]
    

# print("It took ", i ,"repetitions to get out of loop.")

# for x in ['A','B','C']:
#   print(x+'A')

# def add(a):
#    b = a + 1
#    c = a * 2
#    print(b)
#    print(c)
#    return(b, c)

# print(add(5))
# help(add)

# def mult(a,b):
#    c = a * b
#    return(c)
# print (mult(6,6))

# def NoWork():
#    pass
# print(NoWork)

# def printStuff(Stuff):
#    for i,s in enumerate(Stuff):
#       print("Album", i, "Rating is: ", s)

# album_ratings = [10, 9, 8]
# printStuff(album_ratings)

# album_ratings = [10.0, 8.5, 9.5, 7.0, 7.0, 9.5, 9.0, 9.5] 
# c = statistics.mean(album_ratings)
# print(c)

# def freq(strin):
    
#     words = []
    
#     words = strin.split() # or string.lower().split()
    
#     Dict = {}
    
#     for key in words:
#         Dict[key] = words.count(key)
        
#     print("The Frequency of words is:",Dict)
    
# freq("Mary had a little lamb Little lamb, little lamb Mary had a little lamb.Its fleece was white as snow And everywhere that Mary went Mary went, Mary went \
# Everywhere that Mary went The lamb was sure to go")

# using Try- except 
# try:
#     # Attempting to divide 10 by 0
#     result = 10 / 0
# except ZeroDivisionError:
#     # Handling the ZeroDivisionError and printing an error message
#     print("Error: Cannot divide by zero")
# # This line will be executed regardless of whether an exception occurred
# print("outside of try and except block")

# a = 6

# try:
#     # b = int(input("Please enter a number to divide a"))
#     b = 0
#     a = a/b
# except ZeroDivisionError:
#     print("The number you provided cant divide 1 because it is 0")
# except ValueError:
#     print("You did not provide a number")
# except:
#     print("Something went wrong")
# else:
#     print("success a=",a)

# def safe_divide(a,b):
#   try:
#     c = a/b
#     return(c)
#   except ZeroDivisionError:
#     print("Fuk is u doin")
#     return None

# safe_divide(2,0)
# try:
#   safe_divide(2,0)
#   # a = 1
#   # b = 0
#   # c = a/b
# except ZeroDivisionError:
#   print("Fuk is u doin")
#   return None
# else:
#   print("Iight it worked")

# safe_divide(2,0)

# def square(a):
#   try:
#     c = math.sqrt(a)
#     print(a, 'squared is:', {c})
#     # return(c)
#   except ValueError:
#     print("Fuk is u doin")
#   else:
#     print("Iight it worked")

# d = int(4)
# square(d)
# print(square(4))

# def complicated_function(num):

#   try:
#     a = num -5
#     c = num / a
#     print(c)
#     print(num, "divided by", "(",num, a, ")", "is:")
#   except Exception as e:
#     print("TF")
#   else:
#     print("It worked")

# d = "k"
# complicated_function(d)

class Circle(object):
    
    # Constructor
    def __init__(self, radius=3, color='blue'):
        self.radius = radius
        self.color = color 
    
    # Method
    def add_radius(self, r):
        self.radius = self.radius + r
        return(self.radius)
    
    # Method
    def drawCircle(self):
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show()  

redCircle = Circle(10, 'blue')
dir(redCircle)
redCircle.radius
redCircle