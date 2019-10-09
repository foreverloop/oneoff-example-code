"""
Exercise 1:
Write the following three methods in Python:
• Sum(a, b): Receives two vectors a and b represented as lists, and returns their sum.
• Mult(a, lambda): Receives a vector a and a scalar λ and returns λa.
• DotProduct(a, b): Receives two vectors a and b and returns the dot product a· b.
"""

# in real world I would use numpy since it is much more efficient than standard python lists
# today I will use it only to check work
import numpy as np

a = [1, 2, 3]
b = [4, 5, 6]


def vector_addition(c, d):
    return [c_i + d_i for c_i, d_i in zip(c, d)]


#print(vector_addition(a, b))


def vector_multiply_scalar(scalar, c):
    return [x * scalar for x in c]


# scale vector a by 4
#print(vector_multiply_scalar(4, a))


def vector_dot_product(c, d):
    return sum([c_i * d_i for c_i, d_i in zip(c, d)])


#print(vector_dot_product(a, b))
#print(np.dot(a, b))

# enumerate - very useful for accessing even more stuff when looping
# for idx,i in enumerate(range(20,41)):
#    print(i, idx)

# zip explanation
# for a,b in zip([1,2,3], [4,5,6]):
#    print(a,b)

"""
Exercise 2:
Write in Python a method mult(A, B), which receives two n×n matrices A and B, and returns
a new matrix C = AB. Assume that the matrices are represented as lists of lists.
"""


def mult(A, B):
    zip_b = list(zip(*B)) # create a list of tuples of first entry in each row

    #[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b)) for col_b in zip_b]

    #for each row in A
    [print(row_a) for row_a in A]

    # generate a sum of element products
    # for each row a and column b using the zip_b created earlier

    return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b))
             for col_b in zip_b] for row_a in A]


x = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
y = [[1, 2], [1, 2], [3, 4]]

#print("xy matmul", mult(x, y))

#mx = np.mat(x)
#my = np.mat(y)
#print("numpy check: ", mx * my)

"""
Exercise 3:
• What is the meaning of abT? A:(transpose of the dot product of a * b)?
• Write in Python a method transpose(A), which returns the transpose of a vector or a matrix A.
"""


def transpose(A):
    num_rows = len(A)

    if num_rows > 1: # matrix below
        zip_a = list(zip(*A))
        for item in zip_a:
            print(item)

    else: # vector below
        my_transpose = []
        
        [[my_transpose.append([item]) for item in row] for row in A]
        
        #print(my_transpose_vector)
        #for list_item in A:
        #    for item in list_item:
        #        my_transpose.append([item])
        
        return my_transpose


print(x)
print(transpose(x))
print(transpose([[1, 2, 3, 4]])) #double bracket required even for just a vector

"""
Exercise 4:
Write in Python a method isInverse(A,B), which returns True if B is the inverse of A; or False
otherwise. Again, assume that the matrices are represented as lists of lists.
Hint: You are just being asked to verify if B is the inverse, not to actually calculate the inverse.
"""

def ident(n):
    m=[[0 for x in range(n)] for y in range(n)]
    
    #for example, (0,0), (1,1) (2,2) etc... which is why we use [i][i]
    #to access that element in that row and column of the array
    for i in range(0,n):
        m[i][i] = 1
    return m

print("identity: ",ident(3))

def isInverse(A, B):
    #inverse of A given by (provided that A is just a 3 x 3 matrix, as there is no easy )
    #way to determine the dimensions without numpy,other than nrows and ncols type calcs
    print("Visual check: ",mult(A,ident(2)),B)
    
    #t
    return mult(A,ident(2)) == B

original_matrix = [[2,1],[1,1]]

print("inverse fom np to check: ",np.linalg.inv(original_matrix))

inverted_matrix = [[1,-1],[-1,2]]

print(isInverse(original_matrix,inverted_matrix))

"""
Exercise 5:
• Given two points a and b, create a method dist(a, b), which returns their Euclidean distance.
• Given a matrix A, create a method lowDist(A), that returns which pairs of rows in A has
the lowest Euclidean distance.
Again, for these exercises please assume that vectors are represented as Python lists and matrices as lists of lists.
"""


def dist(c, d):
    # two seperate lists e.g [4,6] and [9,1]?
    return np.sqrt((c[0] - d[1]) ** 2 + (d[0] - c[1]) ** 2)

print("euclidean distance" , dist([1,1],[10,20]))

#rough idea is here...but probably not efficient way to do this
def lowDist(A):
    
    #compare each row against each other row
    #but it is computationally expensive to try each iteration manually
    # say number of rows is 5
    
    print("length before: ",len(A))
    
    dist_calcs = []
    for x in A:
        
        other_options = A
        other_options.remove(x)
        
        print("length during: ",len(A))
        print(len(other_options))
        
        for option in other_options:
            dist_calcs.append(dist(x,option))
        
        
        print("minimum distance",min(dist_calcs))
    
    return 0

N = [[12,7],[10,14],[3,4],[7,12],[7,3]]
#lowDist(N)

"""
Exercise 6:
Given two vectors a and b, create a Python method cosSimilarity(a,b), which returns their
cosine similarity.
"""


def cosSimilarity(a, b):
    return 0
