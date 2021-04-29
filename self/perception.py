# input (x,y)  
    # x is a vector
    # y is flag

T=[] # test data
X=[]
Y=[]
# postive 
X.append([3,3])
X.append([4,3])
# negative point 
X.append([1,1])
Y=[1,1,-1]
T=[X, Y]
# print(T)
    # study rate
n=1 

    # w0,b0
w=[0,0]
b=0
def dot(x,y):
    if len(x)!=len(y):
        print("wrong!")
        exit()
    ans=0
    for i in range(len(x)):
        ans+=x[i]*y[i]
    return ans

def perception(T,w,b):
    X,Y=T
    print(X)
    print(Y)
    flag=0
    stop_flag=len(X)
    i=0
    while(flag<stop_flag): 
        x=X[i]
        y=Y[i]
        # print(i)
        if(-y*(dot(x,w)+b)<0):
            flag+=1
        else:
            flag=0
        while(-y*(dot(x,w)+b)>=0):
            for j in range(len(x)):
                w[j]+=y*n*x[j]
            b+=y*n*1   
            print(f"x={i+1},w={w},b={b},{w[0]}x1+{w[1]}x2+{b}")
        i=(i+1)%stop_flag
perception(T,w,b)
        
