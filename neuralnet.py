import numpy as np
import sys


#Linear Module
# linear forward 
def linearForward(x,W):
    out = np.matmul(W,x)
    return(out)
#linear backward
def linearBackward(g,W,x):
    g = g.reshape(-1,1)
    x = x.reshape(-1,1)
    g_x = np.matmul(W.T,g)
    g_W = np.matmul(g,x.T)
    return g_x,g_W

#sigmoid modules
# Sigmoid Forward
def sigmoidForward(x):
    z = np.exp(x)
    z = z/(1+z)
    return(z)
#sigmoid backward
def sigmoidBackward(g,z):
    z = z.reshape(-1,1)
    dl_da = g*z*(1-z)
    return(dl_da)

#softmax modules
#softmax forward
def softmaxForward(x):
    numer = np.exp(x)
    denom = sum(numer)
    yhat = numer/denom
    return yhat
def softmaxBackward(y,yhat):
    dl_db = yhat - y
    return(dl_db)

#cross entropy modules
def crossEntropyForward(y,yhat):
    ylog = np.log(yhat)
    prod = -y*ylog
    j = sum(prod)
    return j
def crossEntropyBackward(y,yhat):
    dl_dy = -y/yhat
    return(dl_dy)

#forward module
def NNForward(xi,yi,alpha,beta):
    a = linearForward(xi,alpha)
    z = sigmoidForward(a)
    z_bias = np.insert(z,0,1)
    b = linearForward(z_bias,beta)
    yhat = softmaxForward(b)
    J = crossEntropyForward(yi,yhat)
    o = np.array([a,z,z_bias,b,yhat,J])
    return o
#backward module
def NNBackward(xi,yi,alpha,beta,o):
    a,z,z_bias,b,yhat,J = o
    gy = crossEntropyBackward(yi,yhat)
    gb = softmaxBackward(yi,yhat)
    gz,g_beta = linearBackward(gb,beta,z_bias)
    ga = sigmoidBackward(gz,z)
    gx,g_alpha = linearBackward(ga,alpha,xi)
    return g_alpha, g_beta

#mean crossentropy
def meanCrossEntropy(x,y,alpha,beta):
    N = x.shape[0]
    J = 0
    for i in range(N):
        *_,j = NNForward(x[i],y[i],alpha,beta)
        J += j
    return J/N

## Training
def SDG(x_train,y_train,x_valid,y_valid,hidden_units,num_epoch,rate,init_flag):
    # initializing the parameters 
    j = hidden_units  
    m = x_train.shape[1] #(Value of m counts bias term also because x_train alrready has bias term)
    k = np.unique(labels_train).size
    if init_flag == 1:
        alpha = np.random.uniform(low = -0.1, high = 0.1, size = (j,m-1))
        alpha = np.insert(alpha, 0, 0, axis=1) #bias term
        beta = np.random.uniform(low = -0.1, high = 0.1, size = (k,j))
        beta_bias = np.insert(beta, 0, 0, axis=1) # including bias term
    else:
        alpha = np.full((j,m),0) # includes bias term directly
        
        beta =  np.full((k,j),0)
        beta_bias = np.insert(beta, 0, 0, axis=1) # including bias term
        
    cross_entropy = []    
    for e in range(num_epoch):
        for i in range(x_train.shape[0]):
            o = NNForward(x_train[i],y_train[i],alpha,beta_bias)
            g_alpha,g_beta = NNBackward(x_train[i],y_train[i],alpha,beta,o) # gradients
            alpha = alpha - rate*g_alpha
            beta_bias = beta_bias - rate*g_beta
            beta = np.delete(beta_bias, 0, 1)
        
        
        J_train = meanCrossEntropy(x_train,y_train,alpha,beta_bias)
        cross_entropy.append((e,'crossentropy(train)',J_train))
        J_test = meanCrossEntropy(x_valid,y_valid,alpha,beta_bias)
        cross_entropy.append((e,'crossentropy(test)',J_test))
        
    return alpha,beta_bias,cross_entropy

# prediction modules
def predict(x,y,alpha,beta):
    N = x.shape[0]
    y_predicted = []
    for i in range(N):
        *_,yhat,j = NNForward(x[i],y[i],alpha,beta)
        y_predicted.append(np.argmax(yhat))
    return(y_predicted)

def errorRate(labels,predicted_labels):
    N = labels.shape[0]
    error = 0
    for i in range(N):
        if (labels[i] != predicted_labels[i]):
            error += 1
    return error/N

# data modules
def read_file(filename):
    f = open(filename,'r')
    data = [x for x in (line.strip().split(",") for line in f)]
    f.close()
    data = np.array(data)
    data = data.astype(int)
    y = data[:,0]
    x = data[:,1:]
    return data,y,x

def one_hot_encoding(labels):
    classes = np.unique(labels).size
    examples = labels.size
    y = np.full((examples,classes),0)
    for i in range(examples):
        y[i,labels[i]] = 1
    return y


######## unpacking the arguments ############
train_input,test_input,train_out,test_out,metrics_out,num_epoch,hidden_units, init_flag,rate = sys.argv[1:]
num_epoch = int(num_epoch)
hidden_units = int(hidden_units)
init_flag = int(init_flag)
rate = float(rate)

############### reading in the data ##############
data_train,labels_train,x_train = read_file(train_input)
y_train = one_hot_encoding(labels_train)
x_train = np.insert(x_train,0,1,axis = 1) #adding bias term

data_test,labels_test,x_test = read_file(test_input)
y_test = one_hot_encoding(labels_test)
x_test = np.insert(x_test,0,1,axis = 1) #adding bias term

########## training and prediction #############

alpha, beta_bias, cross_entropy = SDG(x_train,y_train,x_test,y_test,hidden_units,num_epoch,rate,init_flag)

predicted_labels_train = predict(x_train,y_train,alpha,beta_bias)
error_train = errorRate(labels_train,predicted_labels_train)

predicted_labels_test = predict(x_test,y_test,alpha,beta_bias)
error_test = errorRate(labels_test,predicted_labels_test)

################## output #######################

prediction_train = [str(x)+'\n' for x in predicted_labels_train]
prediction_test = [str(x)+'\n' for x in predicted_labels_test]
open(train_out,'w+').writelines(prediction_train)
open(test_out,'w+').writelines(prediction_test)
ce = ['epoch='+str(x)+' '+y+': '+str(z)+'\n' for x,y,z in cross_entropy]
open(metrics_out,'w+').writelines(ce)
open(metrics_out,'a').write('error(train): '+str(error_train)+'\n'+'error(test): '+str(error_test)+'\n')










