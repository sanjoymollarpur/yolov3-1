import torch

a=[1,2,3,4,5]
b=[10,11,12,13,14]
a=torch.tensor(a)
b=torch.tensor(b)
print(a)
print(torch.div(b,a))

f=torch.div(a,(torch.add(a,b)))
print(f)



