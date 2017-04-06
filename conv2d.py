'''
input:[number,height,width,in_channel]=[N,H,W,Cin]
filter:[height,width,in_channels,out_channels]=[h,w,Cin,Cout]

output[b,i,j,k]=sum_{di,dj,q} input[b,s[1]*i+di,s[2]*j+dj,q]*filter[di,dj,q,k]
'''
