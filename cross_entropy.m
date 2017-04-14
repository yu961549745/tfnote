function z = cross_entropy(x,y)
y=expnorm(y);
z=-sum(x.*log(y));
end 
function x = expnorm(x)
x=exp(x);
x=x/sum(x);
end