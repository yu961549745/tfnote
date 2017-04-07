function y = catImages(x,cols)
n=numel(x);
rows=ceil(n/cols);
y=repmat({zeros(28,28)},[cols,rows]);
y(1:n)=x;
y=cell2mat(y');
end