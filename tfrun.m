clc,clear,close all;

% read data
data=py.data.data;
test=cell(data.test.images.tolist());
test=cellfun(@(x){reshape(cell2mat(cell(x)),[28,28])'},test);

% call tensorflow model
x=test{1};
img=reshape(x,[28,28]);
mnist=py.tf.MNIST();
x=x';% due to different dimention order
y=mnist.predict(x(:)');
cellfun(@double,cell(y.tolist()))